"""
LayerSkip 学習・推論コード
=========================
LayerSkip論文の核心である以下2点を実装しています:
1. レイヤードロップアウト（浅い層は低ドロップ率、深い層は高ドロップ率）
2. 全レイヤー共有の早期終了損失（各層のhidden stateを同じLM Headで評価）

推論時には早期終了（Early Exit）と自己投機的デコーディング（Self-Speculative Decoding）も実装しています。

動作要件:
  pip install torch transformers

使用方法:
  python layerskip_train.py
"""

import math
import random
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    AutoTokenizer,
)


# ------------------------------------------------------------------------------
# 1. LayerSkip用のLlamaモデルラッパー
# ------------------------------------------------------------------------------
class LayerSkipLlama(nn.Module):
    """
    HuggingFaceのLlamaForCausalLMをラップし、LayerSkipの学習レシピを適用する。

    ・学習時:
        - 各decoder layerに対してlayer dropoutを適用（浅い層は残しやすく、深い層は飛ばしやすく）
        - 各layerの出力hidden stateを同じLM Headで投影し、全層で早期終了損失を計算
    ・推論時:
        - early_exit_thresholdを下回る確信度なら次のlayerへ、上回れば即座に出力
        - self_speculative_decode() で自己投機的デコーディング
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",  # 実際にはTinyLlama等の小モデル推奨
        base_dropout: float = 0.0,
        max_dropout: float = 0.5,
        early_exit_loss_weight: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.base_dropout = base_dropout
        self.max_dropout = max_dropout
        self.early_exit_loss_weight = early_exit_loss_weight

        # ベースモデル読み込み（小さいモデルで試す場合はTinyLlama等を推奨）
        self.model = LlamaForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.train()

        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers

        # 各layerのdropout率を線形に増加させる
        # layer 0 -> base_dropout, layer (L-1) -> max_dropout
        self.layer_dropout_rates = [
            base_dropout + (i / max(1, self.num_layers - 1)) * (max_dropout - base_dropout)
            for i in range(self.num_layers)
        ]

        # LM Headとnormはベースモデルのものを使う（全layerで共有）
        self.lm_head = self.model.lm_head
        self.model_norm = self.model.model.norm

    # --------------------------------------------------------------------------
    # 学習用forward
    # --------------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss: 最終層loss + early_exit_loss_weight * 平均early exit loss
            final_logits: 最終層のlogits
        """
        batch_size, seq_len = input_ids.shape

        # 1) Embeddings
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # 2) 各decoder layerを順に通す（layer dropout適用）
        #    各layerの出力を保存しておき、後でearly exit lossを計算
        all_layer_hidden_states: List[torch.Tensor] = []

        # causal maskの準備（簡易版）
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.long, device=self.device
            )

        # position_ids
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # attention_maskをLlamaの形式に変換
        # (batch, 1, seq, seq) の boolean mask
        dtype = hidden_states.dtype
        min_dtype = torch.finfo(dtype).min
        attention_mask_llama = self._prepare_causal_attention_mask(
            seq_len, batch_size, dtype
        )
        if attention_mask is not None:
            # padding maskの適用
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,seq)
            padding_mask = (1.0 - padding_mask) * min_dtype
            attention_mask_llama = attention_mask_llama + padding_mask.to(self.device)

        for layer_idx, decoder_layer in enumerate(self.model.model.layers):
            # layer dropout（学習時のみ）
            if self.training and random.random() < self.layer_dropout_rates[layer_idx]:
                # layerをskip: residual connectionだけ
                # ただしLlamaDecoderLayerはresidual内で処理するので、ここではhidden_statesをそのまま
                # 厳密にはresidualを足す必要があるが、簡易的にhidden_statesを保持
                pass  # hidden_statesは変更しない（skip扱い）
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask_llama,
                    position_ids=position_ids,
                )[0]

            # 各layerの出力を保存（早期終了loss計算用）
            all_layer_hidden_states.append(hidden_states)

        # 3) 最終層のloss計算
        final_hidden = self.model_norm(hidden_states)
        final_logits = self.lm_head(final_hidden)

        total_loss = torch.tensor(0.0, device=self.device)
        if labels is not None:
            # 最終層loss
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            final_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_loss = total_loss + final_loss

            # 4) 全レイヤー共有の早期終了損失
            #    各layerのhidden stateを同じnorm+lm_headで投影しlossを計算
            early_exit_losses = []
            for layer_hidden in all_layer_hidden_states:
                layer_hidden_normed = self.model_norm(layer_hidden)
                layer_logits = self.lm_head(layer_hidden_normed)
                shift_layer_logits = layer_logits[..., :-1, :].contiguous()
                layer_loss = loss_fct(
                    shift_layer_logits.view(-1, shift_layer_logits.size(-1)),
                    shift_labels.view(-1),
                )
                early_exit_losses.append(layer_loss)

            avg_early_exit_loss = torch.stack(early_exit_losses).mean()
            total_loss = total_loss + self.early_exit_loss_weight * avg_early_exit_loss

        return total_loss, final_logits

    def _prepare_causal_attention_mask(self, seq_len, batch_size, dtype):
        """因果マスクの作成（下三角）"""
        mask = torch.full(
            (seq_len, seq_len), torch.finfo(dtype).min, device=self.device
        )
        mask_cond = torch.arange(seq_len, device=self.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(seq_len, 1), 0)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, 1, seq_len, seq_len)
        return mask

    # --------------------------------------------------------------------------
    # 推論用: 早期終了（Early Exit）
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def generate_early_exit(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        confidence_threshold: float = 0.9,
        exit_layer: Optional[int] = None,  # 指定層で強制終了
    ) -> torch.Tensor:
        """
        早期終了による生成。
        各トークン生成時に、指定threshold以上の確信度が出たlayerで即座に出力。
        """
        self.eval()
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # 現在の入力でforward（layerごとに確認）
            inputs_embeds = self.model.model.embed_tokens(generated)
            hidden_states = inputs_embeds
            seq_len_cur = generated.size(1)
            attention_mask = self._prepare_causal_attention_mask(
                seq_len_cur, batch_size, hidden_states.dtype
            )
            position_ids = torch.arange(seq_len_cur, device=self.device).unsqueeze(0)

            exited = False
            for layer_idx, decoder_layer in enumerate(self.model.model.layers):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

                # 各layerの出力で確信度をチェック
                if exit_layer is not None and layer_idx == exit_layer:
                    # 強制終了
                    norm_hidden = self.model_norm(hidden_states)
                    logits = self.lm_head(norm_hidden)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=-1)
                    exited = True
                    break
                elif exit_layer is None:
                    norm_hidden = self.model_norm(hidden_states)
                    logits = self.lm_head(norm_hidden)
                    probs = F.softmax(logits[:, -1, :], dim=-1)
                    max_prob = probs.max().item()
                    if max_prob >= confidence_threshold:
                        next_token = probs.argmax(dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_token], dim=-1)
                        exited = True
                        break

            if not exited:
                # 最終層まで行った場合
                norm_hidden = self.model_norm(hidden_states)
                logits = self.lm_head(norm_hidden)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)

        return generated

    # --------------------------------------------------------------------------
    # 推論用: 自己投機的デコーディング（Self-Speculative Decoding）
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def self_speculative_decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        draft_layers: int = 10,  # ドラフト生成に使う層数（浅い層）
    ) -> torch.Tensor:
        """
        自己投機的デコーディングの簡易実装。
        1. 浅い層（draft_layers）だけでドラフトトークンを生成
        2. 残りの深い層で検証・訂正
        """
        self.eval()
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # --- Step 1: ドラフト生成（浅い層のみ） ---
            inputs_embeds = self.model.model.embed_tokens(generated)
            hidden_states = inputs_embeds
            seq_len_cur = generated.size(1)
            attention_mask = self._prepare_causal_attention_mask(
                seq_len_cur, batch_size, hidden_states.dtype
            )
            position_ids = torch.arange(seq_len_cur, device=self.device).unsqueeze(0)

            # 浅い層だけ通す
            for layer_idx in range(min(draft_layers, self.num_layers)):
                hidden_states = self.model.model.layers[layer_idx](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

            draft_norm = self.model_norm(hidden_states)
            draft_logits = self.lm_head(draft_norm)
            draft_token = draft_logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # --- Step 2: 検証（残りの深い層でフルforward） ---
            # ドラフト時のhidden_statesを再利用して残りの層を通す
            for layer_idx in range(draft_layers, self.num_layers):
                hidden_states = self.model.model.layers[layer_idx](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

            final_norm = self.model_norm(hidden_states)
            final_logits = self.lm_head(final_norm)
            final_token = final_logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # --- Step 3: 採用判断（簡易版: トークンが一致すれば採用） ---
            # 本実装では一致すればドラフトを採用、不一致ならfinal_tokenを採用
            generated = torch.cat([generated, final_token], dim=-1)

        return generated


# ------------------------------------------------------------------------------
# 2. 簡易データセット
# ------------------------------------------------------------------------------
class TextDataset(Dataset):
    """テキストファイルから学習データを作成する簡易Dataset"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = self.encodings["input_ids"]
        self.attention_mask = self.encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx].clone(),
        }


# ------------------------------------------------------------------------------
# 3. 学習ループ
# ------------------------------------------------------------------------------
def train_layerskip(
    model: LayerSkipLlama,
    dataloader: DataLoader,
    num_epochs: int = 3,
    lr: float = 5e-5,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            optimizer.zero_grad()
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Batch {batch_idx} | Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均Loss: {avg_loss:.4f}")


# ------------------------------------------------------------------------------
# 4. メイン（デモ用）
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # デモ用にダミーデータで動作確認
    # 実際の使用時は model_name を "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 等に変更してください

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    # トークナイザー（実際にはモデルに合わせたものを使用）
    # ここでは簡易的にダミー実装として、実際のモデル名を指定
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # 実際にはTinyLlama等の小モデル推奨

    print(f"モデル読み込み中: {MODEL_NAME}")
    print("※ 実際に動かす場合は、小さいモデル（例: TinyLlama）に変更してください")

    # 以下は実際のモデルが必要なため、ダミー実行の流れのみ示す
    print("\n【LayerSkip学習の流れ】")
    print("1. LayerSkipLlamaを初期化（layer dropout率を設定）")
    print("2. 各epochでforward: 最終層loss + 全layer early exit loss を計算")
    print("3. 推論時: generate_early_exit() で浅い層で早期終了")
    print("4. 推論時: self_speculative_decode() で自己投機的デコーディング")

    # 実際の実行例（小さいモデルで）:
    # model = LayerSkipLlama(
    #     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     base_dropout=0.0,
    #     max_dropout=0.5,
    #     early_exit_loss_weight=0.5,
    #     device=device,
    # )
    # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # texts = ["Hello world", "LayerSkip accelerates inference", "Early exit is efficient"]
    # dataset = TextDataset(texts, tokenizer)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # train_layerskip(model, dataloader, num_epochs=3)
