"""
DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference
PyTorch実装コード

論文: Xin et al., "DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference", ACL 2020
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertLayer
from typing import Optional, Tuple, List


class BertHighway(nn.Module):
    """
    BERTの各層の後に配置される「出口（highway）」モジュール。
    各層の出力から分類と早期終了の判断を行う。
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        # プーリング層（[CLS]トークンの出力を取得）
        self.pooler = BertPooler(config)

        # 分類器（2層のMLP）
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
        )

        # 早期終了の判断に使うentropy閾値（-1は早期終了なし）
        self.early_exit_entropy = -1

    def forward(self, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_outputs: [batch_size, seq_length, hidden_size]

        Returns:
            logits: [batch_size, num_labels]
            pooled_output: [batch_size, hidden_size]
        """
        # [CLS]トークンをプーリング
        pooled_output = self.pooler(encoder_outputs)

        # 分類
        logits = self.classifier(pooled_output)

        return logits, pooled_output

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        出力分布のentropyを計算（早期終了の判断に使用）

        Args:
            logits: [batch_size, num_labels]

        Returns:
            entropy: [batch_size]
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy


class DeeBertEncoder(nn.Module):
    """
    BERTエンコーダの各層の後にhighway出口を追加したエンコーダ
    """
    def __init__(self, config: BertConfig, bert_encoder: BertEncoder):
        super().__init__()
        self.config = config

        # 元のBERTエンコーダの層をコピー
        self.layer = bert_encoder.layer

        # 各層の後にhighway出口を追加
        self.highway = nn.ModuleList([
            BertHighway(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        highway_exit: Optional[int] = None,
    ) -> Tuple:
        """
        Args:
            hidden_states: [batch_size, seq_length, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_length]
            highway_exit: 特定の出口で停止する場合の層番号（訓練時）

        Returns:
            最終的な隠れ状態、highwayの出力リストなど
        """
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_highway_outputs = []  # 各highwayの出力を保存

        extended_attention_mask = attention_mask

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            # BERTの層を通す
            layer_outputs = layer_module(
                hidden_states,
                extended_attention_mask,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            # 各層の後にhighway出口を通す
            highway_logits, highway_pooled = self.highway[i](hidden_states)
            all_highway_outputs.append({
                'logits': highway_logits,
                'pooled_output': highway_pooled,
                'layer_index': i,
            })

            # 訓練時に特定の出口で停止する場合
            if highway_exit is not None and i == highway_exit:
                break

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_highway_outputs] if v is not None)

        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'highway_outputs': all_highway_outputs,
            'attentions': all_attentions,
        }


class DeeBertForSequenceClassification(BertPreTrainedModel):
    """
    DeeBERT: 各層早期終了出口を持つBERT分類モデル
    """
    def __init__(self, config: BertConfig, bert_model: Optional[BertModel] = None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # BERT本体
        if bert_model is not None:
            self.bert = bert_model
        else:
            self.bert = BertModel(config)

        # 元のエンコーダをDeeBERTエンコーダに置き換え
        self.bert.encoder = DeeBertEncoder(config, self.bert.encoder)

        # 最終層の分類器（通常のBERTと同じ）
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 早期終了のentropy閾値（推論時に使用）
        self.early_exit_entropy = -1

        # 重みの初期化
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = True,
    ) -> Tuple:
        """
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            token_type_ids: [batch_size, seq_length]
            labels: [batch_size]（訓練時）
            training: Trueなら全ての出口で損失計算、Falseなら早期終了

        Returns:
            損失、logits、出口情報など
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 埋め込み層
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # エンコーダ（DeeBERT版）
        encoder_outputs = self.bert.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        highway_outputs = encoder_outputs['highway_outputs']

        if training:
            # ===== 訓練モード =====
            # 全てのhighway出口と最終層で損失を計算
            total_loss = 0
            all_logits = []

            # 各highway出口のlogits
            for highway_out in highway_outputs:
                logits = highway_out['logits']
                all_logits.append(logits)

                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    total_loss += loss

            # 最終層のlogits
            pooled_output = self.bert.pooler(encoder_outputs['last_hidden_state'])
            pooled_output = self.dropout(pooled_output)
            final_logits = self.classifier(pooled_output)
            all_logits.append(final_logits)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                final_loss = loss_fct(final_logits.view(-1, self.num_labels), labels.view(-1))
                total_loss += final_loss

            return total_loss, all_logits, highway_outputs

        else:
            # ===== 推論モード（早期終了）=====
            exited_layer = len(highway_outputs)  # デフォルトは最終層

            for i, highway_out in enumerate(highway_outputs):
                logits = highway_out['logits']

                # entropyが閾値以下なら早期終了
                if self.early_exit_entropy >= 0:
                    highway_module = self.bert.encoder.highway[i]
                    entropy = highway_module.compute_entropy(logits)

                    if torch.all(entropy < self.early_exit_entropy):
                        exited_layer = i + 1
                        final_logits = logits
                        break
            else:
                # どのhighwayでも終了しなかった場合、最終層を使用
                pooled_output = self.bert.pooler(encoder_outputs['last_hidden_state'])
                pooled_output = self.dropout(pooled_output)
                final_logits = self.classifier(pooled_output)

            return final_logits, exited_layer


# ========== 使用例 ==========

def create_deebert_from_bert(bert_model_name: str = 'bert-base-uncased', num_labels: int = 2):
    """
    既存のBERTモデルからDeeBERTを作成
    """
    from transformers import BertForSequenceClassification

    # 通常のBERT分類モデルを読み込み
    bert_cls = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)
    config = bert_cls.config

    # DeeBERTモデルを作成
    deebert = DeeBertForSequenceClassification(config, bert_model=bert_cls.bert)

    # 最終層の分類器の重みをコピー
    deebert.classifier.weight.data = bert_cls.classifier.weight.data.clone()
    deebert.classifier.bias.data = bert_cls.classifier.bias.data.clone()

    return deebert


def train_step(model, batch, optimizer):
    """
    DeeBERTの訓練ステップ
    全ての出口で損失を計算して合計を逆伝播
    """
    model.train()
    optimizer.zero_grad()

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    loss, all_logits, highway_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        training=True,
    )

    loss.backward()
    optimizer.step()

    return loss.item()


def inference_with_early_exit(model, batch, entropy_threshold: float = 0.5):
    """
    早期終了を使った推論
    """
    model.eval()
    model.early_exit_entropy = entropy_threshold

    with torch.no_grad():
        logits, exited_layer = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            training=False,
        )

    predictions = torch.argmax(logits, dim=-1)
    return predictions, exited_layer


# ========== 簡易テスト ==========
if __name__ == "__main__":
    # モデルの作成
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=2,
    )

    model = DeeBertForSequenceClassification(config)

    # ダミー入力
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, config.num_labels, (batch_size,))

    # 訓練モード
    print("=== 訓練モード ===")
    model.train()
    loss, all_logits, highway_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        training=True,
    )
    print(f"損失: {loss.item():.4f}")
    print(f"出口の数: {len(highway_outputs)} (各層 + 最終層)")
    print(f"各出口のlogits shape: {[logits.shape for logits in all_logits]}")

    # 推論モード（早期終了なし）
    print("\n=== 推論モード（早期終了なし）===")
    model.eval()
    logits, exited_layer = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        training=False,
    )
    print(f"出口層: {exited_layer} / {config.num_hidden_layers}")
    print(f"予測logits shape: {logits.shape}")

    # 推論モード（早期終了あり）
    print("\n=== 推論モード（早期終了あり、entropy閾値=0.5）===")
    model.early_exit_entropy = 0.5
    logits, exited_layer = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        training=False,
    )
    print(f"出口層: {exited_layer} / {config.num_hidden_layers}")
    print(f"予測logits shape: {logits.shape}")
