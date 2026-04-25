import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AdvancedMultiHeadAttention(nn.Module):
    """
    高度なKVキャッシュ付きMultiHeadAttention
    - バッチ対応のKVキャッシュ
    - 因果マスク＋パディングマスク
    - スライディングウィンドウ
    - 事前割り当て（preallocated cache）
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_batch_size: int = 1,
        max_seq_len: int = 1024,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.window_size = window_size  # スライディングウィンドウサイズ（Noneなら無効）

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # 事前割り当てされたKVキャッシュ
        # shape: (max_batch_size, max_seq_len, d_model)
        self.register_buffer(
            "cache_k",
            torch.zeros(max_batch_size, max_seq_len, d_model),
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(max_batch_size, max_seq_len, d_model),
        )
        self.cache_len = torch.zeros(max_batch_size, dtype=torch.long)  # バッチごとのキャッシュ長

    def reset_cache(self, batch_indices: Optional[torch.Tensor] = None):
        """
        キャッシュをリセット
        batch_indices: リセットするバッチインデックスのリスト（Noneなら全バッチ）
        """
        if batch_indices is None:
            self.cache_k.zero_()
            self.cache_v.zero_()
            self.cache_len.zero_()
        else:
            self.cache_k[batch_indices].zero_()
            self.cache_v[batch_indices].zero_()
            self.cache_len[batch_indices] = 0

    def _apply_mask(
        self,
        attn_scores: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Attentionスコアにマスクを適用
        attn_scores: (B, H, T_q, T_k)
        mask: (B, T_q, T_k) または (T_q, T_k)
        """
        if mask is not None:
            # ヘッド次元にブロードキャスト
            mask = mask.unsqueeze(1)  # (B, 1, T_q, T_k) または (1, 1, T_q, T_k)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        return attn_scores

    def _sliding_window(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        スライディングウィンドウで有効なKVのみを残す
        k: (B, T_total, C)
        v: (B, T_total, C)
        cache_len: (B,) 各バッチの現在のキャッシュ長
        """
        if self.window_size is None:
            return k, v

        B, T_total, C = k.shape
        device = k.device

        new_k_list = []
        new_v_list = []
        for b in range(B):
            cl = cache_len[b].item()
            start = max(0, cl - self.window_size)
            # 有効範囲のみを抽出
            new_k = k[b, start:cl, :]  # (T_eff, C)
            new_v = v[b, start:cl, :]  # (T_eff, C)
            new_k_list.append(new_k)
            new_v_list.append(new_v)

        # バッチごとに長さが異なるのでパディングして結合
        max_len = max(len(nk) for nk in new_k_list)
        padded_k = torch.zeros(B, max_len, C, device=device)
        padded_v = torch.zeros(B, max_len, C, device=device)
        for b, (nk, nv) in enumerate(zip(new_k_list, new_v_list)):
            L = nk.size(0)
            padded_k[b, :L] = nk
            padded_v[b, :L] = nv

        return padded_k, padded_v

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        mask: (B, T_q, T_k) または (T_q, T_k)
        use_cache: True ならKVキャッシュを利用
        batch_indices: キャッシュを利用するバッチのインデックス（Noneなら全バッチ）
        """
        B, T, C = x.shape
        device = x.device

        if batch_indices is None:
            batch_indices = torch.arange(B, device=device)

        # Query/Key/Value 投影
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        if use_cache:
            # 過去のキャッシュを取得
            past_k_list = []
            past_v_list = []
            for i, b_idx in enumerate(batch_indices):
                cl = self.cache_len[b_idx].item()
                if cl > 0:
                    pk = self.cache_k[b_idx, :cl, :].unsqueeze(0)  # (1, cl, C)
                    pv = self.cache_v[b_idx, :cl, :].unsqueeze(0)  # (1, cl, C)
                else:
                    pk = torch.empty(1, 0, C, device=device)
                    pv = torch.empty(1, 0, C, device=device)
                past_k_list.append(pk)
                past_v_list.append(pv)

            # バッチごとに長さが異なるのでパディングして結合
            max_cl = max(pk.size(1) for pk in past_k_list)
            past_k = torch.zeros(B, max_cl, C, device=device)
            past_v = torch.zeros(B, max_cl, C, device=device)
            for i, (pk, pv) in enumerate(zip(past_k_list, past_v_list)):
                cl = pk.size(1)
                past_k[i, :cl] = pk
                past_v[i, :cl] = pv

            # 過去のKVと現在のKVを結合
            k = torch.cat([past_k, k], dim=1)  # (B, T_past + T, C)
            v = torch.cat([past_v, v], dim=1)  # (B, T_past + T, C)

            # スライディングウィンドウ適用
            k, v = self._sliding_window(k, v, self.cache_len[batch_indices])

            # キャッシュ更新（新しいトークン分を追加）
            new_k = k[:, -T:, :].detach().clone()
            new_v = v[:, -T:, :].detach().clone()
            for i, b_idx in enumerate(batch_indices):
                cl = self.cache_len[b_idx].item()
                new_cl = cl + T
                if new_cl <= self.max_seq_len:
                    self.cache_k[b_idx, cl:new_cl] = new_k[i]
                    self.cache_v[b_idx, cl:new_cl] = new_v[i]
                    self.cache_len[b_idx] = new_cl
                else:
                    # 上限を超えた場合は先頭から詰める（簡易的な循環）
                    remain = new_cl - self.max_seq_len
                    self.cache_k[b_idx, :-remain] = self.cache_k[b_idx, remain:].clone()
                    self.cache_v[b_idx, :-remain] = self.cache_v[b_idx, remain:].clone()
                    self.cache_k[b_idx, -T:] = new_k[i]
                    self.cache_v[b_idx, -T:] = new_v[i]
                    self.cache_len[b_idx] = self.max_seq_len

        # Multi-head に分割
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, k.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T_k, D)
        v = v.view(B, v.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T_k, D)

        # Attention スコア計算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T_k)

        # マスク適用
        attn_scores = self._apply_mask(attn_scores, mask)

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (B, H, T, D)

        # 結合して出力
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.out_proj(out)
        return out


class AdvancedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_batch_size: int = 1, max_seq_len: int = 1024):
        super().__init__()
        self.attn = AdvancedMultiHeadAttention(d_model, n_heads, max_batch_size, max_seq_len)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        attn_out = self.attn(self.ln1(x), mask=mask, use_cache=use_cache, batch_indices=batch_indices)
        x = x + attn_out
        # Feed-forward
        x = x + self.ff(self.ln2(x))
        return x


class AdvancedGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_batch_size: int = 1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(voc_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            AdvancedTransformerBlock(d_model, n_heads, max_batch_size, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def reset_cache(self, batch_indices: Optional[torch.Tensor] = None):
        for block in self.blocks:
            block.attn.reset_cache(batch_indices)

    def forward(
        self,
        idx: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        idx: (B, T) トークンID
        mask: (B, T_q, T_k) または (T_q, T_k)
        use_cache: True ならKVキャッシュを利用
        batch_indices: キャッシュを利用するバッチのインデックス
        """
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embed(idx)  # (B, T, C)
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)  # (B, T, C)

        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x, mask=mask, use_cache=use_cache, batch_indices=batch_indices)

        x = self.ln_out(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits


def generate_with_advanced_kv_cache(
    model: AdvancedGPT,
    start_tokens: torch.Tensor,
    max_new_tokens: int,
    context_size: int = 1024,
) -> torch.Tensor:
    """
    高度なKVキャッシュを使った自己回帰生成
    """
    model.eval()
    model.reset_cache()

    idx = start_tokens  # (B, T0)
    B, _ = idx.shape
    device = idx.device

    for step in range(max_new_tokens):
        # コンテキストサイズを超えないように切り詰め
        idx_cond = idx[:, -context_size:] if idx.size(1) > context_size else idx

        # 初回は use_cache=False で全トークンを計算し、KV をキャッシュ
        # 2回目以降は use_cache=True で新しいトークンだけを入力
        use_cache = (idx.size(1) > 1)
        logits = model(idx_cond, use_cache=use_cache)  # (B, T, vocab_size)

        # 最後のトークンのロジットから次トークンをサンプリング（ここでは貪欲）
        next_logits = logits[:, -1, :]  # (B, vocab_size)
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1)

        idx = torch.cat([idx, next_token], dim=1)  # (B, T+1)

    return idx


# 使用例
if __name__ == "__main__":
    vocab_size = 1000
    d_model = 128
    n_heads = 4
    n_layers = 2
    max_batch_size = 2
    max_seq_len = 256

    model = AdvancedGPT(voc_size, d_model, n_heads, n_layers, max_batch_size, max_seq_len)

    # ダミーの開始トークン（バッチサイズ2）
    start_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)

    # 高度なKVキャッシュを使って10トークン生成
    generated = generate_with_advanced_kv_cache(model, start_tokens, max_new_tokens=10)
    print("Generated token IDs:", generated.tolist())