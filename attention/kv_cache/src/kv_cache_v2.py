import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    シンプルなマルチヘッドself-attention（デコーダ用）
    KVキャッシュを保持・更新する
    """
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # KVキャッシュ用のバッファ（推論時に使う）
        self.register_buffer("k_cache", None)
        self.register_buffer("v_cache", None)

    def _reset_cache(self):
        """KVキャッシュをリセット"""
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, mask=None, use_cache=False):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len) または (batch, 1, seq_len) など
        use_cache: TrueならKVキャッシュを更新・利用
        """
        batch, seq_len, d_model = x.shape

        # Q, K, V の線形変換
        q = self.wq(x)  # (batch, seq_len, d_model)
        k = self.wk(x)  # (batch, seq_len, d_model)
        v = self.wv(x)  # (batch, seq_len, d_model)

        # マルチヘッドに分割
        q = q.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch, n_heads, seq_len, d_k)
        k = k.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # KVキャッシュの処理
        if use_cache:
            if self.k_cache is None:
                # 初回は現在のK, Vをそのままキャッシュ
                self.k_cache = k
                self.v_cache = v
            else:
                # 既存キャッシュに現在のK, Vを連結
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)
            # キャッシュ全体を使う
            k_use = self.k_cache
            v_use = self.v_cache
        else:
            # キャッシュを使わない場合は現在のK, Vのみ
            k_use = k
            v_use = v
            # キャッシュはリセット
            self._reset_cache()

        # アテンションスコア計算
        scores = torch.matmul(q, k_use.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, n_heads, seq_len, cache_seq_len)

        if mask is not None:
            # マスク適用（例：未来のトークンを隠す）
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, cache_seq_len)

        # 重み付き和で出力
        out = torch.matmul(attn_weights, v_use)  # (batch, n_heads, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)  # (batch, seq_len, d_model)
        out = self.wo(out)

        # 可視化用にKVキャッシュとアテンション重みも返す
        return out, k_use, v_use, attn_weights
    

import matplotlib.pyplot as plt
import numpy as np

def visualize_kv_cache(k_cache, v_cache, attn_weights, head_idx=0, token_idx=0):
    """
    KVキャッシュとアテンション重みを可視化する簡易関数
    
    k_cache: (batch, n_heads, cache_seq_len, d_k)
    v_cache: (batch, n_heads, cache_seq_len, d_k)
    attn_weights: (batch, n_heads, seq_len, cache_seq_len)
    head_idx: 可視化するヘッド番号
    token_idx: 可視化する現在のトークン位置
    """
    batch, n_heads, cache_seq_len, d_k = k_cache.shape

    # 特定ヘッド・トークンのkey/valueの平均活性
    k_mean = k_cache[0, head_idx].mean(dim=-1).detach().cpu().numpy()  # (cache_seq_len,)
    v_mean = v_cache[0, head_idx].mean(dim=-1).detach().cpu().numpy()  # (cache_seq_len,)

    # アテンション重み（特定ヘッド・トークン）
    attn = attn_weights[0, head_idx, token_idx].detach().cpu().numpy()  # (cache_seq_len,)

    # プロット
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Keyの平均活性
    axes[0].plot(k_mean, marker='o')
    axes[0].set_title(f"Head {head_idx}: Mean Key Activation")
    axes[0].set_xlabel("Cache position")
    axes[0].set_ylabel("Activation")

    # Valueの平均活性
    axes[1].plot(v_mean, marker='o')
    axes[1].set_title(f"Head {head_idx}: Mean Value Activation")
    axes[1].set_xlabel("Cache position")
    axes[1].set_ylabel("Activation")

    # アテンション重みのヒートマップ（全ヘッド・全トークン）
    attn_all = attn_weights[0, :, token_idx].detach().cpu().numpy()  # (n_heads, cache_seq_len)
    im = axes[2].imshow(attn_all, aspect='auto', cmap='viridis')
    axes[2].set_title(f"Attention Weights (all heads, token {token_idx})")
    axes[2].set_xlabel("Cache position")
    axes[2].set_ylabel("Head index")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.show()

    # 数値も少し表示
    print(f"Key mean act (head {head_idx}): {k_mean}")
    print(f"Value mean act (head {head_idx}): {v_mean}")
    print(f"Attention (head {head_idx}, token {token_idx}): {attn}")


# モデルと入力の準備
model = MultiHeadAttention(d_model=512, n_heads=8)
batch_size = 1
d_model = 512

# ダミーのトークン列（例：5ステップ分）
tokens_list = [
    torch.randn(batch_size, 1, d_model),  # ステップ1: 1トークン
    torch.randn(batch_size, 1, d_model),  # ステップ2: 1トークン
    torch.randn(batch_size, 1, d_model),  # ステップ3: 1トークン
    torch.randn(batch_size, 1, d_model),  # ステップ4: 1トークン
    torch.randn(batch_size, 1, d_model),  # ステップ5: 1トークン
]

# マスク（未来のトークンを隠す）
def causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # (seq_len, seq_len)

# ステップごとに推論（KVキャッシュ使用）
for step, tokens in enumerate(tokens_list):
    seq_len = tokens.shape[1]
    mask = causal_mask(seq_len).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    # 推論（use_cache=TrueでKVキャッシュ更新）
    out, k_cache, v_cache, attn_weights = model(tokens, mask=mask, use_cache=True)

    print(f"--- Step {step+1} ---")
    print(f"KV cache shape: {k_cache.shape}")  # (batch, n_heads, cache_seq_len, d_k)

    # 可視化（最後のステップだけ詳細に見るなど）
    if step == len(tokens_list) - 1:
        visualize_kv_cache(k_cache, v_cache, attn_weights, head_idx=0, token_idx=0)

# キャッシュをリセット（次のシーケンス用）
model._reset_cache()