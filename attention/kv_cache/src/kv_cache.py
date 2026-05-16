import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)  # (B, S, D)
        k = self.k_proj(x)  # (B, S, D)
        v = self.v_proj(x)  # (B, S, D)

        # マルチヘッド化
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, H, S, head_dim)

        # KVキャッシュがある場合は結合
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # 新しいKVキャッシュを返す
        new_kv = (k, v)

        # スコア計算
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 出力
        out = torch.matmul(attn_weights, v)  # (B, H, S, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        return out, new_kv