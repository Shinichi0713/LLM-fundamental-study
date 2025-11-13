import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V の線形変換
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)

        # 出力の線形変換
        self.W_O = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.size()

        # Q, K, Vを各headに分割
        Q = self.W_Q(Q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(K).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(V).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # スケーリングド・ドットプロダクト・アテンション
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ V  # [B, heads, seq, head_dim]

        # 各headを結合して線形変換
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.W_O(attn_output)

        return output, attn_weights
