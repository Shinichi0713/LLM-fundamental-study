
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------
# MultiHeadAttention with Relative Positional Bias (T5-style simple)
# ------------------------
class MultiHeadAttentionRel(nn.Module):
    def __init__(self, embed_dim, num_heads, max_rel_pos=8, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_lin = nn.Linear(embed_dim, embed_dim)
        self.k_lin = nn.Linear(embed_dim, embed_dim)
        self.v_lin = nn.Linear(embed_dim, embed_dim)
        self.out_lin = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # relative positional bias table: one bias per head and per relative distance bucket
        self.max_rel_pos = max_rel_pos
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_rel_pos + 1))

    def _relative_index(self, seq_len, device):
        # create matrix of relative indices clipped to [-max_rel_pos, max_rel_pos]
        idxs = torch.arange(seq_len, device=device)
        rel = idxs[None, :] - idxs[:, None]   # shape (seq_len, seq_len): i - j
        rel_clipped = rel.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        # now in range [0, 2*max_rel_pos]
        return rel_clipped.long()  # (seq_len, seq_len)

    def _split_heads(self, x):
        b, seq, _ = x.size()
        x = x.view(b, seq, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (b, heads, seq, head_dim)

    def _combine_heads(self, x):
        # x: (b, heads, seq, head_dim)
        b, h, seq, hd = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # (b, seq, heads, head_dim)
        return x.view(b, seq, h * hd)

    def forward(self, query, key, value, mask=None, use_relative=True):
        # inputs: (batch, seq, embed_dim)
        b, seq, _ = query.size()
        Q = self._split_heads(self.q_lin(query))  # (b, h, seq, hd)
        K = self._split_heads(self.k_lin(key))
        V = self._split_heads(self.v_lin(value))

        # scores: (b, h, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if use_relative:
            # get index matrix (seq, seq) -> gather bias per head
            rel_idx = self._relative_index(seq, device=scores.device)  # (seq, seq)
            # self.relative_bias: (heads, 2*max_rel+1)
            # build bias per head per pair: (heads, seq, seq)
            bias_per_head = self.relative_bias[:, rel_idx]  # broadcasting on first dim
            # bias_per_head shape: (heads, seq, seq)
            # expand to batch: (b, heads, seq, seq)
            bias_per_head = bias_per_head.unsqueeze(0).expand(b, -1, -1, -1)
            scores = scores + bias_per_head

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))  # mask shape broadcast

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # (b, h, seq, hd)
        out = self._combine_heads(context)  # (b, seq, embed_dim)
        out = self.out_lin(out)
        return out, attn, scores  # return scores (pre-softmax) and attn (post-softmax)