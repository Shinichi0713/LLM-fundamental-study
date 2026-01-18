import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, T5Tokenizer, T5ForConditionalGeneration

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, q, kv):
        attn_out, _ = self.attn(q, kv, kv)
        return self.ln(q + attn_out)

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.ln(x + attn_out)

class QFormer(nn.Module):
    def __init__(self, dim=768, n_queries=32, n_heads=12, n_layers=6):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, dim))
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                SelfAttentionBlock(dim, n_heads),
                CrossAttentionBlock(dim, n_heads)
            ]))

    def forward(self, vision_features):
        B = vision_features.size(0)
        queries = self.query_embed.expand(B, -1, -1)
        for self_attn, cross_attn in self.layers:
            queries = self_attn(queries)
            queries = cross_attn(queries, vision_features)
        return queries
