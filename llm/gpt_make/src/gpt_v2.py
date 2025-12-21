
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random


# 特殊トークンID
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 1 # GPTでは通常不要だが、例示のため含める
MASK_TOKEN_ID = 2 # GPTでは通常不要だが、MLMのデータ生成を想定
BOS_TOKEN_ID = 3 # Begin Of Sequence (通常はこれが使われる)
EOS_TOKEN_ID = 4 # End Of Sequence


def rotate_half(x):
    # x: (..., dim)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(q, k, seq_len, device):
    """
    q, k: (batch, heads, seq_len, head_dim)
    """
    head_dim = q.size(-1)
    assert head_dim % 2 == 0, "RoPE requires even head_dim"

    # 周波数
    theta = 10000 ** (-torch.arange(0, head_dim, 2, device=device) / head_dim)
    positions = torch.arange(seq_len, device=device)

    freqs = torch.einsum("i,j->ij", positions, theta)  # (seq_len, head_dim/2)
    sin = freqs.sin()[None, None, :, :]
    cos = freqs.cos()[None, None, :, :]

    # 偶奇次元に適用
    q_rot = (q[..., ::2] * cos) + (rotate_half(q)[..., ::2] * sin)
    k_rot = (k[..., ::2] * cos) + (rotate_half(k)[..., ::2] * sin)

    q = torch.cat([q_rot, q[..., 1::2]], dim=-1)
    k = torch.cat([k_rot, k[..., 1::2]], dim=-1)

    return q, k

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ===== RoPE を適用 =====
        q, k = apply_rope(q, k, seq_len, x.device)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_linear(context)
        return output

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# 1.4. MoELayer (MoE オプション時)
class MoELayer(nn.Module):
    def __init__(self, dim, num_experts, top_k, expert_hidden_dim=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_dim = expert_hidden_dim if expert_hidden_dim is not None else dim * 2
        
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)
        
    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])
        N_tokens = x.size(0)
        
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x)
        
        # ロードバランシング損失
        expert_usage_one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1).float()
        expert_router_prob = gate_weights.sum(dim=0) / N_tokens
        expert_fraction_routed = expert_usage_one_hot.sum(dim=0) / N_tokens
        load_balancing_loss = (expert_router_prob * expert_fraction_routed).sum()
        
        for k in range(self.top_k):
            expert_index = top_k_indices[:, k]
            weight = top_k_weights[:, k]       
            
            for i in range(self.num_experts):
                mask = (expert_index == i) 
                if not mask.any():
                    continue
                expert_input = x[mask]
                expert_output = self.experts[i](expert_input)
                weighted_output = expert_output * weight[mask].unsqueeze(1)
                final_output[mask] += weighted_output

        final_output = final_output.view(original_shape)
        return final_output, load_balancing_loss

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim,
                 use_moe=False, num_experts=None, top_k=None):
        super().__init__()

        self.norm1 = RMSNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm2 = RMSNorm(embed_dim)

        self.use_moe = use_moe
        if use_moe:
            self.ffn_or_moe = MoELayer(embed_dim, num_experts, top_k, ffn_hidden_dim)
        else:
            self.ffn_or_moe = SwiGLU(embed_dim, ffn_hidden_dim)

    def forward(self, x, mask):
        x = x + self.attention(self.norm1(x), mask)

        if self.use_moe:
            ffn_out, moe_loss = self.ffn_or_moe(self.norm2(x))
            x = x + ffn_out
            return x, moe_loss
        else:
            x = x + self.ffn_or_moe(self.norm2(x))
            return x, None


# ====================================================================
# 2. GPTモデル本体
# ====================================================================

class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, ffn_hidden_dim,
                 use_moe=False, num_experts=None, top_k=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # トークン埋め込み層
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN_ID)
        # 位置埋め込み層
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        
        # Transformer Decoderブロックのスタック
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ffn_hidden_dim, use_moe, num_experts, top_k)
            for _ in range(num_layers)
        ])
        
        # 最終のLayer Normalization (出力前)
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # 出力層 (語彙サイズへの線形変換)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        self.use_moe = use_moe

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        
        batch_size, seq_len = input_ids.shape
        
        # トークン埋め込み
        token_embeds = self.token_embedding(input_ids)
        
        # 位置埋め込み (torch.arangeで位置IDを生成)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_embeds = self.position_embedding(position_ids)
        
        # 埋め込みの合計
        x = token_embeds + position_embeds
        
        # マスクの作成 (未来のトークンを参照しないようにする)
        # causal_mask: (seq_len, seq_len) の下三角行列
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).bool()
        # パディングマスクはここでは考慮しない (MLMデータセットで対応)
        
        total_moe_aux_loss = 0.0
        
        # Decoderブロックを順に適用
        for layer in self.decoder_layers:
            output, moe_aux_loss = layer(x, causal_mask)
            x = output
            if self.use_moe and moe_aux_loss is not None:
                total_moe_aux_loss += moe_aux_loss
        
        # 最終Layer Normalization
        x = self.final_norm(x)
        
        # 言語モデルヘッド (logits)
        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)
        
        return logits, total_moe_aux_loss
