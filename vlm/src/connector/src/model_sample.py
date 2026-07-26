import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def rotate_half(x):
    """Rotates the last dimension by half."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """Applies rotary positional embedding to x."""
    # x: (batch, seq_len, n_heads, head_dim)
    # cos, sin: (seq_len, head_dim)
    cos = cos.unsqueeze(1).unsqueeze(2)  # (seq_len, 1, 1, head_dim)
    sin = sin.unsqueeze(1).unsqueeze(2)  # (seq_len, 1, 1, head_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :])

    def forward(self, x, seq_len=None):
        # x: (batch, seq_len, n_heads, head_dim)
        if seq_len is not None and seq_len > self.cos_cached.shape[1]:
            # 必要に応じて動的に伸長する実装も可能ですが、ここでは簡略化
            raise ValueError("seq_len exceeds precomputed max_seq_len")
        return (
            self.cos_cached[:, :seq_len, :, :],
            self.sin_cached[:, :seq_len, :, :],
        )

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.rotary_pe = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # Q, K, V の線形変換
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # RoPE適用
        cos, sin = self.rotary_pe(q, seq_len=seq_len)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # マルチヘッドAttentionの計算
        # (batch, n_heads, seq_len, head_dim) に並べ替え
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 元の形に戻して出力
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        return self.w_o(attn_output)

class Expert(nn.Module):
    """単一のExpert（通常のFFN）"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class MoEFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # ゲートスコア計算
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # top-k 選択
        topk_weights, topk_indices = torch.topk(
            gate_probs, self.top_k, dim=-1
        )  # (batch, seq_len, top_k)

        # 正規化（各トークンについて top_k の重みの和が1になるように）
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 出力の初期化
        output = torch.zeros_like(x)

        # 各Expertへのルーティングと加算
        for expert_idx, expert in enumerate(self.experts):
            # このExpertが選ばれている位置のマスク
            expert_mask = (topk_indices == expert_idx).any(dim=-1)  # (batch, seq_len)

            if not expert_mask.any():
                continue

            # マスクを適用した入力
            expert_input = x[expert_mask]  # (n_selected, d_model)

            # Expertの出力
            expert_output = expert(expert_input)  # (n_selected, d_model)

            # 対応する重みを集める
            # (batch, seq_len, top_k) から expert_idx を含む位置の重みを抽出
            weights_for_expert = torch.where(
                topk_indices == expert_idx,
                topk_weights,
                torch.zeros_like(topk_weights),
            ).sum(dim=-1)  # (batch, seq_len)

            # マスクされた位置に重み付きで加算
            output[expert_mask] += expert_output * weights_for_expert[expert_mask].unsqueeze(-1)

        return output

class TransformerBlockWithRoPEAndMoE(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttentionWithRoPE(d_model, n_heads)
        self.ffn = MoEFeedForward(d_model, d_ff, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + Residual + Norm
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # MoE FFN + Residual + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class TransformerWithRoPEAndMoE(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        num_layers=6,
        num_experts=4,
        top_k=2,
        dropout=0.1,
        max_seq_len=2048,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlockWithRoPEAndMoE(
                    d_model, n_heads, d_ff, num_experts, top_k, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, mask=None):
        # input_ids: (batch, seq_len)
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x  # (batch, seq_len, d_model)

if __name__ == "__main__":
    vocab_size = 10000
    batch_size, seq_len = 4, 128

    model = TransformerWithRoPEAndMoE(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        num_layers=6,
        num_experts=4,
        top_k=2,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    print(output.shape)  # torch.Size([4, 128, 512])