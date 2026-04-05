import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    """Rotates half the hidden dim of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    """Applies rotary positional embedding to x."""
    # x: (batch, seq_len, n_heads, head_dim)
    # cos, sin: (seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
    return (x * cos) + (rotate_half(x) * sin)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU: swish(xW1) * (xW3) の後に W2 で射影
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class Attention(nn.Module):
    def __init__(self, dim, n_heads, head_dim, max_seq_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        # RoPE用のcos/sinキャッシュ（簡略版）
        self.register_buffer("cos_cached", None)
        self.register_buffer("sin_cached", None)
        self.max_seq_len = max_seq_len

    def _build_rope_cache(self, seq_len, device):
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

        # 簡略化したRoPEキャッシュ生成
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        self.cos_cached = cos
        self.sin_cached = sin
        return cos, sin

    def forward(self, x, mask=None):
        # x: (batch, seq_len, dim)
        batch_size, seq_len, _ = x.shape
        device = x.device

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE適用
        cos, sin = self._build_rope_cache(seq_len, device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # スコア計算
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 出力形状を戻す
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, head_dim, ff_dim, max_seq_len=2048):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, head_dim, max_seq_len)
        self.ffn = SwiGLUFFN(dim, ff_dim)

    def forward(self, x, mask=None):
        # Pre-Norm + Residual
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LlamaLikeModel(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, head_dim, ff_dim, max_seq_len=2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, head_dim, ff_dim, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # トークン埋め込みとLMヘッドの重みを共有（オプション）
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, mask=None):
        # input_ids: (batch, seq_len)
        x = self.token_emb(input_ids)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
    
def make_model():
    # ハイパーパラメータ（例：LLaMA-2 7B風）
    vocab_size = 32000
    dim = 4096
    n_layers = 32
    n_heads = 32
    head_dim = 128  # dim // n_heads
    ff_dim = 11008  # 通常は dim * 約2.7

    model = LlamaLikeModel(vocab_size, dim, n_layers, n_heads, head_dim, ff_dim)

    # ダミー入力
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # マスク（因果マスク）
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    logits = model(input_ids, mask=mask)
    print(logits.shape)  # (2, 64, 32000)

    return model