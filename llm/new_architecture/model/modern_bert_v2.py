import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. Rotary Position Embedding (RoPE)
# ----------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k shape: (batch_size, num_heads, seq_len, head_dim)
    # cos, sin shape: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

# ----------------------------------------------------------------------
# 2. RMSNorm
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

# ----------------------------------------------------------------------
# 3. Modern Self-Attention (FlashAttention 互換 API 統合)
# ----------------------------------------------------------------------
class ModernAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding, attention_mask: torch.Tensor = None):
        B, S, D = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        
        # (B, S, num_heads, head_dim) -> (B, num_heads, S, head_dim)
        q = qkv[0].view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[1].view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = qkv[2].view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE の適用
        cos, sin = rope(x, S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # PyTorch 2.0+ scaled_dot_product_attention (FlashAttention バックエンドを自動選択)
        # encoder 用のため causal=False
        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attention_mask, 
            is_causal=False
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_out)

# ----------------------------------------------------------------------
# 4. SwiGLU FFN Layer
# ----------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ----------------------------------------------------------------------
# 5. Transformer Encoder Block (Pre-LN)
# ----------------------------------------------------------------------
class ModernBertBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = ModernAttention(hidden_dim, num_heads)
        
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding, attention_mask: torch.Tensor = None):
        # Pre-LN Residual Connection
        x = x + self.attn(self.norm1(x), rope, attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x

# ----------------------------------------------------------------------
# 6. Full ModernBERT Model
# ----------------------------------------------------------------------
class ModernBERT(nn.Module):
    def __init__(
        self, 
        vocab_size: int = 50352,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_dim: int = 2048,
        max_seq_len: int = 8192
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.rope = RotaryEmbedding(dim=hidden_dim // num_heads, max_seq_len=max_seq_len)
        
        self.layers = nn.ModuleList([
            ModernBertBlock(hidden_dim, num_heads, intermediate_dim)
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        x = self.embeddings(input_ids)

        # Padding Mask の整形 (B, 1, 1, S)
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # BoolMask 化（1: Valid, 0: Masked）
            attention_mask = (attention_mask == 0)

        for layer in self.layers:
            x = layer(x, self.rope, attention_mask)

        return self.final_norm(x)