import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. RMSNorm
# ----------------------------------------------------------------------
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)

# ----------------------------------------------------------------------
# 2. Rotary Position Embedding (RoPE)
# ----------------------------------------------------------------------
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, dim)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ----------------------------------------------------------------------
# 3. Grouped-Query Attention (GQA) with QKV Bias
# ----------------------------------------------------------------------
class Qwen2Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = num_heads // num_key_value_heads

        # Qwen の特徴: qkv_proj に bias=True を採用するケースが多い
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        rotary_emb: Qwen2RotaryEmbedding, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = rotary_emb(v, q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: KV Head を Query Head 数に合わせてリピート拡張
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # PyTorch 2.0+ SDPA (Causal Mask 対応)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attention_mask, 
            is_causal=(attention_mask is None)
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)

# ----------------------------------------------------------------------
# 4. SwiGLU MLP
# ----------------------------------------------------------------------
class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ----------------------------------------------------------------------
# 5. Decoder Layer
# ----------------------------------------------------------------------
class Qwen2DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(hidden_size)
        self.self_attn = Qwen2Attention(hidden_size, num_heads, num_key_value_heads)
        self.post_attention_layernorm = Qwen2RMSNorm(hidden_size)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Qwen2RotaryEmbedding, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN Residual Connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary_emb, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

# ----------------------------------------------------------------------
# 6. Full Qwen Model
# ----------------------------------------------------------------------
class Qwen2ForCausalLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int = 151936,
        hidden_size: int = 896,        # Qwen2-0.5B 相当のデフォルト値
        num_hidden_layers: int = 24,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,  # GQA (14 heads / 2 kv_heads)
        intermediate_size: int = 4864
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary_emb = Qwen2RotaryEmbedding(dim=hidden_size // num_attention_heads)
        
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(hidden_size, num_attention_heads, num_key_value_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ])
        
        self.norm = Qwen2RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, self.rotary_emb, attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits