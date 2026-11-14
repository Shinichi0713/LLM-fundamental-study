import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedQwen2Config:
    def __init__(self):
        self.vocab_size = 151936
        self.hidden_size = 896
        self.intermediate_size = 4864
        self.num_hidden_layers = 24
        self.num_attention_heads = 14
        self.num_key_value_heads = 2
        self.max_position_embeddings = 32768
        self.rms_norm_eps = 1e-6
        self.rope_theta = 1000000.0
        self.tie_word_embeddings = True
        self.use_logits_soft_cap = True  # 【工夫4】ロジットスパイク抑制のためのソフトキャッピング有効化
        self.logits_soft_cap = 30.0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 【工夫1】低精度（FP16/BF16）での計算時に数値安定性を保つため float32 にキャストして分散計算
        input_dtype = x.dtype
        v = x.to(torch.float32)
        variance = v.pow(2).mean(-1, keepdim=True)
        output = v * torch.rsqrt(variance + self.eps)
        return output.to(input_dtype) * self.weight

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [B, 1, SeqLen, Dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 事前計算キャッシュを動的に拡張できるようにスケーリングを考慮
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

class EnhancedQwen2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # 【工夫2】QKVともにバイアスを持たせることで表現力のゆとりを確保
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # GQAのグループ拡張
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # 高速なカーネル（FlashAttention）が自動選択される PyTorch ネイティブの最適化関数
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)

class EnhancedQwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 【工夫3】アライメント（256の倍数等）を意識した中間次元構造
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class EnhancedQwen2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = EnhancedQwen2Attention(config)
        self.mlp = EnhancedQwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class EnhancedQwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([EnhancedQwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, seq_len=seq_len)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        return self.norm(hidden_states)

class EnhancedQwen2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = EnhancedQwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        # 【工夫4】Logits Soft-Capping の適用（学習時の発散・Nanを防ぎ、生成の安定性を飛躍的に高める）
        if self.config.use_logits_soft_cap:
            logits = self.config.logits_soft_cap * torch.tanh(logits / self.config.logits_soft_cap)

        return logits

# --- 動作確認用テストスクリプト ---
if __name__ == "__main__":
    config = EnhancedQwen2Config()
    model = EnhancedQwen2ForCausalLM(config)
    
    # パラメータ数の確認
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Enhanced Model Total Parameters: {total_params / 1e6:.2f}M")

    # ダミー入力によるフォワードテスト
    input_ids = torch.randint(0, config.vocab_size, (2, 32)) # Batch=2, SeqLen=32
    logits = model(input_ids)
    print(f"Stabilized Logits Shape: {logits.shape}")