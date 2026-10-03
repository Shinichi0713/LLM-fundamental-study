import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. コンポーネント実装 (RoPE, RMSNorm, SwiGLU)
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ==========================================
# 2. KV キャッシュ対応 Attention
# ==========================================

class KVCacheAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache=None):
        """
        x: [batch_size, seq_len, hidden_size]
        kv_cache: (past_k, past_v) のタプル、初回は None
        """
        batch_size, seq_len, _ = x.shape

        # 新規トークン分の Q, K, V を計算
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # RoPE の適用
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # --- KV キャッシュの保持と更新 ---
        if kv_cache is not None:
            past_k, past_v = kv_cache
            # 過去の K, V と今回の新規 K, V をシーケンス方向に連結
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # 更新後の K, V を新しいキャッシュとして保持
        new_kv_cache = (k, v)

        # GQA: KV ヘッド数の拡張
        if self.num_kv_groups > 1:
            k_expanded = k.repeat_interleave(self.num_kv_groups, dim=2)
            v_expanded = v.repeat_interleave(self.num_kv_groups, dim=2)
        else:
            k_expanded, v_expanded = k, v

        # 転置して [batch, heads, seq_len, head_dim] に変換
        q = q.transpose(1, 2)
        k_expanded = k_expanded.transpose(1, 2)
        v_expanded = v_expanded.transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k_expanded.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Causal Mask (初回プロンプト入力時など seq_len > 1 の場合のみ適用)
        total_seq_len = k_expanded.shape[2]
        if seq_len > 1:
            mask = torch.triu(torch.full((seq_len, total_seq_len), float('-inf'), device=x.device), diagonal=1 + (total_seq_len - seq_len))
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_expanded)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output), new_kv_cache


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int, rms_norm_eps: float):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = KVCacheAttention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache=None):
        residual = x
        normed_x = self.input_layernorm(x)
        attn_out, new_kv_cache = self.self_attn(normed_x, cos, sin, kv_cache=kv_cache)
        x = residual + attn_out

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x) + residual
        return x, new_kv_cache


# ==========================================
# 3. LLM 全体構造 & 高速テキスト生成
# ==========================================

class LLMWithKVCache(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        intermediate_size: int = 1024,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, num_kv_heads, intermediate_size, 1e-6)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Precompute RoPE
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, past_kv_caches=None):
        batch_size, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        # 現在の位置に応じた RoPE の抽出
        cos = self.cos_cached[start_pos : start_pos + seq_len]
        sin = self.sin_cached[start_pos : start_pos + seq_len]

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = past_kv_caches[i] if past_kv_caches is not None else None
            x, new_cache = layer(x, cos, sin, kv_cache=layer_cache)
            new_kv_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 20):
        """KV キャッシュを活用したインクリメンタル生成ループ"""
        self.eval()
        batch_size, prompt_len = prompt_ids.shape
        generated = prompt_ids.clone()

        # Step 1: プロンプト（文脈）全体の初期計算 (Prefill 段階)
        logits, kv_caches = self.forward(prompt_ids, start_pos=0, past_kv_caches=None)
        
        # 最後のトークンの出力から次のトークンを予測
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        # Step 2: 1トークンずつインクリメンタルに生成 (Decode 段階)
        for i in range(max_new_tokens - 1):
            current_pos = prompt_len + i
            # ★ポイント: 新しく生成された 1 トークンのみを入力する
            logits, kv_caches = self.forward(
                next_token, 
                start_pos=current_pos, 
                past_kv_caches=kv_caches
            )
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

        return generated


# ==========================================
# 4. 実行確認
# ==========================================

if __name__ == "__main__":
    model = LLMWithKVCache()
    
    # バッチサイズ 1、長さ 5 のプロンプト例
    prompt = torch.tensor([[101, 2054, 2003, 1037, 3899]])

    print(f"プロンプト形状: {prompt.shape}")
    
    # テキスト生成の実行
    output = model.generate(prompt, max_new_tokens=10)
    
    print("生成結果（トークンID配列）:")
    print(output)
    print(f"最終出力の長さ: {output.shape[1]} トークン")