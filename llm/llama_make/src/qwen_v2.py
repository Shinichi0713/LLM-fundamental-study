import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. モデル名とデバイスの設定
model_name = "Qwen/Qwen2-7B-Instruct"  # Lightweight に試す場合は "Qwen/Qwen2-0.5B-Instruct" や "Qwen/Qwen2-1.5B-Instruct" も利用可能

# CUDA（GPU）が利用可能かチェック
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {model_name}...")

# 2. トークナイザーとモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

# 3. チャット形式のプロンプト構築（Qwen 2 の Chat Template を適用）
messages = [
    {"role": "system", "content": "あなたは優秀で丁寧なAIアシスタントです。"},
    {"role": "user", "content": "Qwen 2モデルの特徴を短く分かりやすく教えてください。"}
]

# トークナイザーの apply_chat_template を使ってプロンプト文字列を作成
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 4. 入力のエンコード
model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

# 5. テキスト生成（推論）
print("\nGenerating response...")
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

# 入力部分を取り除き、新しく生成されたトークンのみを取得
generated_ids = [
    output_ids[len(input_ids):] 
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 6. デコードして応答を表示
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n--- Response ---")
print(response)

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 各種コンポーネントの実装
# ==========================================

class RMSNorm(nn.Module):
    """Qwen2 で採用されている Root Mean Square Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        # 周波数の計算
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # キャッシュの作成
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # [seq_len, dim] -> [1, seq_len, 1, dim]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLUMLP(nn.Module):
    """Qwen2 で利用される SwiGLU アンプ構造"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (SiLU(gate_proj(x)) * up_proj(x)) -> down_proj
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    """Grouped-Query Attention (GQA) を備えた Attention モジュール"""
    def __init__(self, hidden_size: int, num_attention_heads: int, num_key_value_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Q, K, V プロジェクション (Qwen2 では Q, K, V の Linear に bias が存在)
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # プロジェクション変換
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # RoPE の適用
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: KV ヘッド数を Query ヘッド数に合わせて拡張
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # [batch_size, num_heads, seq_len, head_dim] に転置
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot-Product Attention (Causal Mask 含む)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # 因果マスク（未来のトークンを見ないようにする）
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 元の形状に変形して統合
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class Qwen2DecoderLayer(nn.Module):
    """Transformer デコーダーレイヤー 1層分"""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int, rms_norm_eps: float):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen2Attention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Pre-LN 構造 + 残差接続
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin) + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x) + residual
        return x


# ==========================================
# 2. Qwen2 全体モデル構造
# ==========================================

class Qwen2ForCausalLM(nn.Module):
    """Qwen2 モデル全体ブロック"""
    def __init__(
        self,
        vocab_size: int = 151936,       # Qwen2 標準の語彙数
        hidden_size: int = 896,         # 小規模パラメータ例 (0.5B相当)
        num_hidden_layers: int = 4,     # テスト用に層数を縮小 (本来は24等)
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 2048
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_attention_heads, 
            max_position_embeddings=max_position_embeddings
        )
        
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps
            ) for _ in range(num_hidden_layers)
        ])
        
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        # 位置埋め込み（RoPE）の計算
        cos, sin = self.rotary_emb(x, seq_len)

        # Transformer レイヤーの順伝播
        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ==========================================
# 3. 動作確認
# ==========================================

if __name__ == "__main__":
    # ミニマムな設定でモデルを作成
    model = Qwen2ForCausalLM(
        vocab_size=10000,
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=2048
    )

    # バッチサイズ 2、系列長 16 のダミー入力ID
    dummy_input_ids = torch.randint(0, 10000, (2, 16))

    # フォワードパス実行
    logits = model(dummy_input_ids)

    print("モデルの実行に成功しました。")
    print(f"入力形状: {dummy_input_ids.shape}")
    print(f"出力 Logits 形状: {logits.shape}")  # [Batch, Seq_Len, Vocab_Size]