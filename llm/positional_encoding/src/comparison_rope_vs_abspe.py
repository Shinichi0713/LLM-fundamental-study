import torch
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 🔹 基本設定
# ==========================================
seq_len = 10
embed_dim = 8

# ランダムな Query / Key
torch.manual_seed(0)
Q = torch.randn(seq_len, embed_dim)
K = torch.randn(seq_len, embed_dim)

# ==========================================
# 🔹 ① 絶対位置エンコーディング
# ==========================================
def sinusoidal_encoding(seq_len, dim):
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
    angle_rates = 1 / (10000 ** (2 * (i // 2) / dim))
    angles = pos * angle_rates
    enc = torch.zeros_like(angles)
    enc[:, 0::2] = torch.sin(angles[:, 0::2])
    enc[:, 1::2] = torch.cos(angles[:, 1::2])
    return enc

pos_emb = sinusoidal_encoding(seq_len, embed_dim)
Q_abs = Q + pos_emb
K_abs = K + pos_emb

# Attentionスコア
attn_abs = torch.softmax(Q_abs @ K_abs.T / np.sqrt(embed_dim), dim=-1)

# ==========================================
# 🔹 ② RoPE（Rotary Position Embedding）
# ==========================================
def build_rope_cache(seq_len, dim):
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    angles = positions * inv_freq.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    half_dim = x.shape[-1] // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

cos, sin = build_rope_cache(seq_len, embed_dim)
Q_rope = apply_rope(Q, cos, sin)
K_rope = apply_rope(K, cos, sin)
attn_rope = torch.softmax(Q_rope @ K_rope.T / np.sqrt(embed_dim), dim=-1)

# ==========================================
# 🔹 可視化
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(attn_abs.detach().numpy(), cmap='viridis')
axes[0].set_title("Absolute Positional Encoding")
axes[0].set_xlabel("Key index")
axes[0].set_ylabel("Query index")

axes[1].imshow(attn_rope.detach().numpy(), cmap='viridis')
axes[1].set_title("Rotary Position Embedding (RoPE)")
axes[1].set_xlabel("Key index")
axes[1].set_ylabel("Query index")

plt.tight_layout()
plt.show()
