import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================
# 🔹 Rotary Position Embedding (RoPE)
# ==========================================
def build_rope_cache(seq_len, dim):
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    angles = positions * inv_freq.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    dim = x.shape[-1]
    half_dim = dim // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    # 回転操作
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# ==========================================
# 🔹 Multi-Head Attention 実装
# ==========================================
class SimpleMHA(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, use_rope=False):
        B, N, D = x.shape
        H = self.num_heads
        q = self.Wq(x).view(B, N, H, -1)
        k = self.Wk(x).view(B, N, H, -1)
        v = self.Wv(x).view(B, N, H, -1)

        if use_rope:
            cos, sin = build_rope_cache(N, self.head_dim)
            cos, sin = cos.to(x.device), sin.to(x.device)
            q = apply_rope(q, cos[None, :, None, :], sin[None, :, None, :])
            k = apply_rope(k, cos[None, :, None, :], sin[None, :, None, :])

        attn_scores = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v)
        out = out.reshape(B, N, D)
        return self.out(out), attn

# ==========================================
# 🔹 実験設定
# ==========================================
torch.manual_seed(0)
seq_len = 10
embed_dim = 64
x = torch.randn(1, seq_len, embed_dim)

mha = SimpleMHA(embed_dim=embed_dim, num_heads=4)

# RoPEなし
out_no, attn_no = mha(x, use_rope=False)
# RoPEあり
out_rope, attn_rope = mha(x, use_rope=True)

# ==========================================
# 🔹 可視化 (PCA)
# ==========================================
pca = PCA(n_components=2)
out_no_2d = pca.fit_transform(out_no[0].detach().numpy())
out_rope_2d = pca.transform(out_rope[0].detach().numpy())

plt.figure(figsize=(7,7))
plt.scatter(out_no_2d[:,0], out_no_2d[:,1], label="No RoPE", color="blue")
plt.scatter(out_rope_2d[:,0], out_rope_2d[:,1], label="With RoPE", color="red")
for i in range(seq_len):
    plt.text(out_no_2d[i,0], out_no_2d[i,1], str(i), color="blue")
    plt.text(out_rope_2d[i,0], out_rope_2d[i,1], str(i), color="red")
plt.legend()
plt.title("PCA Projection: MHA outputs (RoPE vs No-RoPE)")
plt.show()

# ==========================================
# 🔹 コサイン類似度の平均を表示
# ==========================================
from sklearn.metrics.pairwise import cosine_similarity
sim = torch.nn.functional.cosine_similarity(out_no.view(-1), out_rope.view(-1), dim=0)
print(f"🔹 Cosine similarity between outputs: {sim.item():.4f}")
