# Modern-BERT (PyTorch) — minimal, runnable example
# Features: Pre-LN, RoPE, MultiHeadAttn, SwiGLU (Gated FFN), classification head
# Run in Google Colab / local with `pip install torch` if needed.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# RoPE utilities
# --------------------------
def build_rope_cache(seq_len: int, dim: int, device=None):
    """Return cos, sin of shape (seq_len, dim) where dim is head_dim."""
    # head_dim must be even
    half = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.einsum("i,j->ij", positions, inv_freq)  # (seq_len, half)
    sin = torch.sin(angles).to(device)
    cos = torch.cos(angles).to(device)
    # interleave to shape (seq_len, dim)
    sin = torch.repeat_interleave(sin, 2, dim=1)
    cos = torch.repeat_interleave(cos, 2, dim=1)
    return cos, sin

def apply_rope_to_qk(x, cos, sin):
    """
    x: (..., seq_len, dim)
    cos,sin: (seq_len, dim)
    returns rotated x with broadcasting
    """
    # assume last-two dims are (seq_len, dim)
    *prefix, seq, dim = x.size()
    x_ = x.view(-1, seq, dim)  # (B', seq, dim)
    x1 = x_[..., ::2]  # even
    x2 = x_[..., 1::2]  # odd
    cos_ = cos[None, :, ::2]  # (1, seq, dim/2)
    sin_ = sin[None, :, ::2]
    # rotation
    rx_even = x1 * cos_ - x2 * sin_
    rx_odd  = x1 * sin_ + x2 * cos_
    rx = torch.stack([rx_even, rx_odd], dim=-1).reshape(x_.shape)  # interleave back
    return rx.view(*prefix, seq, dim)

# --------------------------
# MultiHead Attention with RoPE
# --------------------------
class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, cos=None, sin=None):
        # x: (batch, seq, embed_dim)
        B, S, _ = x.size()
        qkv = self.qkv(x)  # (B, S, 3*E)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # qkv: (3, B, heads, S, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, S, head_dim)

        if cos is not None and sin is not None:
            # cos,sin shape: (S, head_dim)
            # bring q/k to shape (B*heads, S, head_dim) for rope application
            q = q.permute(0,1,2,3)  # same shape
            k = k.permute(0,1,2,3)
            q = q.reshape(B * self.num_heads, S, self.head_dim)
            k = k.reshape(B * self.num_heads, S, self.head_dim)
            q = apply_rope_to_qk(q, cos, sin)
            k = apply_rope_to_qk(k, cos, sin)
            q = q.view(B, self.num_heads, S, self.head_dim)
            k = k.view(B, self.num_heads, S, self.head_dim)

        # scaled dot-product
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.head_dim)  # (B, heads, S, S)
        if attn_mask is not None:
            scores = scores + attn_mask  # mask should be additive (e.g. -inf)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)  # (B, heads, S, head_dim)
        out = out.transpose(1,2).contiguous().view(B, S, self.embed_dim)
        return self.out(out), attn

# --------------------------
# SwiGLU feedforward
# --------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim * 2)  # gating: produce 2*hidden
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x_proj = self.w1(x)
        a, b = x_proj.chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)

# --------------------------
# Transformer Encoder Block (Pre-LN)
# --------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, use_rope=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = RoPEMultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn = SwiGLU(embed_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope

    def forward(self, x, cos=None, sin=None, attn_mask=None):
        # Pre-LN
        y = self.ln1(x)
        if self.use_rope:
            out, attn = self.attn(y, attn_mask=attn_mask, cos=cos, sin=sin)
        else:
            out, attn = self.attn(y, attn_mask=attn_mask, cos=None, sin=None)
        x = x + self.dropout(out)

        z = self.ln2(x)
        z = self.ffn(z)
        x = x + self.dropout(z)
        return x, attn

# --------------------------
# ModernBERT Model
# --------------------------
class ModernBert(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim=256,
                 num_heads=8,
                 mlp_dim=1024,
                 depth=6,
                 max_seq=512,
                 dropout=0.1,
                 use_rope=True):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.embed_type = nn.Embedding(2, embed_dim)  # token type ids (segment)
        self.position_embed = None  # we use RoPE, so no learned pos emb (optional)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout=dropout, use_rope=use_rope)
            for _ in range(depth)
        ])
        self.ln_final = nn.LayerNorm(embed_dim, eps=1e-6)
        self.classifier = nn.Linear(embed_dim, 2)  # example: binary classification head
        self.max_seq = max_seq
        self.use_rope = use_rope
        # precompute RoPE cache (will be moved to device on forward)
        self.register_buffer("_rope_cos", torch.zeros(max_seq, embed_dim//num_heads), persistent=False)
        self.register_buffer("_rope_sin", torch.zeros(max_seq, embed_dim//num_heads), persistent=False)
        self._rope_prepared = False

    def _prepare_rope(self, device):
        if self._rope_prepared:
            return
        # head_dim = embed_dim // num_heads
        head_dim = self.layers[0].attn.head_dim
        cos, sin = build_rope_cache(self.max_seq, head_dim, device=device)
        self._rope_cos[:cos.size(0), :cos.size(1)] = cos
        self._rope_sin[:sin.size(0), :sin.size(1)] = sin
        self._rope_prepared = True

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        input_ids: (B, S)
        token_type_ids: (B, S) optional
        attention_mask: (B, S) where 1 indicates keep, 0 indicates pad
        """
        B, S = input_ids.size()
        device = input_ids.device
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = self.embed_tokens(input_ids) + self.embed_type(token_type_ids)
        x = self.dropout(x)

        # prepare RoPE cos/sin for sequence length S on device
        if self.use_rope:
            self._prepare_rope(device)
            cos = self._rope_cos[:S, :].to(device)  # shape (S, head_dim)
            sin = self._rope_sin[:S, :].to(device)
        else:
            cos = sin = None

        # build additive attn mask (optional)
        attn_add = None
        if attention_mask is not None:
            # attention_mask: 1 for keep, 0 for pad
            # convert to shape (B, 1, 1, S) additive mask
            attn_add = (1.0 - attention_mask[:, None, None, :]) * -1e9

        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, cos=cos, sin=sin, attn_mask=attn_add)
            attn_maps.append(attn)  # store per-layer attention for debugging/visual
        x = self.ln_final(x)

        # simple classification: pool CLS (here index 0)
        pooled = x[:, 0, :]  # assume first token as [CLS]
        logits = self.classifier(pooled)
        return logits, attn_maps

# --------------------------
# Quick runnable demo with dummy data
# --------------------------
def demo_run():
    vocab_size = 5000
    model = ModernBert(vocab_size=vocab_size, embed_dim=128, num_heads=8, mlp_dim=512, depth=4, max_seq=128, use_rope=True)
    model.train()

    # dummy batch
    B, S = 8, 32
    input_ids = torch.randint(0, vocab_size, (B, S))
    token_type_ids = torch.zeros((B, S), dtype=torch.long)
    attention_mask = torch.ones((B, S), dtype=torch.float32)

    logits, attn_maps = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    print("logits:", logits.shape)  # (B, num_classes)
    print("len(attn_maps):", len(attn_maps), "attn shape per layer:", attn_maps[0].shape)

    # quick training loop (toy)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    labels = torch.randint(0, 2, (B,))
    loss_fn = nn.CrossEntropyLoss()

    for step in range(10):
        optim.zero_grad()
        logits, _ = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optim.step()
        if step % 2 == 0:
            print(f"step {step} loss {loss.item():.4f}")

if __name__ == "__main__":
    demo_run()
    