
import torch
import math

def rotate_every_two(x):
    # x: (batch, seq, dim)
    x1 = x[..., ::2]   # even dims
    x2 = x[..., 1::2]  # odd dims
    # rotate (x1, x2)
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def apply_rope(x, sin, cos):
    # x, sin, cos: (batch, seq, dim)
    return (x * cos) + (rotate_every_two(x) * sin)

def rope_embedding(seq_len, dim, device):
    # frequency base
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float().to(device) / dim))
    pos = torch.arange(seq_len, device=device).float()

    # angles: (seq, dim/2)
    angles = pos[:, None] * theta[None, :]
    sin = torch.sin(angles).repeat_interleave(2, dim=1)  # (seq, dim)
    cos = torch.cos(angles).repeat_interleave(2, dim=1)

    return sin, cos


class RoPEAttention(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.o = torch.nn.Linear(dim, dim)

    def forward(self, x):
        B, S, D = x.shape
        Q = self.q(x).view(B, S, self.num_heads, self.head_dim)
        K = self.k(x).view(B, S, self.num_heads, self.head_dim)
        V = self.v(x).view(B, S, self.num_heads, self.head_dim)

        # RoPE: apply to each head
        sin, cos = rope_embedding(S, self.head_dim, x.device)
        sin = sin[None, :, None, :]  # (1, seq, 1, dim)
        cos = cos[None, :, None, :]

        Q = apply_rope(Q, sin, cos)
        K = apply_rope(K, sin, cos)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.reshape(B, S, D)
        return self.o(out)
