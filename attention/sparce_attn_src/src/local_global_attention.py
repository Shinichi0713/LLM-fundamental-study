import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def build_sparse_attention_mask(
    seq_len: int,
    window: int,
    global_token_indices: list[int],
):
    """
    Sparse Attention mask with explicit Local + Global Attention
    
    mask[i, j] = 0      -> attention allowed
    mask[i, j] = -inf   -> attention blocked
    """

    mask = torch.full((seq_len, seq_len), float("-inf"))

    is_global = torch.zeros(seq_len, dtype=torch.bool)
    is_global[global_token_indices] = True

    for i in range(seq_len):
        if is_global[i]:
            # Global token attends to ALL tokens
            mask[i, :] = 0
        else:
            # 1. Local attention
            left = max(0, i - window)
            right = min(seq_len, i + window + 1)
            mask[i, left:right] = 0

            # 2. Global attention (explicit)
            mask[i, is_global] = 0

    return mask

def attention_with_weights(Q, K, V, attn_mask):
    d_k = Q.size(-1)

    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores + attn_mask

    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ V

    return output, attn_weights

def plot_attention_map(attn_weights, title="Attention Map"):
    attn = attn_weights.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(attn, aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Key Token Index")
    plt.ylabel("Query Token Index")
    plt.title(title)
    plt.tight_layout()
    plt.show()


seq_len = 32
d_model = 64
window = 2

CLS_INDEX = 0
global_tokens = [CLS_INDEX]

torch.manual_seed(0)

Q = torch.randn(seq_len, d_model)
K = torch.randn(seq_len, d_model)
V = torch.randn(seq_len, d_model)

mask = build_sparse_attention_mask(
    seq_len=seq_len,
    window=window,
    global_token_indices=global_tokens,
)

output, attn_weights = attention_with_weights(Q, K, V, mask)

plot_attention_map(attn_weights)
