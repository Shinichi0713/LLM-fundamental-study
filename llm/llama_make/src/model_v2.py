"""
Llama (Inference-only) in pure NumPy.
No PyTorch, TensorFlow, JAX, or transformers library is used.
"""

import numpy as np
import math
import re
from typing import List, Tuple, Optional, Dict


# -----------------------------------------------------------------------------
# Activations
# -----------------------------------------------------------------------------

def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish: x * sigmoid(x)"""
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# -----------------------------------------------------------------------------
# RMSNorm
# -----------------------------------------------------------------------------

class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (..., dim)
        variance = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        x = x * np.reciprocal(np.sqrt(variance + self.eps))
        return (x * self.weight).astype(x.dtype)


# -----------------------------------------------------------------------------
# RoPE (Rotary Positional Embedding)
# -----------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute cos/sin for rotary embeddings.
    Returns (cos, sin) each of shape (end, dim//2)
    """
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(end, dtype=np.float32)
    freqs = np.outer(t, freqs)  # (end, dim//2)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    return cos, sin


def apply_rotary_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Apply rotary embedding to x.
    x: (seq_len, n_heads, head_dim)
    cos, sin: (seq_len, head_dim//2)
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    # reshape to interleave pairs: (..., d//2, 2) -> rotate
    # Here we use the simplified approach: rotate consecutive pairs
    return np.concatenate([
        x1 * cos[:, None, :] - x2 * sin[:, None, :],
        x1 * sin[:, None, :] + x2 * cos[:, None, :]
    ], axis=-1)


# -----------------------------------------------------------------------------
# Attention (Grouped Query Attention with KV Cache)
# -----------------------------------------------------------------------------

class Attention:
    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int] = None):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        # Weight matrices (stored transposed for matmul: out = x @ W.T)
        self.wq = np.random.randn(dim, dim).astype(np.float32) * 0.02
        self.wk = np.random.randn(self.n_kv_heads * self.head_dim, dim).astype(np.float32) * 0.02
        self.wv = np.random.randn(self.n_kv_heads * self.head_dim, dim).astype(np.float32) * 0.02
        self.wo = np.random.randn(dim, dim).astype(np.float32) * 0.02

        # KV cache
        self.cache_k: Optional[np.ndarray] = None
        self.cache_v: Optional[np.ndarray] = None
        self.cache_len: int = 0

    def forward(
        self,
        x: np.ndarray,
        start_pos: int,
        cos: np.ndarray,
        sin: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        bsz, seqlen, _ = x.shape

        xq = x @ self.wq.T  # (bsz, seqlen, dim)
        xk = x @ self.wk.T  # (bsz, seqlen, n_kv_heads * head_dim)
        xv = x @ self.wv.T  # (bsz, seqlen, n_kv_heads * head_dim)

        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply RoPE to queries and keys
        xq = apply_rotary_emb(xq, cos, sin)
        xk = apply_rotary_emb(xk, cos, sin)

        # Update KV cache
        if start_pos == 0:
            self.cache_k = xk
            self.cache_v = xv
            self.cache_len = seqlen
        else:
            self.cache_k = np.concatenate([self.cache_k, xk], axis=1)
            self.cache_v = np.concatenate([self.cache_v, xv], axis=1)
            self.cache_len += seqlen

        keys = self.cache_k   # (bsz, cache_len, n_kv_heads, head_dim)
        values = self.cache_v # (bsz, cache_len, n_kv_heads, head_dim)

        # Repeat k/v heads if n_kv_heads < n_heads (Grouped Query Attention)
        if self.n_rep > 1:
            keys = np.repeat(keys, self.n_rep, axis=2)   # (bsz, cache_len, n_heads, head_dim)
            values = np.repeat(values, self.n_rep, axis=2)

        # Transpose for attention: (bsz, n_heads, seqlen, head_dim)
        xq = np.transpose(xq, (0, 2, 1, 3))
        keys = np.transpose(keys, (0, 2, 1, 3))
        values = np.transpose(values, (0, 2, 1, 3))

        # Scaled dot-product attention
        scores = (xq @ np.transpose(keys, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = softmax(scores, axis=-1)

        output = scores @ values  # (bsz, n_heads, seqlen, head_dim)
        output = np.transpose(output, (0, 2, 1, 3)).reshape(bsz, seqlen, -1)

        return output @ self.wo.T


# -----------------------------------------------------------------------------
# FeedForward (SwiGLU)
# -----------------------------------------------------------------------------

class FeedForward:
    def __init__(self, dim: int, hidden_dim: int):
        self.w1 = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02  # gate
        self.w2 = np.random.randn(dim, hidden_dim).astype(np.float32) * 0.02  # down
        self.w3 = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02  # up

    def forward(self, x: np.ndarray) -> np.ndarray:
        # SwiGLU: silu(gate) * up -> down
        gate = x @ self.w1.T
        up = x @ self.w3.T
        hidden = silu(gate) * up
        return hidden @ self.w2.T


# -----------------------------------------------------------------------------
# Transformer Block
# -----------------------------------------------------------------------------

class TransformerBlock:
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int, norm_eps: float):
        self.attention = Attention(dim, n_heads, n_kv_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: np.ndarray,
        start_pos: int,
        cos: np.ndarray,
        sin: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        h = x + self.attention.forward(self.attention_norm.forward(x), start_pos, cos, sin, mask)
        out = h + self.feed_forward.forward(self.ffn_norm.forward(h))
        return out


# -----------------------------------------------------------------------------
# Llama Model
# -----------------------------------------------------------------------------

class LlamaModel:
    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 512,
        norm_eps: float = 1e-6,
        theta: float = 10000.0,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.hidden_dim = hidden_dim or int(2 * (4 * dim) / 3)  # heuristic like LLaMA
        self.max_seq_len = max_seq_len
        self.head_dim = dim // n_heads

        self.tok_embeddings = np.random.randn(vocab_size, dim).astype(np.float32) * 0.02
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = np.random.randn(vocab_size, dim).astype(np.float32) * 0.02

        self.layers = [
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=self.n_kv_heads,
                hidden_dim=self.hidden_dim,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ]

        # Precompute RoPE frequencies
        self._cos, self._sin = precompute_freqs_cis(self.head_dim, max_seq_len, theta)

    def forward(self, tokens: List[int], start_pos: int = 0) -> np.ndarray:
        """
        tokens: list of token IDs
        Returns logits of shape (seqlen, vocab_size)
        """
        seqlen = len(tokens)
        x = self.tok_embeddings[np.array(tokens)]  # (seqlen, dim)
        x = x[None, ...]  # (1, seqlen, dim)

        cos = self._cos[start_pos : start_pos + seqlen]
        sin = self._sin[start_pos : start_pos + seqlen]

        # Causal mask for training-like behavior during first forward
        mask = None
        if seqlen > 1:
            # (1, 1, seqlen, cache_len+seqlen) shaped mask
            # For simplicity, create a triangular mask for the current sequence
            mask = np.triu(np.ones((seqlen, seqlen)), k=1) * -1e9
            mask = mask[None, None, ...]  # (1, 1, seqlen, seqlen)

        for layer in self.layers:
            x = layer.forward(x, start_pos, cos, sin, mask)

        x = self.norm.forward(x)
        logits = x @ self.output.T  # (1, seqlen, vocab_size)
        return logits[0]  # (seqlen, vocab_size)

    def reset_cache(self):
        """Reset KV caches for all layers."""
        for layer in self.layers:
            layer.attention.cache_k = None
            layer.attention.cache_v = None
            layer.attention.cache_len = 0


# -----------------------------------------------------------------------------
# Simple Character-level Tokenizer
# -----------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.itos[t] for t in tokens])


# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------

def generate(
    model: LlamaModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> str:
    model.reset_cache()
    tokens = tokenizer.encode(prompt)

    for i in range(max_new_tokens):
        start_pos = 0 if i == 0 else len(tokens)
        logits = model.forward(tokens if i == 0 else [tokens[-1]], start_pos=start_pos)
        next_logit = logits[-1] / temperature

        if top_k is not None:
            indices = np.argpartition(-next_logit, top_k)[:top_k]
            mask = np.ones_like(next_logit) * -1e9
            mask[indices] = 0
            next_logit = next_logit + mask

        probs = softmax(next_logit)
        next_token = int(np.random.choice(len(probs), p=probs))
        tokens.append(next_token)

    return tokenizer.decode(tokens)


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Small demo corpus
    text = (
        "Hello world! This is a pure NumPy implementation of the Llama architecture. "
        "Llama uses RMSNorm, RoPE, SwiGLU, and grouped query attention. "
        "It contains no PyTorch, TensorFlow, JAX, or transformers code. "
        "The model runs inference using only NumPy and the Python standard library. "
    ) * 10  # repeat to get a bit more vocab diversity

    tokenizer = CharTokenizer(text)

    model = LlamaModel(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=4,       # Grouped Query Attention
        hidden_dim=688,     # ~2.7 * dim, typical LLaMA ratio
        max_seq_len=256,
        norm_eps=1e-6,
        theta=10000.0,
    )

    prompt = "Hello world!"
    print(f"Prompt: {prompt!r}")
    print("Generating...\n")

    out = generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=40)
    print(out)