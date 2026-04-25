import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # KV キャッシュ用バッファ（バッチサイズは後で拡張）
        self.register_buffer("cache_k", torch.zeros(max_seq_len, d_model))
        self.register_buffer("cache_v", torch.zeros(max_seq_len, d_model))
        self.cache_len = 0  # 現在キャッシュされているトークン数

    def reset_cache(self):
        """キャッシュをクリア"""
        self.cache_len = 0
        self.cache_k.zero_()
        self.cache_v.zero_()

    def forward(self, x, use_cache=False):
        """
        x: (batch_size, seq_len, d_model)
        use_cache: True なら過去の KV を再利用
        """
        B, T, C = x.shape
        device = x.device

        # 現在のステップで計算する Key/Value
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        if use_cache and self.cache_len > 0:
            # 過去のキャッシュと結合
            past_k = self.cache_k[: self.cache_len].unsqueeze(0).expand(B, -1, -1)
            past_v = self.cache_v[: self.cache_len].unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([past_k, k], dim=1)  # (B, cache_len + T, C)
            v = torch.cat([past_v, v], dim=1)  # (B, cache_len + T, C)

        # キャッシュを更新（新しいトークン分を追加）
        if use_cache:
            # 単純化のためバッチ次元を無視して保存（バッチ対応は必要に応じて拡張）
            new_k = k[:, -T:, :].detach().clone()
            new_v = v[:, -T:, :].detach().clone()
            self.cache_k[self.cache_len : self.cache_len + T] = new_k.mean(dim=0)
            self.cache_v[self.cache_len : self.cache_len + T] = new_v.mean(dim=0)
            self.cache_len += T

        # Multi-head に分割
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, k.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T_past, D)
        v = v.view(B, v.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T_past, D)

        # Attention スコア計算（簡略化のためマスクなし）
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T_past)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (B, H, T, D)

        # 結合して出力
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.out_proj(out)
        return out


class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.ln1(x), use_cache=use_cache)
        x = x + self.ff(self.ln2(x))
        return x


class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len=1024):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def reset_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()

    def forward(self, idx, use_cache=False):
        """
        idx: (B, T) トークンID
        use_cache: True なら KV キャッシュを利用
        """
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embed(idx)  # (B, T, C)
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)  # (B, T, C)

        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x, use_cache=use_cache)

        x = self.ln_out(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits


def generate_with_kv_cache(model, start_tokens, max_new_tokens, context_size=1024):
    """
    KV キャッシュを使った自己回帰生成
    """
    model.eval()
    model.reset_cache()

    idx = start_tokens  # (1, T0)
    for _ in range(max_new_tokens):
        # コンテキストサイズを超えないように切り詰め
        idx_cond = idx[:, -context_size:] if idx.size(1) > context_size else idx

        # 初回は use_cache=False で全トークンを計算し、KV をキャッシュ
        # 2回目以降は use_cache=True で新しいトークンだけを入力
        use_cache = (idx.size(1) > 1)
        logits = model(idx_cond, use_cache=use_cache)  # (1, T, vocab_size)

        # 最後のトークンのロジットから次トークンをサンプリング（ここでは貪欲）
        next_logits = logits[:, -1, :]  # (1, vocab_size)
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (1, 1)

        idx = torch.cat([idx, next_token], dim=1)  # (1, T+1)

    return idx


# 使用例
if __name__ == "__main__":
    vocab_size = 1000
    d_model = 128
    n_heads = 4
    n_layers = 2
    max_seq_len = 256

    model = SimpleGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len)

    # ダミーの開始トークン
    start_tokens = torch.tensor([[1, 2, 3]])  # (1, 3)

    # KV キャッシュを使って 10 トークン生成
    generated = generate_with_kv_cache(model, start_tokens, max_new_tokens=10)
    print("Generated token IDs:", generated.tolist())