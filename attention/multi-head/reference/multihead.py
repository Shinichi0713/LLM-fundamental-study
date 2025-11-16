import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """
        embed_dim: 入力埋め込み次元 (d_model)
        num_heads: ヘッド数 h
        head_dim = embed_dim // num_heads が整数であることが前提
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 線形投影：Q, K, V 用
        self.q_lin = nn.Linear(embed_dim, embed_dim)
        self.k_lin = nn.Linear(embed_dim, embed_dim)
        self.v_lin = nn.Linear(embed_dim, embed_dim)

        # 出力を結合した後の線形変換
        self.out_lin = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        x: (batch, seq_len, embed_dim)
        return: (batch, num_heads, seq_len, head_dim)
        """
        b, seq, _ = x.size()
        # (batch, seq_len, num_heads, head_dim)
        x = x.view(b, seq, self.num_heads, self.head_dim)
        # (batch, num_heads, seq_len, head_dim)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x):
        """
        x: (batch, num_heads, seq_len, head_dim)
        return: (batch, seq_len, embed_dim)
        """
        b, h, seq, hd = x.size()
        # (batch, seq_len, num_heads, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(b, seq, h * hd)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch, seq_len, embed_dim)
        mask: optional, broadcastable to (batch, num_heads, seq_q, seq_k)
              mask positions to be ignored should be True (or 1) where masked.
        returns:
          out: (batch, seq_len, embed_dim)
          attn_weights: (batch, num_heads, seq_q, seq_k)
        """
        # 1) 線形変換してヘッドに分割
        Q = self.q_lin(query)  # (b, seq_q, embed_dim)
        K = self.k_lin(key)    # (b, seq_k, embed_dim)
        V = self.v_lin(value)  # (b, seq_k, embed_dim)

        Q = self._split_heads(Q)  # (b, h, seq_q, head_dim)
        K = self._split_heads(K)  # (b, h, seq_k, head_dim)
        V = self._split_heads(V)  # (b, h, seq_k, head_dim)

        # 2) スケールド・ドットプロダクトを計算
        # scores: (b, h, seq_q, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3) マスク（あれば）を適用
        if mask is not None:
            # mask が bool なら True をマスク（値を -inf に）
            # ここでは mask==True の位置をマスクする前提
            # mask を (b, 1, seq_q, seq_k) などにしておくとブロードキャストされる
            scores = scores.masked_fill(mask, float('-inf'))

        # 4) softmax と dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 5) attention を value に乗じて出力を得る
        # context: (b, h, seq_q, head_dim)
        context = torch.matmul(attn, V)

        # 6) ヘッドを結合して最終線形出力
        out = self._combine_heads(context)  # (b, seq_q, embed_dim)
        out = self.out_lin(out)             # (b, seq_q, embed_dim)

        return out, attn
