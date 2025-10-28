import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # embed_dimがnum_headsで割り切れることを確認
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dimはnum_headsで割り切れる必要があります"
        
        # Q, K, Vのための線形層
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 最終的な出力を生成する線形層
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        # Q, K, Vを線形変換
        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)

        # Multi-Headのために次元を分割
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # スケーリングされたドット積アテンションの計算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # マスクの適用（オプション）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # ソフトマックスを適用して重みを取得
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 重みとVを掛けて出力を計算
        output = torch.matmul(attn_weights, V)

        # ヘッドを結合
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # 最終的な線形変換
        output = self.out_proj(output)
        
        return output, attn_weights

# --- 使用例 ---
embed_dim = 512
num_heads = 8
batch_size = 16
seq_len = 100

# 入力データの準備 (バッチサイズ, シーケンス長, 埋め込み次元)
q_data = torch.rand(batch_size, seq_len, embed_dim)
k_data = torch.rand(batch_size, seq_len, embed_dim)
v_data = torch.rand(batch_size, seq_len, embed_dim)

# MultiHeadAttentionのインスタンス化
custom_multihead_attn = MultiHeadAttention(embed_dim, num_heads)

# フォワードパス
attn_output, attn_weights = custom_multihead_attn(q_data, k_data, v_data)

print("出力の形状:", attn_output.shape)
