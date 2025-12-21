import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Qは従来通りヘッド数分だけ用意
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        
        # 【MQAの肝】KとVはヘッド数に関わらず「1つ分」だけ用意
        self.k_proj = nn.Linear(embed_dim, self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.head_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Q, K, V の投影
        # Q: [Batch, Seq, Heads, Head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # K, V: [Batch, Seq, 1, Head_dim]  <-- ヘッド次元が1になる
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim)
        
        # 2. 計算のために軸を入れ替え [Batch, Heads, Seq, Head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 3. スコア計算 (scaled dot-product)
        # ここで q は num_heads 個あるが、k は 1個しかない
        # PyTorchのブロードキャスト機能により、1つのkがすべてのqに対して計算される
        # scores: [Batch, Heads, Seq, Seq]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        # 4. Valueとの掛け合わせ
        # v も 1個しかないが、すべてのヘッドにブロードキャストされる
        # out: [Batch, Heads, Seq, Head_dim]
        out = torch.matmul(attn_probs, v)
        
        # 5. 結合して出力
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)

# --- 動作確認 ---
embed_dim = 128
num_heads = 8
mqa = MultiQueryAttention(embed_dim, num_heads)

x = torch.randn(1, 32, embed_dim) # [Batch, Seq, Dim]
output = mqa(x)
print(f"入力形状: {x.shape}")
print(f"出力形状: {output.shape}")