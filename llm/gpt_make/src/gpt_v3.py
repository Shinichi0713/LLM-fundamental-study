class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        # ===== 論文推奨: G1ポジション・ヘッド固有ゲートの設定 =====
        # 入力次元からヘッド数(num_heads)分のスカラーを出力する
        self.gate_proj = nn.Linear(embed_dim, num_heads)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 0. ゲートスコアの計算 (G1ゲート)
        # 論文ではSDPA直後に適用されるが、ゲートの元となる値はSDPAの入力(x)
        # 形状: (batch_size, seq_len, num_heads)
        gate_scores = torch.sigmoid(self.gate_proj(x))

        # 1. Q, K, V の射影
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE を適用
        q, k = apply_rope(q, k, seq_len, x.device)

        # 3. Scaled Dot-Product Attention (SDPA)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim=-1)
        
        # SDPA出力 (y)
        # 形状: (batch_size, num_heads, seq_len, head_dim)
        y = torch.matmul(attention_weights, v)
        
        # ===== 4. ゲートの適用 (G1ポジション) =====
        # gate_scores を y の形状に合わせて変形
        # (batch, seq, heads) -> (batch, heads, seq, 1)
        gate_scores = gate_scores.transpose(1, 2).unsqueeze(-1)
        
        # ヘッドごとに独立したスカラー値を乗算 (Multiplicative Gating)
        y = y * gate_scores
        
        # 5. 出力の統合
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_linear(y)
        return output