import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedSDPA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # ゲート用の学習パラメータ (W_theta)
        # 各ヘッドが独自のゲート値を持てるように、ヘッド数分の出力を設定
        self.gate_proj = nn.Linear(d_model, num_heads)

    def forward(self, query, key, value, x_hidden):
        """
        query, key, value: (batch, heads, seq_len, d_head)
        x_hidden: (batch, seq_len, d_model) - プレ・ノーマライゼーション後の隠れ状態
        """
        # 1. 標準的な Scaled Dot-Product Attention (SDPA)
        # 出力 Y の形状: (batch, heads, seq_len, d_head)
        y = F.scaled_dot_product_attention(query, key, value)

        # 2. ゲートスコアの計算 (G1: SDPA出力ゲート)
        # x_hidden から各ヘッドごとのゲートスコアを生成
        # scores 形状: (batch, seq_len, num_heads)
        gate_scores = torch.sigmoid(self.gate_proj(x_hidden))

        # 3. ゲートの適用
        # 形状を合わせるためにゲートスコアを転置し、次元を拡張
        # (batch, heads, seq_len, 1)
        gate_scores = gate_scores.transpose(1, 2).unsqueeze(-1)
        
        # アテンション出力 Y にシグモイド・ゲートを乗算 (Multiplicative)
        y_gated = y * gate_scores
        
        return y_gated

# --- 全体構造のイメージ ---
class GatedAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = GatedSDPA(d_model, num_heads)
        self.w_o = nn.Linear(d_model, d_model) # 出力プロジェクション
        self.norm = nn.RMSNorm(d_model) # プレ・ノーマライゼーション用

    def forward(self, x):
        # プレ・ノーマライゼーション後の状態 X をゲート計算に利用
        x_norm = self.norm(x)
        
        # 本来はここで Q, K, V プロジェクションを行う (中略)
        # y = self.mha(q, k, v, x_norm)
        
        # 最後に連結して W_o をかける
        # out = self.w_o(y_recombined)
        pass