import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.inner_dim = int(expand * d_model)
        
        # 入力を2つの経路（主経路とゲート経路）に投影
        self.in_proj = nn.Linear(d_model, self.inner_dim * 2, bias=False)
        
        # 1. 因果1次元畳み込み (局所的な特徴抽出)
        # チャンネルごとに独立して計算するDepthwise Conv
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1, # 因果性を保つためのパディング
        )
        
        # 2. Selective SSM 用のパラメータ生成 (取捨選択の鍵)
        # 入力 x に基づいて SSM のパラメータ (Δ, B, C) を動的に生成する
        self.x_proj = nn.Linear(self.inner_dim, dt_rank + d_state * 2, bias=False)
        
        # 3. ゲート経路用の活性化関数
        self.act = nn.SiLU()
        
        # 出力投影
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        # x: (Batch, SeqLen, d_model)
        batch, seq_len, _ = x.shape
        
        # 入力投影と分割
        projected = self.in_proj(x) # (B, L, 2*inner)
        x, res = projected.chunk(2, dim=-1) # 主経路 x と ゲート経路 res
        
        # --- 畳み込みフェーズ ---
        x = x.transpose(1, 2) # (B, inner, L) へ変換
        # 左パディングにより因果性を確保し、末尾をカット
        x = self.conv1d(x)[:, :, :seq_len] 
        x = self.act(x)
        x = x.transpose(1, 2) # (B, L, inner) に戻す
        
        # --- Selective SSM フェーズ (概念的) ---
        # 本来はここで Δ, B, C を計算し、離散化 SSM を適用します
        # 簡略化のため、ここではゲートとの乗算をメインに示します
        
        # ゲートによる取捨選択
        # ゲート経路(res)をSiLUで活性化し、SSM後の信号と掛け合わせる
        y = x * self.act(res)
        
        # 最終出力
        return self.out_proj(y)
    
