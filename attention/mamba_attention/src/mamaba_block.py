import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # 入力投影（入力次元を拡張）
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # 短期記憶のための1次元畳み込み
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # 選択的メカニズムのためのパラメータ投影（xからDelta, B, Cを生成）
        self.x_proj = nn.Linear(self.d_inner, 1 + d_state * 2, bias=False)
        
        # Delta（ステップサイズ）のための線形投影
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # 状態空間モデルの行列 A (学習可能なパラメータ)
        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))

        # 出力投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        (batch, seq_len, d_model) = x.shape
        
        # 1. 入力投影と分岐
        x_and_res = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        (x, res) = x_and_res.split(split_size=self.d_inner, dim=-1)

        # 2. 畳み込みパス (Conv1d)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x) # 活性化関数

        # 3. 選択的SSMの実行
        y = self.selective_ssm(x)

        # 4. ゲートパス（res）との結合と出力
        y = y * F.silu(res)
        output = self.out_proj(y)

        return output

    def selective_ssm(self, x):
        """
        x に基づいて A, B, C を離散化し、状態更新を行う
        """
        (batch, seq_len, d_inner) = x.shape
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # 入力 x から Delta, B, C を動的に生成 (Selection)
        x_dbl = self.x_proj(x)  # (batch, seq_len, 1 + 2*d_state)
        dt, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        
        # Delta の適用
        dt = F.softplus(self.dt_proj(dt))  # (batch, seq_len, d_inner)

        # 離散化 (Discretization) と スキャン計算
        # 本来は並列スキャン(Parallel Scan)を用いるが、理解のために逐次的な概念で記述
        # y = scan(dt, A, B, C, x)
        
        # 簡易的な離散化の適用例:
        # dA = exp(dt * A)
        # dB = dt * B
        
        # --- ここで実際にはカスタムCUDAカーネルや associative scan が呼ばれる ---
        # 今回は形状維持のためのプレースホルダとして入力を返す
        y = x 
        return y