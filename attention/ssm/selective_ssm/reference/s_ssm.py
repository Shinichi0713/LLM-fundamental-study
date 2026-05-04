import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state, d_inner=None, dt_rank="auto", dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner or d_model
        self.dt_rank = dt_rank if dt_rank != "auto" else max(1, d_model // 16)

        # 入力からパラメータを生成するための線形層
        self.in_proj = nn.Linear(d_model, d_inner, bias=False)
        
        # Δ (time step) を生成する部分
        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        
        # 固定の A 行列（対角行列としてパラメータ化）
        self.A_log = nn.Parameter(torch.log(torch.randn(d_model, d_state) * 0.02))
        
        # D (skip connection) パラメータ
        self.D = nn.Parameter(torch.randn(d_model))

        # dt のスケーリング用
        self.dt_min = dt_min
        self.dt_max = dt_max

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        return: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # 入力射影
        x_in = self.in_proj(x)  # (batch, seq_len, d_inner)
        
        # パラメータ生成 (Δ, B, C)
        # x_proj_out: (batch, seq_len, dt_rank + d_state*2)
        x_proj_out = self.x_proj(x_in)
        dt, B, C = torch.split(
            x_proj_out,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        # Δ をスケーリング
        dt = self.dt_proj(dt)  # (batch, seq_len, d_model)
        dt = F.softplus(dt) * (self.dt_max - self.dt_min) + self.dt_min
        
        # A を取得（対角行列として扱う簡略化）
        A = -torch.exp(self.A_log)  # (d_model, d_state)
        
        # 離散化パラメータの計算（簡易版：1次近似）
        # 本来は exp(ΔA) を使うが、ここでは簡略化
        dA = torch.einsum('bse,ds->bdse', dt, A)  # (batch, d_model, seq_len, d_state)
        bar_A = torch.exp(dA)  # 簡易的な exp(ΔA) の近似
        bar_B = dA  # 簡易的な (exp(ΔA)-I)/A * ΔB の近似（Bは後で乗算）
        
        # 状態更新（scan）の簡易実装
        # ここではバッチごとに for ループで書く（実際の Mamba は高速カーネルを使用）
        h = torch.zeros(batch, d_model, d_state, device=x.device)  # 初期状態
        outputs = []
        
        for t in range(seq_len):
            # bar_A_t: (batch, d_model, d_state)
            bar_A_t = bar_A[:, :, t, :]
            # bar_B_t: (batch, d_model, d_state)
            bar_B_t = bar_B[:, :, t, :]
            # B_t: (batch, d_state)
            B_t = B[:, t, :]
            # C_t: (batch, d_state)
            C_t = C[:, t, :]
            
            # 状態更新: h_t = bar_A_t * h_{t-1} + bar_B_t * (B_t * u_t)
            u_t = x_in[:, t, :]  # (batch, d_inner)
            # B_t を d_model 次元にブロードキャストして乗算（簡略化）
            B_broadcast = B_t.unsqueeze(1)  # (batch, 1, d_state)
            input_term = bar_B_t * B_broadcast * u_t.unsqueeze(-1)  # (batch, d_model, d_state)
            h = bar_A_t * h + input_term.sum(dim=1, keepdim=True)  # 簡易的な集約
            
            # 出力: y_t = C_t * h_t + D * u_t
            y_t = torch.einsum('bds,bs->bd', h, C_t.unsqueeze(1)) + self.D * u_t
            outputs.append(y_t.unsqueeze(1))
        
        y = torch.cat(outputs, dim=1)  # (batch, seq_len, d_model)
        return y

# 使用例
if __name__ == "__main__":
    d_model = 64
    d_state = 16
    seq_len = 100
    batch = 4
    
    model = SelectiveSSM(d_model=d_model, d_state=d_state)
    x = torch.randn(batch, seq_len, d_model)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)