import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model) # 内部次元を拡張 (通常2倍)

        # 入力を内部次元の2倍に投影（SSMパス用とゲートパス用）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Selective SSM (先ほど実装したクラス)
        # d_model の代わりに d_inner を使用する点に注意
        self.ssm = SelectiveSSM(
            d_model=self.d_inner, 
            d_state=d_state, 
            d_conv=d_conv, 
            is_causal=True
        )

        # 最終的な投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (Batch, SeqLen, d_model)
        """
        # 1. 入力投影と分割
        # (B, L, D) -> (B, L, 2*D_inner)
        combined_proj = self.in_proj(x)
        u, z = torch.split(combined_proj, [self.d_inner, self.d_inner], dim=-1)

        # 2. SSM パス (Selective SSM)
        # 内部で CausalConv1d -> Selective Scan が行われる
        y = self.ssm(u)

        # 3. ゲートパスとの結合 (Mamba の核心的な非線形処理)
        # y: SSMの出力, z: ゲート入力
        # y * silu(z)
        y = y * F.silu(z)

        # 4. 出力投影
        out = self.out_proj(y)

        return out

class Mamba(nn.Module):
    def __init__(self, d_model, n_layers, d_state=16, expand=2):
        super().__init__()
        # 通常、各ブロックの前に RMSNorm を配置するのが標準的
        self.layers = nn.ModuleList([
            nn.ModuleList([
                RMSNorm(d_model),
                MambaBlock(d_model, d_state=d_state, expand=expand)
            ])
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)

    def forward(self, x):
        for norm, block in self.layers:
            # 残差接続 (Residual Connection)
            x = x + block(norm(x))
        return self.norm_f(x)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight