import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, num_heads=8, head_dim=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.in_proj = nn.Linear(d_model, self.d_inner * 2 + num_heads * 2 + d_state * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=False,
        )
        self.dt_bias =nn.Parameter(torch.rand(num_heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, num_heads + 1, dtype=torch.float32)))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x):
        batch, seq_len, _ = x.size()
        z_x_dt_bc = self.in_proj(x)
        z, x_inner, dt, B, C = torch.split(z_x_dt_bc, 
            [self.d_inner, self.d_inner, self.num_heads, self.d_state, self.d_state], dim=-1)
        
        # Conv1d 処理
        x_inner = x_inner.transpose(1, 2) # [B, D, L]
        x_inner = self.conv1d(x_inner)[:, :, :seq_len]
        x_inner = x_inner.transpose(1, 2) # [B, L, D]
        x_inner = F.silu(x_inner)

        
