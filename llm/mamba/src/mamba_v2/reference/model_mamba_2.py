import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import AutoTokenizer



class CausalConv1dEquivalent(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = kernel_size - 1  # 左側のみに padding を適用

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        x: [B, C, L]
        """
        # 左側に padding を適用（F.pad で明示的に左側のみパディング）
        x_padded = F.pad(x, (self.padding, 0))  # (left, right)
        # 通常の conv1d を適用
        x_conv = self.conv(x_padded)
        # 出力を入力長 L にスライス
        x_causal = x_conv[:, :, :x.size(-1)]
        return x_causal

class Mamba2Block(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        nheads=8,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_init_range=(0.9, 0.999),
        use_mem_eff_path=False,
        chunk_size=None,
        layer_idx=None,
        **factory_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.nheads = nheads
        self.ngroups = ngroups
        self.use_mem_eff_path = use_mem_eff_path
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # head次元（簡易版では固定）
        self.headdim = self.d_inner // self.nheads
        assert self.d_inner % self.nheads == 0, "d_inner must be divisible by nheads"

        d_mlp = self.d_inner
        d_in_proj = self.d_model
        d_out_proj = self.d_model