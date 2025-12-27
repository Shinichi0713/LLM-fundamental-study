import torch
import torch.nn as nn

# サンプルデータ（バッチサイズ1, 次元数4）
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
dim = x.shape[-1]
eps = 1e-6

# --- 1. LayerNorm の手動実装 ---
mean = x.mean(dim=-1, keepdim=True)        # 平均: (1+2+3+4)/4 = 2.5
var = x.var(dim=-1, keepdim=True, unbiased=False) # 分散
x_ln = (x - mean) / torch.sqrt(var + eps)  # 平均を引いてから割る

# --- 2. RMSNorm の手動実装 ---
# 平均を計算するプロセスがない！
rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps) # 二乗平均の平方根
x_rmsn = x / rms                           # そのまま割る

print(f"Original:  {x}")
print(f"LayerNorm: {x_ln}")
print(f"RMSNorm:   {x_rmsn}")