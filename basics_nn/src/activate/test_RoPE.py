import numpy as np
import matplotlib.pyplot as plt
import torch
import time

# GPUが使えるか確認
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def get_rope_encoding_matrix(max_pos=32, dim=64, base=10000.0):
    """
    RoPE の各位置 (pos) と各次元 (i) のエンコーディング行列 (cos, sin) を生成する関数
    """
    # 2次元ごとの周波数 theta_k を計算
    # dim は偶数を想定 (例: head_dim = 64)
    half_dim = dim // 2
    theta = 1.0 / (base ** (np.arange(0, half_dim) * 2 / dim))  # shape: (dim/2,)
    
    positions = np.arange(max_pos)  # shape: (max_pos,)
    
    # 角度 m * theta の行列を計算 (max_pos, dim/2)
    angles = np.outer(positions, theta)
    
    # 偶数次元に cos, 奇数次元に sin を交互に配置
    pe_matrix = np.zeros((max_pos, dim))
    pe_matrix[:, 0::2] = np.cos(angles)  # 偶数インデックス (0, 2, 4, ...)
    pe_matrix[:, 1::2] = np.sin(angles)  # 奇数インデックス (1, 3, 5, ...)
    
    return pe_matrix

# パラメータ設定
MAX_POS = 64   # 表示する最大系列長 (縦軸)
DIM = 64       # 表示する次元数 (横軸)

# 行列の取得
rope_pe = get_rope_encoding_matrix(max_pos=MAX_POS, dim=DIM)

# 描画処理
plt.figure(figsize=(10, 7))
im = plt.imshow(rope_pe, cmap='viridis', aspect='auto')

# カラーバーとラベルの設定
cbar = plt.colorbar(im)
cbar.set_label('Encoding Value', rotation=270, labelpad=15)

plt.title('Rotary Position Encoding (RoPE)', fontsize=14)
plt.xlabel('Dimension Index', fontsize=12)
plt.ylabel('Position Index', fontsize=12)

plt.tight_layout()
plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt

def simple_rope_2d(x, pos, theta_base=3.0):
    theta = pos * theta_base
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_rot = torch.stack([
        x[0] * cos_theta - x[1] * sin_theta,
        x[0] * sin_theta + x[1] * cos_theta
    ], dim=-1)
    return x_rot

# 同じ内容ベクトル
q_content = torch.tensor([1.0, 0.0])
k_content = torch.tensor([1.0, 0.0])

# 位置の範囲
seq_len = 20
positions = torch.arange(0, seq_len, dtype=torch.float32)

# Q_i @ K_j の内積行列を計算
inner_prod_matrix = torch.zeros((seq_len, seq_len))
for i in range(seq_len):
    q_rot_i = simple_rope_2d(q_content, pos=torch.tensor(float(i)), theta_base=1.0)
    for j in range(seq_len):
        k_rot_j = simple_rope_2d(k_content, pos=torch.tensor(float(j)), theta_base=1.0)
        inner_prod = torch.dot(q_rot_i, k_rot_j)
        inner_prod_matrix[i, j] = inner_prod

inner_prod_matrix_np = inner_prod_matrix.numpy()

# 相関の強さ = 内積の絶対値（正の値に正規化）
correlation_strength = np.abs(inner_prod_matrix_np)


center = seq_len // 2  # 中央を基準位置

# 距離行列（基準位置からのユークリッド距離）
dist_matrix = np.zeros((seq_len, seq_len))
for i in range(seq_len):
    for j in range(seq_len):
        dist_i = abs(i - center)
        dist_j = abs(j - center)
        dist_matrix[i, j] = np.sqrt(dist_i**2 + dist_j**2)

# 距離ごとの平均相関強度を計算
max_dist = int(np.max(dist_matrix)) + 1
dist_bins = np.arange(0, max_dist + 1)
mean_corr_by_dist = []

for d in range(max_dist):
    mask = (dist_matrix >= d) & (dist_matrix < d+1)
    if mask.sum() > 0:
        mean_val = correlation_strength[mask].mean()
    else:
        mean_val = np.nan
    mean_corr_by_dist.append(mean_val)

mean_corr_by_dist = np.array(mean_corr_by_dist)

plt.figure(figsize=(8, 4))
plt.plot(dist_bins[:-1], mean_corr_by_dist, 'o-', linewidth=2, markersize=6)
plt.xlabel('Distance from center position')
plt.ylabel('Average correlation strength |Q_i @ K_j|')
plt.title('RoPE: Correlation strength decays with distance\n(Same content vector)')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
im = plt.imshow(correlation_strength, cmap='viridis', aspect='auto', origin='lower',
                extent=[0, seq_len-1, 0, seq_len-1], vmin=0.0, vmax=1.0)
plt.colorbar(im, label='Correlation strength |Q_i @ K_j|')
plt.xlabel('Key position j')
plt.ylabel('Query position i')
plt.title('RoPE: Correlation strength matrix\n(Brighter = stronger correlation)')
plt.grid(False)
plt.show()

