import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. 真の因果畳み込み層の定義 ---
class MambaCausalConv(nn.Module):
    def __init__(self, d_model, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # Mambaの構成を模して、各チャンネル独立（groups=d_model）の1D畳み込み
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=0 # 手動でパディングするため0
        )
        
    def forward(self, x):
        # x: (Batch, Channels, SeqLen)
        # 左側だけにパディングを施すことで因果性を担保
        x_padded = nn.functional.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x_padded)

# --- 2. データの準備 ---
seq_len = 20
d_model = 1  # 可視化のため1チャンネル
model = MambaCausalConv(d_model, kernel_size=4)
model.eval()

# 入力データ (Batch=1, Channel=1, SeqLen=20)
input_tensor = torch.sin(torch.linspace(0, 4 * np.pi, seq_len)).view(1, 1, -1)
with torch.no_grad():
    # 全体の出力を計算しておく
    full_output = model(input_tensor).squeeze().numpy()

input_data = input_tensor.squeeze().numpy()

# --- 3. 可視化のセットアップ ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)

def setup_plot():
    ax1.set_xlim(-1, seq_len)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_title("Input & Actual PyTorch Conv1d Kernel Scope")
    ax2.set_xlim(-1, seq_len)
    ax2.set_ylim(full_output.min() - 0.5, full_output.max() + 0.5)
    ax2.set_title("Actual Conv1d Output (Feature Map)")

# プロット要素
input_line, = ax1.plot(range(seq_len), input_data, 'o', color='lightgray')
kernel_rect = plt.Rectangle((0, -1.2), 4, 2.4, color='green', alpha=0.2, label='Conv Kernel')
ax1.add_patch(kernel_rect)
active_dots, = ax1.plot([], [], 'o', color='green')
out_line, = ax2.plot([], [], 's-', color='red', markersize=4)

def update(frame):
    # カーネルの範囲を表示 (現在のフレームがカーネルの右端)
    k_size = model.kernel_size
    rect_x = frame - (k_size - 1)
    kernel_rect.set_x(rect_x)
    kernel_rect.set_width(k_size)
    
    # 実際に計算に使われている入力点
    indices = np.arange(max(0, rect_x), frame + 1).astype(int)
    active_dots.set_data(indices, input_data[indices])
    
    # そこまでの出力を描画
    out_line.set_data(np.arange(frame + 1), full_output[:frame + 1])
    
    return kernel_rect, active_dots, out_line

setup_plot()
ani = FuncAnimation(fig, update, frames=range(seq_len), blit=True, interval=300)

# 保存する場合は以下のコメントを外してください
# ani.save('real_mamba_conv.mp4', writer='ffmpeg')
plt.show()