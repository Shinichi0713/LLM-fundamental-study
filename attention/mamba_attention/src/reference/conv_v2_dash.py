import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class CausalConv1d(nn.Module):
  def __init__(self, d_model, kernel_size):
    super().__init__()
    self.kernel_size
    self.conv1d = nn.Conv1d(
        in_channnels=d_model,
        out_channels=d_model,
        kernel_size=kernel_size,
        groups=d_model,
        padding=0
    )

    def forward(self, x):
      # ここでカーネルの範囲を制御する
      x_padded = nn.functional.pad(x, (self.kernel_size - 1, 0))
      return self.conv1d(x_padded)


seq_len = 20
d_model = 1  # 可視化のため1チャンネル
model = CausalConv1d(d_model, kernel_size=4)
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


