import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

# --- パラメータ設定 ---
seq_len = 15
kernel_size = 4
input_data = np.sin(np.linspace(0, 4 * np.pi, seq_len)) + np.random.normal(0, 0.1, seq_len)
output_data = np.zeros(seq_len)

# 因果畳み込みのシミュレーション（簡易版）
# 実際はパディング（kernel_size - 1）を行い、未来を見ないようにシフトします
padded_input = np.concatenate([np.zeros(kernel_size - 1), input_data])
weights = np.exp(np.linspace(-1, 0, kernel_size))  # 指数的な重み
weights /= weights.sum()

for i in range(seq_len):
    window = padded_input[i : i + kernel_size]
    output_data[i] = np.dot(window, weights)

# --- 可視化のセットアップ ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)

def setup_plot():
    ax1.set_xlim(-1, seq_len)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_title("Input Sequence & Causal Convolution Kernel")
    ax1.set_ylabel("Amplitude")
    
    ax2.set_xlim(-1, seq_len)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_title("Output Sequence (Feature Map)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Amplitude")

# 要素の初期化
input_dots, = ax1.plot(range(seq_len), input_data, 'o', color='gray', alpha=0.3)
kernel_rect = plt.Rectangle((0, -1.2), kernel_size, 2.4, color='orange', alpha=0.3, label='Kernel')
ax1.add_patch(kernel_rect)
active_input_dots, = ax1.plot([], [], 'o', color='orange')
current_output_dots, = ax2.plot([], [], 's', color='blue', label='Generated Output')

# アニメーション更新関数
def update(frame):
    # カーネルの位置（因果性を保つため、右端が現在のフレームに来るように配置）
    rect_x = frame - (kernel_size - 1)
    kernel_rect.set_x(rect_x)
    
    # 現在のウィンドウ内の入力点
    indices = np.arange(max(0, rect_x), frame + 1).astype(int)
    active_input_dots.set_data(indices, input_data[indices])
    
    # 出力点の更新
    output_indices = np.arange(frame + 1)
    current_output_dots.set_data(output_indices, output_data[output_indices])
    
    return kernel_rect, active_input_dots, current_output_dots

# アニメーションの生成
setup_plot()
ani = FuncAnimation(fig, update, frames=range(seq_len), blit=True, interval=500)

# 保存（ffmpegが必要な場合があります）
ani.save('causal_conv_mamba.mp4', writer='ffmpeg')
plt.show()