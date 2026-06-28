import matplotlib.pyplot as plt
import numpy as np

# 学習ログ（Epoch, Loss, Accuracy）
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
losses = [3.2473, 20.0477, 15.6726, 9.5198, 3.7745, 1.5388, 0.3015, 0.0216, 0.0001, 0.0000]
accuracies = [0.0, 10.0, 10.0, 20.0, 60.0, 70.0, 90.0, 100.0, 100.0, 100.0]

# グラフの設定
fig, ax1 = plt.subplots(figsize=(8, 5))

# Loss（左軸）
color_loss = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color_loss)
ax1.plot(epochs, losses, marker='o', color=color_loss, label='Loss')
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.grid(True, linestyle='--', alpha=0.7)

# Accuracy（右軸）
ax2 = ax1.twinx()
color_acc = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color_acc)
ax2.plot(epochs, accuracies, marker='s', color=color_acc, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color_acc)

# タイトルと凡例
plt.title('Training Loss and Accuracy over Epochs')
fig.tight_layout()
plt.show()