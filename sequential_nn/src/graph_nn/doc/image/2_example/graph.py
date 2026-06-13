import matplotlib.pyplot as plt

# 学習ログ（エポック, 損失, 精度）
epochs = [1, 20, 40, 60, 80, 100]
train_loss = [11.0581, 0.4705, 0.1629, 0.0796, 0.0372, 0.0153]
test_acc = [13.6, 68.5, 74.1, 79.4, 79.5, 78.4]

fig, ax1 = plt.subplots(figsize=(8, 5))

# 損失（左軸）
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(epochs, train_loss, marker='o', color=color, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')  # 損失が大きく変化するため対数スケール推奨

# 精度（右軸）
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Test Accuracy (%)', color=color)
ax2.plot(epochs, test_acc, marker='s', color=color, label='Test Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 100)

plt.title('GNN Training Progress (Digits Graph)')
fig.tight_layout()
plt.show()