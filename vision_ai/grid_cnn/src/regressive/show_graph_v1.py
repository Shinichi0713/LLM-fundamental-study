import matplotlib.pyplot as plt

# ログからEpochごとのAvg LossとAvg L1 Distを抽出
epochs = list(range(1, 21))
avg_losses = [
    1.6644, 0.0710, 0.0133, 0.0107, 0.0056,
    0.0031, 0.0026, 0.0014, 0.0018, 0.0022,
    0.0008, 0.0007, 0.0006, 0.0016, 0.0044,
    0.0038, 0.0008, 0.0003, 0.0002, 0.0003
]
avg_l1_dists = [
    2.1307, 0.3724, 0.1660, 0.1542, 0.1131,
    0.0783, 0.0649, 0.0517, 0.0595, 0.0642,
    0.0447, 0.0393, 0.0350, 0.0470, 0.0564,
    0.0778, 0.0410, 0.0286, 0.0229, 0.0255
]

# グラフ描画
plt.figure(figsize=(10, 6))
plt.plot(epochs, avg_losses, marker='o', label='Avg Loss')
plt.plot(epochs, avg_l1_dists, marker='s', label='Avg L1 Dist')

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and L1 Distance over Epochs')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(epochs)
plt.yscale('log')  # 値の幅が大きいので対数スケール推奨
plt.tight_layout()
plt.show()