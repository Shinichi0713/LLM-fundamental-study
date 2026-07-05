import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling

# Cora データセットの読み込み
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

print(f"ノード数: {data.num_nodes}")
print(f"エッジ数: {data.edge_index.size(1)}")
print(f"特徴量次元: {data.num_features}")

# 既存のエッジを正例とする
pos_edge_index = data.edge_index  # 形状: [2, num_edges]

# 負例エッジをサンプリング（既存エッジと被らないように）
neg_edge_index = negative_sampling(
    edge_index=pos_edge_index,
    num_nodes=data.num_nodes,
    num_neg_samples=pos_edge_index.size(1)  # 正例と同数
)

print(f"正例エッジ数: {pos_edge_index.size(1)}")
print(f"負例エッジ数: {neg_edge_index.size(1)}")

import torch
import matplotlib.pyplot as plt
import numpy as np

print(f"test_data.x.shape: {test_data.x.shape}")  # 例: [num_nodes, num_features]

# 先頭から20ノード分の特徴ベクトルを抽出
num_samples = 20
sample_x = test_data.x[:num_samples].cpu().numpy()  # shape: [20, num_features]

print(f"サンプルデータ形状: {sample_x.shape}")

plt.figure(figsize=(12, 6))

# ヒートマップ描画
im = plt.imshow(sample_x, cmap='coolwarm', aspect='auto', interpolation='nearest')

# カラーバー
plt.colorbar(im, label='feature value')

# 軸ラベル
plt.xlabel('feature order')
plt.ylabel('node no (0-19)')
plt.title('heatmap test_data.x \n (red: high, blue: low value)')

# グリッド線（任意）
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()