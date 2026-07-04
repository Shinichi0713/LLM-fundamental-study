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

