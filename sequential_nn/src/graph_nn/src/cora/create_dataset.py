import torch
from torch_geometric.datasets import Planetoid

# Cora データセットのダウンロード
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

print(f"データセット名: {dataset}")
print(f"グラフ数: {len(dataset)}")
print(f"ノード数: {data.num_nodes}")
print(f"エッジ数: {data.num_edges}")
print(f"特徴量次元: {data.num_features}")
print(f"クラス数: {dataset.num_classes}")
print(f"学習用マスク数: {data.train_mask.sum().item()}")
print(f"検証用マスク数: {data.val_mask.sum().item()}")
print(f"テスト用マスク数: {data.test_mask.sum().item()}")

import torch
from torch_geometric.datasets import Planetoid

# Cora データセットの読み込み
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

print("=== Cora データセットの基本情報 ===")
print(f"データセット名: {dataset}")
print(f"ノード数: {data.num_nodes}")
print(f"エッジ数: {data.num_edges}")
print(f"特徴量次元: {data.num_features}")
print(f"クラス数: {dataset.num_classes}")
print(f"学習用ノード数: {data.train_mask.sum().item()}")
print(f"検証用ノード数: {data.val_mask.sum().item()}")
print(f"テスト用ノード数: {data.test_mask.sum().item()}")
print()

print("=== ノード特徴量 (x) のサンプル ===")
print(f"x.shape: {data.x.shape}")
print("最初の5ノードの特徴ベクトル（非ゼロ要素のみ表示）:")
for i in range(5):
    nonzero_idx = data.x[i].nonzero(as_tuple=True)[0]
    print(f"ノード {i}: 非ゼロ特徴インデックス {len(nonzero_idx)}個, 例: {nonzero_idx[:5].tolist()}")
print()

print("=== エッジ情報 (edge_index) のサンプル ===")
print(f"edge_index.shape: {data.edge_index.shape}")
print("最初の10エッジ:")
for i in range(min(10, data.edge_index.size(1))):
    src, dst = data.edge_index[:, i]
    print(f"エッジ {i}: 論文 {src.item()} → 論文 {dst.item()}")
print()

print("=== ノードラベル (y) のサンプル ===")
print(f"y.shape: {data.y.shape}")
print("最初の10ノードのラベル:")
for i in range(10):
    print(f"ノード {i}: ラベル {data.y[i].item()} (クラスID)")
print("クラス名の例（ID 0〜6 に対応）:")
print("0: Case-Based, 1: Genetic Algorithms, 2: Neural Networks, 3: Probabilistic Methods, 4: Reinforcement Learning, 5: Rule Learning, 6: Theory")
print()

print("=== マスクのサンプル (train/val/test) ===")
print(f"train_mask.shape: {data.train_mask.shape}")
print(f"val_mask.shape: {data.val_mask.shape}")
print(f"test_mask.shape: {data.test_mask.shape}")
print("最初の10ノードのマスク状態:")
for i in range(10):
    train = data.train_mask[i].item()
    val = data.val_mask[i].item()
    test = data.test_mask[i].item()
    role = "学習" if train else ("検証" if val else ("テスト" if test else "未使用"))
    print(f"ノード {i}: {role} (train={train}, val={val}, test={test})")


import torch
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt

# Cora データセットの読み込み
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# NetworkX のグラフオブジェクトに変換
edge_index = data.edge_index
G = nx.Graph()
G.add_edges_from(edge_index.t().tolist())

# サブグラフとして「ノード0とその1-hop近傍」を抽出
center_node = 0
neighbors = list(G.neighbors(center_node))
sub_nodes = [center_node] + neighbors
subgraph = G.subgraph(sub_nodes)

print(f"サブグラフのノード数: {subgraph.number_of_nodes()}")
print(f"サブグラフのエッジ数: {subgraph.number_of_edges()}")

plt.figure(figsize=(10, 8))

# レイアウト（ノードの配置）を計算
pos = nx.spring_layout(subgraph, seed=42)

# ノードの色をラベルに応じて変える（ラベルが無いノードはグレー）
node_color = []
for node in subgraph.nodes():
    if node == center_node:
        node_color.append('red')  # 中心ノードを赤に
    elif node in data.train_mask and data.train_mask[node]:
        node_color.append('blue')   # 学習用ノードを青に
    elif node in data.val_mask and data.val_mask[node]:
        node_color.append('green') # 検証用ノードを緑に
    elif node in data.test_mask and data.test_mask[node]:
        node_color.append('orange') # テスト用ノードをオレンジに
    else:
        node_color.append('gray')   # それ以外はグレー

# グラフ描画
nx.draw(subgraph, pos,
        node_color=node_color,
        node_size=300,
        font_size=8,
        with_labels=True,
        edge_color='lightgray')

plt.title(f"Cora 引用ネットワークのサブグラフ（中心ノード {center_node} とその近傍）")
plt.show()

import torch
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt

# Cora データセットの読み込み
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# NetworkX のグラフオブジェクトに変換
edge_index = data.edge_index
G = nx.Graph()
G.add_edges_from(edge_index.t().tolist())

# ノード 0 から 3-hop 先までのノードを BFS で取得
center_node = 0
bfs_tree = nx.bfs_tree(G, source=center_node, depth_limit=3)
sub_nodes = list(bfs_tree.nodes())
subgraph = G.subgraph(sub_nodes)

print(f"Subgraph nodes: {subgraph.number_of_nodes()}")
print(f"Subgraph edges: {subgraph.number_of_edges()}")

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(subgraph, seed=42)

# ノードの色を「何 hop 目か」で分ける
node_colors = []
for node in subgraph.nodes():
    try:
        # ノード 0 からの最短距離（hop 数）を取得
        dist = nx.shortest_path_length(G, source=center_node, target=node)
    except nx.NetworkXNoPath:
        dist = -1  # 到達不能（ここでは発生しないはず）
    
    if node == center_node:
        color = 'red'      # 中心ノード（0-hop）
    elif dist == 1:
        color = 'blue'     # 1-hop
    elif dist == 2:
        color = 'green'    # 2-hop
    elif dist == 3:
        color = 'orange'   # 3-hop
    else:
        color = 'gray'     # それ以上 or 到達不能
    
    node_colors.append(color)

nx.draw(subgraph, pos,
        node_color=node_colors,
        node_size=300,
        font_size=8,
        with_labels=True,
        edge_color='lightgray')

plt.title(f"Cora Citation Network Subgraph (center node {center_node}, up to 3-hop neighbors)")
plt.show()