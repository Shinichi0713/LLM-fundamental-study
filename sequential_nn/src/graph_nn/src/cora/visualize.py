import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

# モデルを評価モードに
model.eval()

# 埋め込み z を取得（encode ではなく forward を使う）
with torch.no_grad():
    z = model(test_data.x, test_data.edge_index)  # shape: [num_nodes, hidden_dim]
    pred_logits = model.decode(z, test_data.edge_label_index)
    pred = torch.sigmoid(pred_logits).cpu().numpy()

# 正解ラベル（1: 正例, 0: 負例）
true_labels = test_data.edge_label.cpu().numpy()

# テストエッジの端点
src, dst = test_data.edge_label_index.cpu().numpy()

# t-SNE でノード埋め込みを2次元に圧縮
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_2d = tsne.fit_transform(z.cpu().numpy())

# グラフオブジェクトを作成
G = nx.Graph()
for i in range(z_2d.shape[0]):
    G.add_node(i, pos=z_2d[i])

# 予測スコアが高い上位N本のエッジを可視化
top_k = 50
top_indices = np.argsort(pred)[-top_k:]

plt.figure(figsize=(10, 8))

# ノードをプロット
pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx_nodes(G, pos, node_size=20, node_color='lightgray', alpha=0.6)

# エッジを描画
for idx in top_indices:
    u, v = src[idx], dst[idx]
    if true_labels[idx] == 1:
        # True positive (red)
        nx.draw_networkx_edges(G, {u: pos[u], v: pos[v]}, edgelist=[(u, v)],
                               edge_color='red', width=2, alpha=0.7,
                               label='True positive' if idx == top_indices[0] else "")
    else:
        # False positive (blue)
        nx.draw_networkx_edges(G, {u: pos[u], v: pos[v]}, edgelist=[(u, v)],
                               edge_color='blue', width=1, alpha=0.5,
                               label='False positive' if idx == top_indices[0] else "")

plt.title(f'Cora Link Prediction: Top-{top_k} Predicted Edges\n(Red: True Positive, Blue: False Positive)')
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.show()

# ノード数
num_nodes = test_data.num_nodes

# 予測スコア行列と実際の隣接行列
pred_matrix = np.zeros((num_nodes, num_nodes))
true_matrix = np.zeros((num_nodes, num_nodes))

for i in range(len(pred)):
    u, v = src[i], dst[i]
    pred_matrix[u, v] = pred[i]
    true_matrix[u, v] = true_labels[i]

# 一部のノードだけを表示（例: 先頭100ノード）
subset_nodes = 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 予測スコアのヒートマップ
im0 = axes[0].imshow(pred_matrix[:subset_nodes, :subset_nodes],
                     cmap='Reds', aspect='auto', interpolation='nearest')
axes[0].set_title('Predicted Scores (higher = redder)')
axes[0].set_xlabel('Target Node')
axes[0].set_ylabel('Source Node')
plt.colorbar(im0, ax=axes[0])

# 実際のエッジのヒートマップ
im1 = axes[1].imshow(true_matrix[:subset_nodes, :subset_nodes],
                     cmap='Blues', aspect='auto', interpolation='nearest')
axes[1].set_title('Actual Edges (1: exists, 0: absent)')
axes[1].set_xlabel('Target Node')
axes[1].set_ylabel('Source Node')
plt.colorbar(im1, ax=axes[1])

plt.suptitle('Cora Link Prediction: Predicted vs Actual')
plt.tight_layout()
plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

# モデルを評価モードに
model.eval()

# 埋め込み z を取得
with torch.no_grad():
    z = model(test_data.x, test_data.edge_index)  # shape: [num_nodes, hidden_dim]
    pred_logits = model.decode(z, test_data.edge_label_index)
    pred = torch.sigmoid(pred_logits).cpu().numpy()

# 正解ラベル（1: 正例, 0: 負例）
true_labels = test_data.edge_label.cpu().numpy()

# テストエッジの端点
src, dst = test_data.edge_label_index.cpu().numpy()

# t-SNE でノード埋め込みを2次元に圧縮
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_2d = tsne.fit_transform(z.cpu().numpy())

# 対象ノード（例: 先頭20ノード）
target_nodes = list(range(20))  # [0, 1, 2, ..., 19]
num_target_nodes = len(target_nodes)

# 予測スコアが 0.6 以上のエッジを抽出
threshold = 0.6
high_prob_indices = np.where(pred >= threshold)[0]

# その中で、端点が両方とも target_nodes に含まれるエッジだけを選ぶ
selected_indices = []
for idx in high_prob_indices:
    u, v = src[idx], dst[idx]
    if u in target_nodes and v in target_nodes:
        selected_indices.append(idx)

print(f"Number of edges with predicted score ≥ {threshold} and endpoints in target_nodes: {len(selected_indices)}")

# 実際に存在するエッジ（正例）のうち、端点が target_nodes 内のものを抽出
true_edge_indices = np.where(true_labels == 1)[0]
true_edges_in_subset = []
for idx in true_edge_indices:
    u, v = src[idx], dst[idx]
    if u in target_nodes and v in target_nodes:
        true_edges_in_subset.append((u, v))

print(f"Number of actual edges within target_nodes: {len(true_edges_in_subset)}")

# 対象ノードだけを含む部分グラフを作成
G_sub = nx.Graph()
for i in target_nodes:
    G_sub.add_node(i, pos=z_2d[i])

plt.figure(figsize=(10, 8))

# ノードをプロット
pos_sub = nx.get_node_attributes(G_sub, 'pos')
nx.draw_networkx_nodes(G_sub, pos_sub, node_size=50, node_color='lightgray', alpha=0.8)

# 1. 実際に接続しているエッジ（薄いグレー）
nx.draw_networkx_edges(G_sub, pos_sub, edgelist=true_edges_in_subset,
                       edge_color='red', width=1, alpha=0.3,
                       label='Actual edges')

# 2. モデルが高スコアと予測したエッジ
for idx in selected_indices:
    u, v = src[idx], dst[idx]
    if true_labels[idx] == 1:
        # True positive (red)
        nx.draw_networkx_edges(G_sub, {u: pos_sub[u], v: pos_sub[v]}, edgelist=[(u, v)],
                               edge_color='red', width=2, alpha=0.7,
                               label='True positive' if idx == selected_indices[0] else "")
    else:
        # False positive (blue)
        nx.draw_networkx_edges(G_sub, {u: pos_sub[u], v: pos_sub[v]}, edgelist=[(u, v)],
                               edge_color='blue', width=1, alpha=0.5,
                               label='False positive' if idx == selected_indices[0] else "")

plt.title(f'Cora Link Prediction: Predicted vs Actual (Score ≥ {threshold})\n(Gray: Actual edges, Red: True positive, Blue: False positive)')
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.show()