from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# モデルを評価モードにして、全ノードの埋め込みを取得
model.eval()
with torch.no_grad():
    # 1層目の出力（隠れ表現）を取得
    h1 = model.conv1(data.x, data.edge_index)
    h1 = F.relu(h1)

    # 最終層の出力（クラススコア）を取得
    out = model(data.x, data.edge_index)

# t-SNE で 2次元に圧縮
tsne = TSNE(n_components=2, random_state=42)
h1_2d = tsne.fit_transform(h1.numpy())
out_2d = tsne.fit_transform(out.numpy())

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1層目の埋め込み
scatter1 = ax1.scatter(h1_2d[:, 0], h1_2d[:, 1], c=data.y, cmap='Set1', s=5)
ax1.set_title('GCN Layer 1 Embeddings (t-SNE)')
ax1.set_xlabel('t-SNE 1')
ax1.set_ylabel('t-SNE 2')
plt.colorbar(scatter1, ax=ax1)

# 最終層の埋め込み
scatter2 = ax2.scatter(out_2d[:, 0], out_2d[:, 1], c=data.y, cmap='Set1', s=5)
ax2.set_title('GCN Final Output Embeddings (t-SNE)')
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
plt.colorbar(scatter2, ax=ax2)

plt.tight_layout()
plt.show()

# モデルを評価モードに
model.eval()

# 全ノードの出力を取得（log_softmax 済み）
with torch.no_grad():
    out = model(data.x, data.edge_index)  # shape: [num_nodes, num_classes]

# ノード 0 の予測結果を確認
node_id = 0

# 各クラスの対数確率 → 確率に変換
log_probs = out[node_id]
probs = torch.exp(log_probs)  # log_softmax の逆変換

# 予測ラベル（最大確率のクラス）
pred_label = out[node_id].argmax().item()
true_label = data.y[node_id].item()

# クラス名の対応（Cora の 7 クラス）
class_names = [
    "Case-Based",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Reinforcement Learning",
    "Rule Learning",
    "Theory"
]

# いくつかのノードについて一覧表示
node_ids = [0, 10, 100, 500, 1000]

print("=== 複数ノードの予測結果一覧 ===")
for node_id in node_ids:
    pred_label = out[node_id].argmax().item()
    true_label = data.y[node_id].item()
    match = "✓" if pred_label == true_label else "✗"
    print(f"ノード {node_id:4d}: 予測 {pred_label} ({class_names[pred_label]:20s}) | "
          f"正解 {true_label} ({class_names[true_label]:20s}) | {match}")
    
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

# 対象ノードだけを含む部分グラフを作成
G_sub = nx.Graph()
for i in target_nodes:
    G_sub.add_node(i, pos=z_2d[i])

# 実際に存在するエッジ（正例）のうち、端点が target_nodes 内のものを抽出
true_edge_indices = np.where(true_labels == 1)[0]
true_edges_in_subset = []
for idx in true_edge_indices:
    u, v = src[idx], dst[idx]
    if u in target_nodes and v in target_nodes:
        true_edges_in_subset.append((u, v))

plt.figure(figsize=(10, 8))

# ノードをプロット
pos_sub = nx.get_node_attributes(G_sub, 'pos')
nx.draw_networkx_nodes(G_sub, pos_sub, node_size=50, node_color='lightgray', alpha=0.8)

# 実際に存在するエッジ（薄いグレー）
nx.draw_networkx_edges(G_sub, pos_sub, edgelist=true_edges_in_subset,
                       edge_color='lightgray', width=1, alpha=0.3,
                       label='Actual edges')

# モデルが高スコアと予測したエッジ
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