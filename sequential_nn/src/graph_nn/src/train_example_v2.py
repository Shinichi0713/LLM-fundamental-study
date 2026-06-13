import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. 内蔵データセット（Digits）の読み込みとグラフ構造の自動構築
# =====================================================================
print("内蔵データセットを読み込んでいます...")
digits = load_digits()

# =====================================================================
# 読み込んだデータ（画像とラベル）を10枚並べて表示
# =====================================================================
plt.figure(figsize=(15, 3))

# 先頭から10枚の画像を表示
for index in range(10):
    # 8x8のグリッド状にサブプロットを配置 (1行10列)
    plt.subplot(1, 10, index + 1)
    
    # digits.images には、すでに 8x8 の行列データが入っています
    # cmap=plt.cm.gray_r で「白黒反転（文字を黒、背景を白）」にして見やすくします
    plt.imshow(digits.images[index], cmap=plt.cm.gray_r, interpolation='nearest')
    
    # 各画像の上部に、対応するラベル（目的変数）を表示
    plt.title(f"Label: {digits.target[index]}", fontsize=12, fontweight="bold")
    
    # グラフの目盛り（軸の数字）は邪魔なので非表示にします
    plt.axis('off')

plt.suptitle("Input Data (Features: 8x8 Pixels) & Target Labels", fontsize=14, y=1.1, fontweight="bold")
plt.show()

# =====================================================================
# データの「中身の形状（次元）」もテキストで確認
# =====================================================================
print("\n=== データ構造の詳細 ===")
print(f"1枚目の画像（行列）の形状: {digits.images[0].shape} (縦8ピクセル × 横8ピクセル)")
print(f"GNNに入力した説明変数（1次元化）の形状: {digits.data[0].shape} (64次元のベクトル)")
print(f"1枚目の画像の実データ（ピクセル値の一例）:\n{digits.data[0].round().astype(int).reshape(8,8)}")

# 説明変数 X: 各ノード（画像）のプロフィール（8x8マスのピクセル値 = 64次元）
features = torch.tensor(digits.data, dtype=torch.float32)

# 目的変数 Y: 各ノードの正解ラベル（0 〜 9 の数字クラス）
labels = torch.tensor(digits.target, dtype=torch.long)

num_nodes = features.shape[0]   # 1797枚の画像ノード
input_dim = features.shape[1]   # 64次元の説明変数
num_classes = 10                # 10クラス分類

print(f"データ読み込み完了! 画像数(ノード): {num_nodes}, 説明変数の次元: {input_dim}, クラス数: {num_classes}")

print("画像同士の類似度からネットワーク（エッジ）を構築中...")
# 各画像間の距離を計算し、「特に見た目が似ている画像トップ5」同士を線（エッジ）で結ぶ
dist_matrix = pairwise_distances(digits.data, metric='euclidean')
adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

top_k = 5
for i in range(num_nodes):
    # 自分自身を除外して、最も近い（似ている）インデックスを取得
    nearest_neighbors = np.argsort(dist_matrix[i])[1:top_k+1]
    for n in nearest_neighbors:
        adj_matrix[i, n] = 1.0
        adj_matrix[n, i] = 1.0 # 無向グラフ

# 訓練用（100個）とテスト用（残りの1000個）のマスクを作成
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[:100] = True
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[100:1100] = True

# =====================================================================
# 2. GCNモデルの定義
# =====================================================================
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # 対称正規化隣接行列の計算
        adj_tilde = adj + torch.eye(adj.size(0))
        degree = torch.sum(adj_tilde, dim=1)
        D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj_tilde), D_inv_sqrt)

        # メッセージパッシング
        return torch.mm(adj_norm, torch.mm(x, self.weight))

class DigitsGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gcn2(x, adj)
        return F.log_softmax(x, dim=1)

# =====================================================================
# 3. 訓練と評価の実行
# =====================================================================
# 64次元(画素値) -> 32次元(隠れ層) -> 10次元(予測数字)
model = DigitsGNN(input_dim=input_dim, hidden_dim=32, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

print("\nGNNの学習を開始します...")
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()

    output = model(features, adj_matrix)
    loss = criterion(output[train_mask], labels[train_mask])

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            preds = output.max(1)[1]
            correct = preds[test_mask].eq(labels[test_mask]).sum().item()
            acc = correct / test_mask.sum().item()
            print(f"Epoch: {epoch:03d} | 訓練データの損失: {loss.item():.4f} | 未知のデータに対する予測精度: {acc*100:.1f}%")

import matplotlib.pyplot as plt
import networkx as nx

# =====================================================================
# 可視化用の設定（最初の100ノードに絞る）
# =====================================================================
num_viz_nodes = 100

# 1. モデルを評価モードにして、最終的な予測結果（クラス）を取得
model.eval()
with torch.no_grad():
    all_output = model(features, adj_matrix)
    all_preds = all_output.max(1)[1].numpy()  # AIの予測した数字 (0〜9)

# 2. 最初の100ノード分の部分隣接行列を切り出し
sub_adj = adj_matrix[:num_viz_nodes, :num_viz_nodes].numpy()
sub_preds = all_preds[:num_viz_nodes]
sub_labels = labels[:num_viz_nodes].numpy()

# 3. NetworkXのグラフオブジェクトを構築
G_viz = nx.from_numpy_array(sub_adj)

# =====================================================================
# グラフの描画
# =====================================================================
plt.figure(figsize=(12, 10))

# ノードの配置ルール（春モデルレイアウト：繋がりの強いノードほど近くに配置される）
pos_viz = nx.spring_layout(G_viz, k=0.3, seed=42)

# ノードを描画（色はAIが予測した「数字クラス」に基づいて10色に色分け）
# coolwarmやtab10などのカラーマップを使用
scatter = nx.draw_networkx_nodes(
    G_viz, pos_viz,
    node_color=sub_preds,
    cmap=plt.cm.tab10,
    node_size=350,
    edgecolors="black",
    linewidths=0.8
)

# ノードの中に「実際の正解の数字」をテキストとして描画
labels_dict = {i: str(sub_labels[i]) for i in range(num_viz_nodes)}
nx.draw_networkx_labels(G_viz, pos_viz, labels=labels_dict, font_size=8, font_color="black", font_weight="bold")

# エッジ（画像同士の類似度の線）を描画（少し薄くして見やすくします）
nx.draw_networkx_edges(G_viz, pos_viz, alpha=0.2, edge_color="gray")

# カラーバー（どの色が何の数字を表すか）を表示
cbar = plt.colorbar(scatter, ticks=range(10), label="AI Predicted Digits (0-9)")
cbar.set_ticklabels([str(i) for i in range(10)])

plt.title("GNN Classification Result Visualized (First 100 Nodes)", fontsize=14, fontweight="bold")
plt.axis("off") # 軸は非表示
plt.show()