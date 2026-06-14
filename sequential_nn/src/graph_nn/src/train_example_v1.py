import time
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output

# =====================================================================
# 1. データの準備（公開データセット：空手クラブ）
# =====================================================================
# 34人のメンバー（ノード）と、その友情関係（エッジ）のグラフ
G = nx.karate_club_graph()
num_nodes = G.number_of_nodes()

# 隣接行列 A の作成 [34, 34]
adj = nx.to_numpy_array(G)
adj = torch.tensor(adj, dtype=torch.float32)

# 特徴量行列 X の作成 [34, 34]
# 今回は個人のプロファイルがないため、単位行列（ID情報）を初期特徴量とします
features = torch.eye(num_nodes)

# 正解ラベルの作成 (Mr. Hiの派閥=0, John A.の派閥=1)
labels = torch.tensor([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in range(num_nodes)])

# ★GNNの凄さを体感するポイント★
# 34人中、わずか「2人（リーダー2人）」の正解だけを訓練データとし、残り32人を予測（テスト）します
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[0] = True   # Mr. Hi (コミュニティ1のリーダー)
train_mask[33] = True  # John A. (コミュニティ2のリーダー)

test_mask = ~train_mask

# =====================================================================
# 2. ミニマルなGNN（GCN）モデルの定義
# =====================================================================
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # 自己ループと対称正規化のシミュレート
        adj_tilde = adj + torch.eye(adj.size(0))
        degree = torch.sum(adj_tilde, dim=1)
        D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj_tilde), D_inv_sqrt)
        
        # メッセージパッシング
        return torch.mm(adj_norm, torch.mm(x, self.weight))

class KarateGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNLayer(input_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, x, adj):
        # 1層目を通した後の、ノードの2次元空間プロット用特徴量を保持
        self.embedding = self.conv1(x, adj)
        x = F.relu(self.embedding)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)

# =====================================================================
# 3. 訓練ループとリアルタイム可視化
# =====================================================================
model = KarateGNN(input_dim=num_nodes, hidden_dim=4, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=0.05)
criterion = nn.NLLLoss()

pos = nx.spring_layout(G, seed=42) # グラフの描画レイアウトを固定

print("学習を開始します...")
for epoch in range(1, 31):
    model.train()
    optimizer.zero_grad()
    
    output = model(features, adj)
    
    # わずか2人のリーダーの損失（エラー）だけで学習！
    loss = criterion(output[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    
    # 予測結果の取得
    preds = output.max(1)[1].numpy()
    
    # --- Colab上での動的グラフ描画 ---
    clear_output(wait=True)
    plt.figure(figsize=(8, 6))
    
    # 予測されたクラス（0=紫、1=黄色など）でノードを色分け
    nx.draw(
        G, pos, with_labels=True, 
        node_color=preds, cmap=plt.cm.coolwarm, 
        node_size=500, font_color="white", font_weight="bold"
    )
    
    # リーダー2人を強調表示（四角形にする）
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_shape="s", node_size=700, node_color=[preds[0]], cmap=plt.cm.coolwarm, edgecolors="black", linewidths=2)
    nx.draw_networkx_nodes(G, pos, nodelist=[33], node_shape="s", node_size=700, node_color=[preds[33]], cmap=plt.cm.coolwarm, edgecolors="black", linewidths=2)
    
    plt.title(f"GNN Training - Epoch: {epoch:02d} | Loss: {loss.item():.4f}", fontsize=14)
    plt.show()
    time.sleep(0.2) # アニメーション速度の調整

# 最終結果の評価
model.eval()
final_output = model(features, adj)
final_preds = final_output.max(1)[1]
correct = final_preds[test_mask].eq(labels[test_mask]).sum().item()
accuracy = correct / test_mask.sum().item()

print(f"【実験結果】")
print(f"正解を知らされていなかった32人に対する予測正解率: {accuracy * 100:.1f}%")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import networkx as nx

# =====================================================================
# 1. データの読み込みと「ε-近傍グラフ」の構築
# =====================================================================
digits = load_digits()
features = torch.tensor(digits.data, dtype=torch.float32)
labels = torch.tensor(digits.target, dtype=torch.long)

num_nodes = features.shape[0]
input_dim = features.shape[1]
num_classes = 10

# 全画像間の距離を計算
dist_matrix = pairwise_distances(digits.data, metric='euclidean')

# ★ここが新しいバリエーションの核★
# 距離の閾値（epsilon）を設定。これより「近い（似ている）」画像同士だけを結ぶ
# 距離の最大値と最小値のバランスを見て、程よい繋がりになる閾値を設定します
epsilon = 18.5  
adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if dist_matrix[i, j] < epsilon:
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0

# 訓練マスク・テストマスク（前回と同条件）
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
        adj_tilde = adj + torch.eye(adj.size(0))
        degree = torch.sum(adj_tilde, dim=1)
        D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj_tilde), D_inv_sqrt)
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
# 3. 訓練
# =====================================================================
model = DigitsGNN(input_dim=input_dim, hidden_dim=32, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

print("新しいグラフ構造（ε-近傍）でのGNN学習開始...")
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_matrix)
    loss = criterion(output[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    final_output = model(features, adj_matrix)
    all_preds = final_output.max(1)[1].numpy()

# =====================================================================
# 4. 新しいグラフ構造の可視化（最初の100ノード）
# =====================================================================
num_viz_nodes = 100
sub_adj = adj_matrix[:num_viz_nodes, :num_viz_nodes].numpy()
sub_preds = all_preds[:num_viz_nodes]
sub_labels = labels[:num_viz_nodes].numpy()

G_viz = nx.from_numpy_array(sub_adj)

plt.figure(figsize=(12, 10))
pos_viz = nx.spring_layout(G_viz, k=0.3, seed=42)

# ノード描画（AIの予測で色分け）
scatter = nx.draw_networkx_nodes(
    G_viz, pos_viz, node_color=sub_preds, cmap=plt.cm.tab10, 
    node_size=350, edgecolors="black", linewidths=0.8
)

# 正解の数字を描画
labels_dict = {i: str(sub_labels[i]) for i in range(num_viz_nodes)}
nx.draw_networkx_labels(G_viz, pos_viz, labels=labels_dict, font_size=8, font_color="black", font_weight="bold")

# エッジを描画
nx.draw_networkx_edges(G_viz, pos_viz, alpha=0.2, edge_color="gray")

cbar = plt.colorbar(scatter, ticks=range(10), label="AI Predicted Digits (0-9)")
cbar.set_ticklabels([str(i) for i in range(10)])

plt.title("Epsilon-Neighborhood Graph Variety Visualization", fontsize=14, fontweight="bold")
plt.axis("off")
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_auc_score

# =====================================================================
# 1. 疑似データの構築と「未来のリンク（検証用）」の隠蔽
# =====================================================================
np.random.seed(42)
torch.manual_seed(42)

num_products = 60  # 分かりやすさのためノード数を60に調整
labels_np = np.array([0]*20 + [1]*20 + [2]*20) # 3カテゴリ

# 説明変数 X: 2次元のダミープロフィール
features = torch.tensor(np.random.randn(num_products, 2) * 2 + 3, dtype=torch.float32)

# 本来のすべてのリンク（グラウンドトゥルース）を定義
edges = []
for i in range(num_products):
    for j in range(i + 1, num_products):
        if labels_np[i] == labels_np[j] and np.random.rand() < 0.3:
            edges.append((i, j))
        elif labels_np[i] != labels_np[j] and np.random.rand() < 0.01:
            edges.append((i, j))

all_edges = np.array(edges)
num_edges = len(all_edges)

# 20%のリンクを「未来（テスト用）」として隠し、80%を「現在（学習用）」とする
shuffled_indices = np.random.permutation(num_edges)
train_edge_idx = shuffled_indices[:int(num_edges * 0.8)]
test_edge_idx = shuffled_indices[int(num_edges * 0.8):]

train_edges = all_edges[train_edge_idx]
test_edges = all_edges[test_edge_idx]

# 学習用の隣接行列 A を構築（隠されたリンクは0になっている）
adj_train = torch.zeros((num_products, num_products), dtype=torch.float32)
for u, v in train_edges:
    adj_train[u, v] = 1.0
    adj_train[v, u] = 1.0

# 同数の「絶対に繋がらないペア（負例）」をテスト用にサンプリング
negative_edges = []
while len(negative_edges) < len(test_edges):
    u, v = np.random.randint(0, num_products, size=2)
    if u != v and (u, v) not in edges and (v, u) not in edges and (u, v) not in negative_edges:
        negative_edges.append((u, v))
negative_edges = np.array(negative_edges)

# =====================================================================
# 2. リンク予測用 GNN（GCN）モデルの定義
# =====================================================================
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        adj_tilde = adj + torch.eye(adj.size(0))
        degree = torch.sum(adj_tilde, dim=1)
        D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj_tilde), D_inv_sqrt)
        return torch.mm(adj_norm, torch.mm(x, self.weight))

class LinkPredictionGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim) # 最終ノードベクトルを出力

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        return self.gcn2(x, adj)

    def compute_link_score(self, z, edge_list):
        # 2つのノードベクトルの内積（類似度）を計算し、リンクの存在確率（0〜1）にする
        u_embed = z[edge_list[:, 0]]
        v_embed = z[edge_list[:, 1]]
        scores = torch.sum(u_embed * v_embed, dim=1)
        return torch.sigmoid(scores)

# =====================================================================
# 3. 訓練ループ（リンクの有無を当てるバイナリ分類として学習）
# =====================================================================
model = LinkPredictionGNN(input_dim=2, hidden_dim=16)
optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)

print("隠された未来の購入リンクを予測する学習を開始...")
for epoch in range(1, 151):
    model.train()
    optimizer.zero_grad()
    
    # 1. GNNで全ノードの構造ベクトル（z）を生成
    z = model(features, adj_train)
    
    # 2. 存在するリンク（正例）と存在しないリンク（負例）のスコアを計算
    pos_scores = model.compute_link_score(z, train_edges)
    
    # 訓練用の負例をランダムサンプリング
    train_neg_edges = np.array([np.random.randint(0, num_products, size=2) for _ in range(len(train_edges))])
    neg_scores = model.compute_link_score(z, train_neg_edges)
    
    # 損失関数（存在するものは1に、存在しないものは0に近づける）
    loss = -torch.mean(torch.log(pos_scores + 1e-15) + torch.log(1.0 - neg_scores + 1e-15))
    
    loss.backward()
    optimizer.step()
    
    if epoch % 30 == 0:
        # 検証：隠されていたテスト用リンクの予測精度（AUCスコア）を測定
        model.eval()
        with torch.no_grad():
            z_eval = model(features, adj_train)
            test_pos_preds = model.compute_link_score(z_eval, test_edges).numpy()
            test_neg_preds = model.compute_link_score(z_eval, negative_edges).numpy()
            
            y_true = np.ones(len(test_pos_preds))
            y_true = np.concatenate([y_true, np.zeros(len(test_neg_preds))])
            y_pred = np.concatenate([test_pos_preds, test_neg_preds])
            
            auc = roc_auc_score(y_true, y_pred)
            print(f"Epoch: {epoch:03d} | Train Loss: {loss.item():.4f} | 未来のリンク予測精度 (AUC): {auc*100:.1f}%")

# =====================================================================
# 4. 「未来のリンク」の予測結果をビジュアル確認
# =====================================================================
model.eval()
with torch.no_grad():
    z_final = model(features, adj_train)
    # 隠されていた未来のリンクの予測確率をトップ5だけ表示してみる
    test_scores = model.compute_link_score(z_final, test_edges).numpy()

print("\n=== AIが見つけた【未来のあわせて買いたいペア】TOP 5 ===")
top_5_idx = np.argsort(test_scores)[::-1][:5]
for idx in top_5_idx:
    u, v = test_edges[idx]
    print(f"商品 {u:02d} と 商品 {v:02d} -> AIの予測共同購入確率: {test_scores[idx]*100:.1f}%  (正解: 実際に未来で購入された)")