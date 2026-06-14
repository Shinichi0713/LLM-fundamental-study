import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# =====================================================================
# 1. 疑似ECサイトデータの作成（プロフィール ＋ 共同購入ネットワーク）
# =====================================================================
np.random.seed(42)
torch.manual_seed(42)

num_products = 90  # 全商品数
# カテゴリ: 0=電子機器(Gadget), 1=書籍(Book), 2=ファッション(Cloth)
labels_np = np.array([0]*30 + [1]*30 + [2]*30) 
labels = torch.tensor(labels_np, dtype=torch.long)

# 説明変数 X: [価格(正規化), レビューの星(1~5)] の2次元プロフィール
# ※わざとカテゴリ間で価格とレビューに大きな差が出ないように設定（プロフィールだけでは分類困難にする）
features_np = np.random.randn(num_products, 2) * 2 + 3
features = torch.tensor(features_np, dtype=torch.float32)

# 隣接行列 A: 共同購入ネットワークの構築
adj_matrix = torch.zeros((num_products, num_products), dtype=torch.float32)

# 同じカテゴリ同士は「一緒に買われやすい」リンクをランダムに作成
for i in range(num_products):
    for j in range(i + 1, num_products):
        if labels_np[i] == labels_np[j]:
            if np.random.rand() < 0.25: # 同カテゴリの共同購入確率
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0
        else:
            if np.random.rand() < 0.01: # 異カテゴリ間の偶発的な共同購入確率
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0

# 訓練用（各カテゴリ3個ずつ、計9個だけ正解を教える）
train_mask = torch.zeros(num_products, dtype=torch.bool)
train_mask[[0, 1, 2, 30, 31, 32, 60, 61, 62]] = True
test_mask = ~train_mask

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

class ECommerceGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, adj)
        return F.log_softmax(x, dim=1)

# =====================================================================
# 3. 訓練の実行
# =====================================================================
model = ECommerceGNN(input_dim=2, hidden_dim=16, num_classes=3)
optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
criterion = nn.NLLLoss()

print("ECサイトのレコメンドネットワークでのGNN学習開始...")
for epoch in range(1, 151):
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
    correct = final_output.max(1)[1][test_mask].eq(labels[test_mask]).sum().item()
    print(f"最終予測精度 (未知の商品81個に対して): {correct / test_mask.sum().item() * 100:.1f}%")

# =====================================================================
# 4. 共同購入グラフの可視化
# =====================================================================
G_viz = nx.from_numpy_array(adj_matrix.numpy())
plt.figure(figsize=(10, 8))
pos_viz = nx.spring_layout(G_viz, k=0.25, seed=42)

# ノードを描画（色は予測カテゴリ: 赤=ガジェット、青=本、緑=服 のように3色に分かれます）
scatter = nx.draw_networkx_nodes(
    G_viz, pos_viz, node_color=all_preds, cmap=plt.cm.Set1, 
    node_size=200, edgecolors="black", linewidths=0.6
)

# 訓練に使ったわずか9個の「ヒント商品」を大きな四角で囲って強調
train_indices = np.where(train_mask.numpy() == True)[0]
nx.draw_networkx_nodes(G_viz, pos_viz, nodelist=list(train_indices), node_shape="s", node_size=400, node_color=all_preds[train_indices], cmap=plt.cm.Set1, edgecolors="black", linewidths=2)

# 共同購入の結びつき（エッジ）を描画
nx.draw_networkx_edges(G_viz, pos_viz, alpha=0.15, edge_color="gray")

plt.title("E-Commerce Product Network (Square = Trained Hint Products)", fontsize=12, fontweight="bold")
plt.axis("off")
plt.show()


import matplotlib.pyplot as plt
import networkx as nx

# =====================================================================
# 1. GNNが予測した「未来のリンク（確率90%以上）」を抽出
# =====================================================================
model.eval()
with torch.no_grad():
    # 全ノードの最終ベクトル（埋め込み）を取得
    z_final = model(features, adj_train)
    
    # すべての商品ペアの組み合わせに対して相性スコア（確率）を計算
    predicted_edges = []
    for i in range(num_products):
        for j in range(i + 1, num_products):
            # すでに「現在わかっているリンク（学習用）」は除外
            if adj_train[i, j] == 0:
                # 商品iと商品jの相性をGNNベクトルから計算
                score = model.compute_link_score(z_final, np.array([[i, j]]))[0].item()
                # AIが「90%以上の確率で将来一緒に買われる！」と予言したペアを抽出
                if score > 0.90:
                    predicted_edges.append((i, j))

print(f"AIが強く予言した『未来の購入リンク』の数: {len(predicted_edges)}本")

# =====================================================================
# 2. グラフオブジェクトの構築
# =====================================================================
# 学習に使った既存のグラフ（80%のリンク）をベースにする
G_future = nx.from_numpy_array(adj_train.numpy())

# AIが予言した「未来のリンク」を別の種類のエッジとしてグラフに追加
G_future.add_edges_from(predicted_edges)

# =====================================================================
# 3. グラフの描画
# =====================================================================
plt.figure(figsize=(12, 10))

# 繋がりの強さで配置を決める（同じジャンルが固まるレイアウト）
pos_future = nx.spring_layout(G_future, k=0.25, seed=42)

# ① ノードの描画（実際の商品カテゴリで色分け。0=赤, 1=青, 2=緑）
scatter = nx.draw_networkx_nodes(
    G_future, pos_future, 
    node_color=labels_np, 
    cmap=plt.cm.Set1, 
    node_size=250, 
    edgecolors="black", 
    linewidths=0.8
)

# ② 既存のリンク（学習で見せていた80%の線）を細いグレーで描画
existing_edges = [(u, v) for u, v in G_future.edges() if (u, v) not in predicted_edges]
nx.draw_networkx_edges(G_future, pos_future, edgelist=existing_edges, alpha=0.15, edge_color="gray")

# ③ ★ここが主役★ AIが予言した「未来のリンク（隠されていた20%）」を太い赤線で描画
nx.draw_networkx_edges(G_future, pos_future, edgelist=predicted_edges, alpha=0.8, edge_color="red", width=2.0)

# ④ 【修正ポイント】ノード数60に合わせて、各カテゴリの先頭3ノード（計9個）を四角で強調
# カテゴリ0: 0,1,2 | カテゴリ1: 20,21,22 | カテゴリ2: 40,41,42
hint_nodes = [0, 1, 2, 20, 21, 22, 40, 41, 42]
nx.draw_networkx_nodes(
    G_future, pos_future, 
    nodelist=hint_nodes, 
    node_shape="s", 
    node_size=350, 
    node_color=labels_np[hint_nodes], 
    cmap=plt.cm.Set1, 
    edgecolors="black", 
    linewidths=1.5
)

# 凡例の設定
plt.title("GNN Link Prediction Visualization\n(Gray lines = Existing Purchase, Red lines = AI Predicted Future Purchase)", fontsize=14, fontweight="bold")
plt.axis("off")
plt.show()