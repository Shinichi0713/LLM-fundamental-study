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

