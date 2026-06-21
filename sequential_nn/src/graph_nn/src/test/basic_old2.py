
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MovieLens1M
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt

# 乱数固定
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 1. データセットのロードと属性名の確認
# =============================================================================
dataset = MovieLens1M(root="./data/MovieLens")
data = dataset[0]  # HeteroDataオブジェクト

print("データの情報:")
print(f"ユーザー数: {data['user'].num_nodes}")
print(f"映画数: {data['movie'].num_nodes}")
print(f"評価エッジ数: {data['user', 'rates', 'movie'].num_edges}")
print(f"ユーザー特徴次元: {data['user'].x.shape if data['user'].x is not None else 'なし'}")
print(f"映画特徴次元: {data['movie'].x.shape if data['movie'].x is not None else 'なし'}")

# エッジストレージの属性名を確認
edge_storage = data['user', 'rates', 'movie']
print("\nエッジストレージの属性名:")
print(edge_storage.keys())

# 評価ラベルは 'rating' 属性に格納されている
edge_index = edge_storage.edge_index  # (2, num_edges)
ratings = edge_storage.rating         # 1〜5の評価値

print(f"評価ラベルの形状: {ratings.shape}")
print(f"評価ラベルの例: {ratings[:5]}")

# =============================================================================
# 2. 前処理：2値化とデータ分割
# =============================================================================
# 評価が4以上を「高評価（リンク強い）」、3以下を「低評価（リンク弱い）」として2値化
y_binary = (ratings >= 4).float()

# エッジを学習用とテスト用に分割（8:2）
num_edges = edge_index.size(1)
indices = torch.randperm(num_edges)
train_idx = indices[:int(0.8 * num_edges)]
test_idx = indices[int(0.8 * num_edges):]

train_edge_index = edge_index[:, train_idx]
train_y = y_binary[train_idx]
test_edge_index = edge_index[:, test_idx]
test_y = y_binary[test_idx]

print(f"学習用エッジ数: {train_edge_index.size(1)}")
print(f"テスト用エッジ数: {test_edge_index.size(1)}")

# =============================================================================
# 3. シンプルなGNNモデル（LightGCN風）の定義
# =============================================================================
class SimpleLightGCN(nn.Module):
    def __init__(self, user_dim, movie_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # ユーザーと映画の埋め込み（IDベースの協調フィルタリング）
        self.user_embed = nn.Embedding(user_dim, hidden_dim)
        self.movie_embed = nn.Embedding(movie_dim, hidden_dim)
        
        # 重み初期化
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.movie_embed.weight)

    def forward(self, edge_index):
        """
        edge_index: (2, num_edges)  [user_idx, movie_idx]
        """
        user_idx, movie_idx = edge_index
        
        # 埋め込み取得
        user_emb = self.user_embed(user_idx)  # (num_edges, hidden_dim)
        movie_emb = self.movie_embed(movie_idx)  # (num_edges, hidden_dim)
        
        # 内積でスコア計算（協調フィルタリング）
        scores = torch.sum(user_emb * movie_emb, dim=1)  # (num_edges,)
        
        # シグモイドで確率に変換（0〜1）
        return torch.sigmoid(scores)

# =============================================================================
# 4. モデル・オプティマイザの準備
# =============================================================================
model = SimpleLightGCN(
    user_dim=data['user'].num_nodes,
    movie_dim=data['movie'].num_nodes,
    hidden_dim=64,
    num_layers=3
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# =============================================================================
# 5. 学習ループ（リンク予測＝2値分類）
# =============================================================================
def train():
    model.train()
    optimizer.zero_grad()
    
    # 学習用エッジのスコアを計算
    pred = model(train_edge_index)
    
    # 損失関数：バイナリ交差エントロピー
    loss = F.binary_cross_entropy(pred, train_y)
    
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(edge_index, y_true):
    model.eval()
    pred = model(edge_index).cpu().numpy()
    y_true = y_true.cpu().numpy()
    
    auc = roc_auc_score(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    return auc, rmse

print("学習を開始します...")
train_losses = []
test_aucs = []
test_rmses = []

for epoch in range(1, 101):
    loss = train()
    train_losses.append(loss)
    
    if epoch % 20 == 0:
        auc, rmse = test(test_edge_index, test_y)
        test_aucs.append(auc)
        test_rmses.append(rmse)
        print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Test AUC: {auc:.4f} | Test RMSE: {rmse:.4f}")

# =============================================================================
# 6. 学習曲線の可視化
# =============================================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")

plt.subplot(1, 2, 2)
plt.plot(test_aucs, label="AUC")
plt.title("Test AUC over epochs")
plt.xlabel("Epoch (every 20 steps)")
plt.ylabel("AUC")
plt.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# 7. 推論例：特定ユーザーに対する映画推薦ランキング
# =============================================================================
@torch.no_grad()
def recommend_for_user(user_id, top_k=10):
    model.eval()
    
    # 全映画ID
    movie_ids = torch.arange(data['movie'].num_nodes)
    
    # ユーザーIDを全映画数分複製
    user_ids = torch.full((movie_ids.size(0),), user_id)
    
    # エッジとして組み立て
    edge_index = torch.stack([user_ids, movie_ids], dim=0)
    
    # 各映画の「高評価確率」を予測
    scores = model(edge_index).cpu().numpy()
    
    # スコア上位の映画IDを取得
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_movies = movie_ids[top_indices].numpy()
    top_scores = scores[top_indices]
    
    return top_movies, top_scores

# 例：ユーザーID=0 に対する推薦TOP5
user_id = 0
top_movies, top_scores = recommend_for_user(user_id, top_k=5)

print(f"\nユーザー {user_id} へのおすすめ映画 TOP5:")
for i, (mid, score) in enumerate(zip(top_movies, top_scores)):
    print(f"{i+1}. 映画ID {mid} -> 高評価確率: {score*100:.1f}%")

# =============================================================================
# 8. まとめ
# =============================================================================
print("\n=== 実験のポイント ===")
print("・MovieLens 1Mを二部グラフ（ユーザー–映画）として扱い、")
print("  評価4以上を「リンクあり」、3以下を「リンクなし」として2値分類。")
print("・シンプルなLightGCN風モデルで、ユーザーと映画の埋め込みの内積から")
print("  「高評価をつける確率」を予測。")
print("・テストセットでAUCとRMSEを評価し、特定ユーザーへの推薦ランキングを出力。")