import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# モジュール1: 特徴量変換・集約層 (Graph Convolution Module)
# =====================================================================
class GCNConv(nn.Module):
    """
    1つ1つのノードが「隣人の情報を集めて自分の特徴量を更新する」ためのモジュール。
    GNNの最もコアとなる計算（メッセージパッシング）を担当します。
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        # 学習可能なパラメータ（重み W と バイアス b）
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # 初期化
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x  : ノードの特徴量 [ノード数, 入力次元]
        adj: 隣接行列       [ノード数, ノード数]
        """
        num_nodes = x.size(0)
        
        # 1. 自己ループの追加 (自分自身の情報も混ぜる)
        adj_tilde = adj + torch.eye(num_nodes, device=adj.device)
        
        # 2. 次数行列による対称正規化 (値の爆発を防ぐ)
        degree = torch.sum(adj_tilde, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj_tilde), D_inv_sqrt)
        
        # 3. 特徴量変換 (X * W) と 近傍情報の集約
        support = torch.mm(x, self.weight)
        output = torch.mm(adj_normalized, support)
        
        return output + self.bias


# =====================================================================
# モジュール2: グラフ全体集約層 (Global Pooling Module)
# =====================================================================
class GlobalPooling(nn.Module):
    """
    ノードごと（点単位）のベクトルを、グラフ1個全体の「要約ベクトル」に変換するモジュール。
    分子全体の性質を予測したり、グラフ全体のコミュニティを分類するタスクで必須となります。
    """
    def __init__(self, pool_type="mean"):
        super().__init__()
        self.pool_type = pool_type

    def forward(self, x):
        """
        x: ノードの特徴量 [ノード数, 特徴量次元]
        """
        if self.pool_type == "mean":
            # 全ノードの平均ベクトルをとる
            return torch.mean(x, dim=0, keepdim=True)
        elif self.pool_type == "sum":
            # 全ノードの総和をとる
            return torch.sum(x, dim=0, keepdim=True)
        elif self.pool_type == "max":
            # 全ノードの最大値をとる
            return torch.max(x, dim=0, keepdim=True)[0]
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")


# =====================================================================
# モジュール3: 最終予測予測層 (Classifier / MLP Module)
# =====================================================================
class GNNClassifier(nn.Module):
    """
    集約されたベクトルを受け取り、全結合層（多層パーセプトロン：MLP）を通して
    最終的なクラスの予測確率を出力するモジュール。
    """
    def __init__(self, in_features, num_classes, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.fc2 = nn.Linear(in_features // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: グラフ全体の要約ベクトル [1, 特徴量次元]
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # 確率分布として扱いやすいよう、Log Softmaxをかける
        return F.log_softmax(x, dim=-1)


# =====================================================================
# 3つの必要モジュールを統合した「完成版GNNシステム」
# =====================================================================
class CompleteGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # 上記で作成した3つの必要モジュールをそれぞれインスタンス化して配置
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pooling = GlobalPooling(pool_type="mean")
        self.classifier = GNNClassifier(hidden_dim, num_classes)

    def forward(self, x, adj):
        # ステップ1: 特徴量変換と隣人集約（2層重ねてより広い範囲の文脈を捉える）
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        
        # ステップ2: グラフ全体の特徴を1つのベクトルにプール（要約）する
        graph_vector = self.pooling(x)
        
        # ステップ3: 分類器に通して予測を出力
        output = self.classifier(graph_vector)
        return output

# =====================================================================
# 動作検証
# =====================================================================
if __name__ == "__main__":
    # ダミーのグラフ1個（5ノード、各ノードに10次元の特徴量、3クラス分類タスク）
    dummy_x = torch.randn(5, 10)
    dummy_adj = torch.tensor([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ], dtype=torch.float32)

    # 統合したGNNモデルを動かす
    gnn_model = CompleteGNN(input_dim=10, hidden_dim=16, num_classes=3)
    gnn_model.eval()
    
    final_output = gnn_model(dummy_x, dummy_adj)
    
    print("=== 各モジュールの役割と流れ ===")
    print(f"1. GCNConv   : 各ノードの特徴量を 10次元 -> 16次元 へ集約しながら変換")
    print(f"2. Pooling   : 5個のノードベクトル [5, 16] を 1個のグラフベクトル [1, 16] へ要約")
    print(f"3. Classifier: グラフベクトルを [1, 3] の予測スコアへ変換\n")
    print("最終予測出力 (形状):", final_output.shape)
    print("予測結果 (Log Softmax):", final_output)