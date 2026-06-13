import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. サブモジュール：GCNLayer（グラフ畳み込み層）の一から実装
# =====================================================================
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 学習パラメータ（重み行列 W と バイアス b）の定義
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # パラメータの初期化（Xavier初期化）
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        x: ノードの特徴量行列 [NodeNum, InFeatures]
        adj: 隣接行列（重みなし、または重みあり） [NodeNum, NodeNum]
        """
        # 1. 自己ループの追加 (A~ = A + I)
        # 自分のノードの特徴量も集約に含めるため、単位行列を足す
        idx = torch.eye(adj.size(0), device=adj.device)
        adj_tilde = adj + idx
        
        # 2. 次数行列 D~ の計算と対称正規化（D~^-1/2 * A~ * D~^-1/2）
        # 各ノードの「つながりの数（次数）」を計算
        degree = torch.sum(adj_tilde, dim=1)
        # 0除算を防ぎつつ、-1/2乗を計算
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        # 対角行列に変換
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        # 正規化隣接行列の計算
        # これにより、繋がりの多いノード（ハブ）からの影響が強くなりすぎるのを防ぐ
        adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj_tilde), D_inv_sqrt)
        
        # 3. メッセージパッシング（特徴量変換と周辺情報の集約）
        # 線形変換: X * W
        support = torch.mm(x, self.weight)
        # 近傍ノード情報の集約: A_norm * (X * W)
        output = torch.mm(adj_normalized, support)
        
        return output + self.bias

# =====================================================================
# 2. メインネットワーク：2層のGCNモデル
# =====================================================================
class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout=0.5):
        super().__init__()
        # 自作したサブモジュール（層）を重ねる
        self.gc1 = GraphConvolutionLayer(n_feat, n_hid)
        self.gc2 = GraphConvolutionLayer(n_hid, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        # 第1層 + 活性化関数(ReLU) + ドロップアウト
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 第2層（分類スコアの出力用）
        x = self.gc2(x, adj)
        
        # クラス分類タスクを想定してLog Softmaxを出力
        return F.log_softmax(x, dim=1)

# =====================================================================
# 3. ダミーデータによる動作検証
# =====================================================================
if __name__ == "__main__":
    # パラメータ設定
    num_nodes = 5      # ノード数
    input_dim = 16     # 各ノードの初期特徴量次元（例: 単語ベクトルなど）
    hidden_dim = 8     # 隠れ層の次元
    num_classes = 3    # 分類クラス数（例: コミュニティの種類など）

    # 1. ランダムなノード特徴量 X の作成 [5, 16]
    dummy_features = torch.randn(num_nodes, input_dim)
    
    # 2. 隣接行列 A の作成（5ノードの適当なつながり） [5, 5]
    # 例: ノード0と1、0と2、1と3、2と4、3と4が繋がっている無向グラフ
    dummy_adj = torch.tensor([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=torch.float32)

    # 3. GCNモデルの初期化
    model = GCN(n_feat=input_dim, n_hid=hidden_dim, n_class=num_classes, dropout=0.2)
    model.eval() # 推論モード
    
    # 4. フォワードプロパゲーションの実行
    output = model(dummy_features, dummy_adj)
    
    print("=== GNN(GCN) スクラッチ実装検証 ===")
    print("入力特徴量の形状:", dummy_features.shape)
    print("隣接行列の形状  :", dummy_adj.shape)
    print("GNN出力の形状   :", output.shape) # [ノード数, クラス数] の確率分布が出力される
    print("\n各ノードのクラス予測スコア（Log Softmax）:")
    print(output)

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. サブモジュール：GATLayer（グラフ・アテンション層）の一から実装
# =====================================================================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha      # LeakyReLUの傾き
        self.concat = concat    # 最終層以外はTrue（結合）、最終層はFalse（平均）にするためのフラグ
        
        # 1. 特徴量変換用の重み W
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xor_normal_ if hasattr(nn.init, 'xor_normal_') else nn.init.xavier_normal_(self.W.data, gain=1.414)
        
        # 2. 自己注意（Self-Attention）用のパラメータベクトル a
        # 2つのノードの特徴量を結合(concat)するため、次元数は 2 * out_features
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        
        # 3. 活性化関数
        self.leakyrelu = nn.Linear(1, 1) # ダミーではなく数式通りに適用
        
    def forward(self, h, adj):
        """
        h: ノード特徴量 [NumNodes, InFeatures]
        adj: 隣接行列   [NumNodes, NumNodes]
        """
        num_nodes = h.size(0)
        
        # 1. 特徴量の線形変換: Wh = H * W  -> [NumNodes, OutFeatures]
        Wh = torch.mm(h, self.W)
        
        # 2. すべてのノードペア(i, j)間の一致度（アテンションの素）を計算
        # 行列演算で高速化するため、全ペアの組み合わせを作る
        Wh_i = Wh.repeat_interleave(num_nodes, dim=0) # [N*N, OutFeatures]
        Wh_j = Wh.repeat(num_nodes, 1)                # [N*N, OutFeatures]
        
        # 2つのベクトルを横に結合: [N*N, 2 * OutFeatures]
        Wh_concat = torch.cat([Wh_i, Wh_j], dim=-1)
        Wh_concat = Wh_concat.view(num_nodes, num_nodes, 2 * self.out_features)
        
        # aとの内積を計算してスコア化 -> [NumNodes, NumNodes]
        # einsumを使うことで「各ペアとベクトル a の掛け算」を美しく記述できます
        score = torch.einsum('ijo,oi->ij', Wh_concat, self.a)
        score = F.leakyrelu(score, negative_slope=self.alpha)
        
        # 3. 隣接行列マスクの適用
        # 繋がっていないノード（adj == 0）へのアテンションを極小値（-9e15）にして、Softmax後に0になるようにする
        zero_vec = -9e15 * torch.ones_like(score)
        # 自分自身へのアテンションも考慮するため、隣接行列に単位行列を考慮
        adj_with_self = adj + torch.eye(num_nodes, device=adj.device)
        attention = torch.where(adj_with_self > 0, score, zero_vec)
        
        # 4. 行（各ノードの近傍）ごとにSoftmaxをかけて確率（重み）に変換
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 5. アテンション重みに基づいて周辺情報を集約
        h_prime = torch.mm(attention, Wh) # [NumNodes, OutFeatures]
        
        if self.concat:
            return F.elu(h_prime) # ELU活性化関数を適用して返す
        else:
            return h_prime

# =====================================================================
# 2. メインネットワーク：Multi-Head GATモデル
# =====================================================================
class MultiHeadGAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, n_heads=8, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        self.n_heads = n_heads
        
        # --- 第1層: マルチヘッド・アテンション ---
        # 独立した複数のアテンション層（ヘッド）をリストとして保持
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(n_feat, n_hid, dropout=dropout, concat=True) 
            for _ in range(n_heads)
        ])
        
        # --- 第2層（出力層）: 各ヘッドの出力を統合して最終クラス分類 ---
        # 入力次元は「隠れ層の次元 * ヘッド数」になります（結合されるため）
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 1. すべてのヘッドの出力を計算して横に結合（Concat）する
        # 各ヘッドの出力 [N, n_hid] -> 結合後 [N, n_hid * n_heads]
        x_heads = [att(x, adj) for att in self.attentions]
        x = torch.cat(x_heads, dim=-1)
        
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 2. 出力層へ流す（ここでは各ヘッドの出力を結合せず、平均または単一アテンションで集約）
        x = self.out_att(x, adj)
        
        return F.log_softmax(x, dim=1)

# =====================================================================
# 3. 動作検証
# =====================================================================
if __name__ == "__main__":
    # パラメータ設定
    num_nodes = 5
    input_dim = 16
    hidden_dim = 8    # 各ヘッドが出力する次元
    num_heads = 4     # ヘッド数（4つの異なる視点で関係性を学習）
    num_classes = 3

    # ダミーのグラフ構造（前回のGCNと同様）
    dummy_features = torch.randn(num_nodes, input_dim)
    dummy_adj = torch.tensor([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=torch.float32)

    # GATモデルの初期化
    model = MultiHeadGAT(n_feat=input_dim, n_hid=hidden_dim, n_class=num_classes, n_heads=num_heads, dropout=0.1)
    model.eval()
    
    # 計算実行
    output = model(dummy_features, dummy_adj)
    
    print("=== Multi-Head GAT スクラッチ実装検証 ===")
    print("入力特徴量の形状:", dummy_features.shape)
    print("マルチヘッド集約層の出力総次元数:", hidden_dim * num_heads)
    print("GAT最終出力の形状:", output.shape)
    print("\n各ノードのクラス予測スコア:")
    print(output)