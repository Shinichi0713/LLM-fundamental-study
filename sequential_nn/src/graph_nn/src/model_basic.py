import torch
import torch.nn as nn
import torch.nn.functional as F

# 簡単なグラフ例：4ノード、エッジは 0-1, 1-2, 2-3, 3-0
edge_index = torch.tensor([[0, 1, 2, 3],
                           [1, 2, 3, 0]], dtype=torch.long)  # shape: (2, num_edges)

# 各ノードの特徴量（例：2次元）
x = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0],
                  [0.0, 0.0]], dtype=torch.float)

num_nodes = x.size(0)

class SimpleMessagePassingLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # メッセージ計算用の線形層（送信元ノード特徴を変換）
        self.msg_lin = nn.Linear(in_dim, out_dim)
        # 更新用の線形層（集約メッセージ＋自己ループを変換）
        self.update_lin = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x, edge_index):
        """
        x: (num_nodes, in_dim)
        edge_index: (2, num_edges)
        """
        src, dst = edge_index  # src -> dst のエッジ

        # 1. メッセージ計算：各エッジについて送信元ノードの特徴を変換
        msg = self.msg_lin(x[src])  # (num_edges, out_dim)

        # 2. 集約：各ノードが受け取るメッセージを合計
        agg = torch.zeros(x.size(0), msg.size(1), device=x.device)
        agg = agg.index_add_(0, dst, msg)  # dstノードごとにmsgを加算

        # 3. 更新：自身の特徴と集約メッセージを結合して変換
        #    ここでは自己ループとして元の特徴 x も使う
        combined = torch.cat([x, agg], dim=-1)  # (num_nodes, in_dim + out_dim)
        new_x = self.update_lin(combined)

        return new_x
    
class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        # 入力層（最初のメッセージパッシング層）
        self.layers = nn.ModuleList()
        self.layers.append(SimpleMessagePassingLayer(input_dim, hidden_dim))

        # 中間層
        for _ in range(num_layers - 2):
            self.layers.append(SimpleMessagePassingLayer(hidden_dim, hidden_dim))

        # 出力層（最後のメッセージパッシング層）
        if num_layers > 1:
            self.layers.append(SimpleMessagePassingLayer(hidden_dim, output_dim))
        else:
            # num_layers == 1 の場合は input_dim -> output_dim
            self.layers.append(SimpleMessagePassingLayer(input_dim, output_dim))

    def forward(self, x, edge_index):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            # 非線形活性化（ReLUなど）を挟む
            h = F.relu(h)
        return h

# モデル定義
model = SimpleGNN(
    input_dim=2,   # xの特徴次元
    hidden_dim=16,
    output_dim=8, # 最終的なノード表現の次元
    num_layers=2
)

# フォワード計算
node_embeddings = model(x, edge_index)

print("入力特徴量 shape:", x.shape)           # (4, 2)
print("出力ノード埋め込み shape:", node_embeddings.shape)  # (4, 8)

class SimpleGNNWithReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim, node_out_dim, graph_out_dim):
        super().__init__()
        # ノード表現を学習するGNN
        self.gnn = SimpleGNN(input_dim, hidden_dim, node_out_dim, num_layers=2)
        # グラフ読み出し用MLP（全ノードを集約してグラフ表現に変換）
        self.readout_mlp = nn.Sequential(
            nn.Linear(node_out_dim, graph_out_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, batch=None):
        """
        x: (num_nodes, input_dim)
        edge_index: (2, num_edges)
        batch: 各ノードが属するグラフID（単一グラフの場合はNoneでOK）
        """
        # 1. ノード表現をGNNで更新
        node_emb = self.gnn(x, edge_index)  # (num_nodes, node_out_dim)

        # 2. グラフ読み出し（ここでは全ノードの平均を取る）
        if batch is None:
            # 単一グラフの場合：全ノード平均
            graph_emb = node_emb.mean(dim=0, keepdim=True)  # (1, node_out_dim)
        else:
            # 複数グラフが1つのバッチにまとまっている場合：グラフごとに平均
            # ここでは簡略化のため省略（実際は scatter_mean などを使う）
            raise NotImplementedError("batch処理は省略しています")

        # 3. グラフ表現をMLPで変換
        graph_out = self.readout_mlp(graph_emb)  # (1, graph_out_dim)

        return graph_out

from torchviz import make_dot

# モデルとダミー入力
model = SimpleGNN(input_dim=2, hidden_dim=16, output_dim=8, num_layers=2)
x_dummy = torch.randn(4, 2)  # ノード数4, 特徴次元2
edge_index_dummy = torch.tensor([[0,1,2,3],[1,2,3,0]], dtype=torch.long)

# 一度フォワード計算してグラフを取得
out = model(x_dummy, edge_index_dummy)

# 計算グラフを可視化（出力ノードとパラメータを表示）
dot = make_dot(out, params=dict(model.named_parameters()))
dot.render("gnn_architecture", format="png")  # gnn_architecture.png が生成される