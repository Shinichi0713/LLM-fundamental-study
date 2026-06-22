import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 1層目: 入力特徴 → 隠れ表現
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # 2層目: 隠れ表現 → クラススコア
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


from torchviz import make_dot

# ダミー入力で forward を通し、計算グラフを可視化
model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes
)

# ダミー入力（バッチサイズ1のノード特徴とエッジ）
x_dummy = torch.randn(1, dataset.num_features)
edge_index_dummy = torch.tensor([[0], [0]], dtype=torch.long)

# forward を通す
out = model(x_dummy, edge_index_dummy)

# 計算グラフを可視化
dot = make_dot(out, params=dict(model.named_parameters()))
dot.render('gcn_model', format='png')  # PNG ファイルとして保存（Colab 左ペインに表示される）


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