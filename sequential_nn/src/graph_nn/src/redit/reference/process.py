import torch
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# データセットのロード
dataset = Reddit(root='/tmp/Reddit')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

print(f'ノード数: {data.num_nodes}')
print(f'エッジ数: {data.num_edges}')
print(f'特徴量次元: {data.num_node_features}')
print(f'クラス数: {dataset.num_classes}')

# メモリ節約のための NeighborLoader 設定
train_loader = NeighborLoader(
    data,
    num_neighbors=[5, 3],  # 1-hop で 5 ノード、2-hop で 3 ノード（小さめに設定）
    batch_size=512,         # バッチサイズを小さく
    input_nodes=data.train_mask,
    shuffle=True
)

val_loader = NeighborLoader(
    data,
    num_neighbors=[5, 3],
    batch_size=512,
    input_nodes=data.val_mask
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[5, 3],
    batch_size=512,
    input_nodes=data.test_mask
)

# モデルも小さめに設定
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(
    in_channels=data.num_node_features,
    hidden_channels=64,  # 隠れ層を小さく
    out_channels=dataset.num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
    return total_loss / len(train_loader)

@torch.no_grad()
def eval_loader(loader, mask_name='val_mask'):
    model.eval()
    correct = total = 0
    for batch in loader:
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=-1)
        mask = getattr(batch, mask_name)
        correct += int((pred[mask] == batch.y[mask]).sum())
        total += int(mask.sum())
    return correct / total

# 学習実行（エポック数も少なめに）
for epoch in range(1, 21):
    loss = train()
    val_acc = eval_loader(val_loader, mask_name='val_mask')
    if epoch % 5 == 0:
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

# テスト評価
test_acc = eval_loader(test_loader, mask_name='test_mask')
print(f'Test Accuracy: {test_acc:.4f}')