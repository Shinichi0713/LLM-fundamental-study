import torch
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F

# データセットのダウンロード
dataset = Reddit(root='/tmp/Reddit')
data = dataset[0]

print(f'ノード数: {data.num_nodes}')
print(f'エッジ数: {data.num_edges}')
print(f'特徴量次元: {data.num_node_features}')
print(f'クラス数: {dataset.num_classes}')
print(f'学習/検証/テストマスクの有無: {data.has_train_mask()}, {data.has_val_mask()}, {data.has_test_mask()}')


# GPU にデータを転送
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# NeighborLoader でミニバッチを作成
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],  # 1-hop で 10 ノード、2-hop で 5 ノードをサンプル
    batch_size=1024,
    input_nodes=data.train_mask,
    shuffle=True
)

val_loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],
    batch_size=1024,
    input_nodes=data.val_mask
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],
    batch_size=1024,
    input_nodes=data.test_mask
)

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
def test(loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=-1)
        # 検証/テストマスク部分のみ評価
        mask = batch.val_mask if loader == val_loader else batch.test_mask
        correct += int((pred[mask] == batch.y[mask]).sum())
        total += int(mask.sum())
    return correct / total

# 学習実行
for epoch in range(1, 51):
    loss = train()
    val_acc = test(val_loader)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

test_acc = test(test_loader)
print(f'Test Accuracy: {test_acc:.4f}')