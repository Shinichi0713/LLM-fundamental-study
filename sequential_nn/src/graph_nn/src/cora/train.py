import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Cora データセットの読み込み
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# GCN モデル（先ほどのクラスを再利用）
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# モデルとオプティマイザの設定
model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

# 学習用関数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 評価用関数
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = (pred[mask] == data.y[mask]).float().mean().item()
        accs.append(acc)
    return accs

# 学習ループ（早期終了付き）
max_epochs = 200
patience = 40  # 検証精度が改善しないエポック数の許容範囲
best_val_acc = 0.0
epochs_without_improve = 0

print("開始: GCN の学習（Cora ノード分類）")
print("=" * 50)

for epoch in range(1, max_epochs + 1):
    loss = train()
    train_acc, val_acc, test_acc = test()
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    # 早期終了の判定
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improve = 0
        # 必要ならここでモデルの重みを保存
        # torch.save(model.state_dict(), 'best_gcn_cora.pth')
    else:
        epochs_without_improve += 1
    
    if epochs_without_improve >= patience:
        print(f"\n早期終了: {patience} エポック連続で検証精度が改善しませんでした。")
        print(f"最終検証精度: {best_val_acc:.4f}")
        break

# 最終的なテスト精度を表示
final_train_acc, final_val_acc, final_test_acc = test()
print("\n" + "=" * 50)
print("最終結果:")
print(f"学習用精度: {final_train_acc:.4f}")
print(f"検証用精度: {final_val_acc:.4f}")
print(f"テスト用精度: {final_test_acc:.4f}")