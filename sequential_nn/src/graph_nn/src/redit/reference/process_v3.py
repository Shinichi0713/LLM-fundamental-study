import torch
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# 1. データセットのロード (CPU上に配置)
dataset = Reddit(root='/tmp/Reddit')
data = dataset[0]

# ★絶対ここで data = data.to(device) をしない！
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'ノード数: {data.num_nodes}')
print(f'エッジ数: {data.num_edges}')
print(f'特徴量次元: {data.num_node_features}')
print(f'クラス数: {dataset.num_classes}')

# 2. NeighborLoader の設定 (pin_memory=True を追加)
train_loader = NeighborLoader(
    data,
    num_neighbors=[5, 3],
    batch_size=256,         # 念のためバッチサイズを256に落とすとより安全です
    input_nodes=data.train_mask,
    shuffle=True,
    pin_memory=True
)

val_loader = NeighborLoader(
    data,
    num_neighbors=[5, 3],
    batch_size=256,
    input_nodes=data.val_mask,
    pin_memory=True
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[5, 3],
    batch_size=256,
    input_nodes=data.test_mask,
    pin_memory=True
)

# 3. モデル定義
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
    hidden_channels=64,
    out_channels=dataset.num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 4. 学習ループ (ループ内で .to(device) を行う)
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        # ★ここで必要なミニバッチだけを GPU に送る
        batch = batch.to(device)
        
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        
        # ★バッチ内の特定の train_mask を使用
        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
    return total_loss / len(train_loader)

# 5. 評価関数
@torch.no_grad()
def eval_loader(loader, mask_name='val_mask'):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=-1)
        
        # 1. バッチから指定されたマスク（'val_mask' または 'test_mask'）を取得
        mask = getattr(batch, mask_name)
        
        # 2. 【超重要】NeighborLoader がサンプリングしたルートノードの数（＝元のバッチサイズ）
        # 以降のノードは近傍（Neighbor）として追加されたノードなので、評価からは除外する
        batch_size = batch.batch_size
        
        # 3. ルートノードの範囲内だけでマスクを有効にする
        root_mask = mask[:batch_size]
        
        # 4. 正解率の計算（ルートノードかつマスクが True のものだけを集計）
        correct += int((pred[:batch_size][root_mask] == batch.y[:batch_size][root_mask]).sum())
        total += int(root_mask.sum())
        
    return correct / total if total > 0 else 0.0

# 学習実行
for epoch in range(1, 11):  # 10エポック
    loss = train()
    
    # 毎エポックの評価は重いので5エポックごとに変更
    if epoch % 5 == 0 or epoch == 1:
        # ★引数に 'val_mask' を明示的に指定
        val_acc = eval_loader(val_loader, mask_name='val_mask')
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

# テスト評価
# ★引数に 'test_mask' を明示的に指定
test_acc = eval_loader(test_loader, mask_name='test_mask')
print(f'Test Accuracy: {test_acc:.4f}')