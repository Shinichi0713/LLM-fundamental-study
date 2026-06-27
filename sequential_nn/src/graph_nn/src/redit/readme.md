
最近続けている[GNNの例題シリーズ](https://yoshishinnze.hatenablog.com/entry/2026/06/25/043000)について話していきます。
今回はネットワークで比較的大きなデータセットを使ってみたいということでRedditというデータセットを使ってみます。

本日テーマ：
>Redditを使った学習について取り組んでみる

## Redditとは

Reddit データセットは、**大規模グラフ上での GNN のスケーラビリティを評価するための代表的なベンチマーク**です。  
PyTorch Geometric では `torch_geometric.datasets.Reddit` として提供されています[PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.Reddit.html)。

### 1. データセットの由来

- 元論文：**“Inductive Representation Learning on Large Graphs”**（Hamilton et al., 2017）  
- 目的：大規模グラフ上で GNN を学習するための**スケーラビリティ評価用ベンチマーク**として設計されました。

### 2. グラフの構造（ノードとエッジ）

- **ノード**：Reddit の投稿（post）  
  - ノード数：約 **232,965** ノード[Kumo.ai PyG Guide](https://kumo.ai/pyg/datasets/reddit)。
- **エッジ**：投稿間の「共コメント（co-comment）」関係  
  - エッジ数：約 **114,615,892** エッジ（約 1.14 億）[Kumo.ai PyG Guide](https://kumo.ai/pyg/datasets/reddit)。
  - 同じスレッド内で同じユーザーがコメントした投稿同士をリンクさせたものとされています。

### 3. ノード特徴量

- 特徴量次元：**602 次元**
- 内容：各ノード（投稿）の **Bag-of-Words（BoW）表現**  
  - 投稿テキストを単語単位でベクトル化したもの（単語の出現頻度など）です[Kumo.ai PyG Guide](https://kumo.ai/pyg/datasets/reddit)。

### 4. ラベルとタスク

- **クラス数**：**41 クラス**
- ラベルの意味：投稿が属する **Reddit のコミュニティ（サブレディット）**  
  - 例：`r/aww`, `r/funny`, `r/AskReddit` など、人気の高いサブレディット群。
- **タスク**：ノード分類（コミュニティ分類）  
  - 各投稿（ノード）がどのサブレディットに属するかを予測する、**マルチクラス分類問題**です。

### 5. マスク（学習・検証・テスト分割）

- `train_mask` / `val_mask` / `test_mask` が付与されています。
- それぞれ `[num_nodes]` の `bool` テンソルで、対応するノードが学習・検証・テストに使われるかを示します。
- タスク設定は**inductive（帰納的）**：  
  - 学習時に見たノードと、テスト時に評価するノードが明確に分かれています。

### 6. データセットの特徴・用途

- **大規模かつスパース**：  
  - ノード数 23 万、エッジ数 1.1 億と非常に大きく、**フルバッチ学習では GPU メモリに載りきらない**ため、  
    GraphSAGE や ClusterGCN、GraphSAINT などの**サンプリング・ミニバッチ手法**の評価に適しています[Kumo.ai PyG Guide](https://kumo.ai/pyg/datasets/reddit)。
- **コミュニティ構造がはっきりしている**：  
  - サブレディットごとに話題が分かれているため、GNN がうまくコミュニティを捉えられれば高い精度が出やすいです。
- **GNN スケーラビリティ研究の標準ベンチマーク**：  
  - 「Reddit でうまく動く手法は、実運用レベルの大規模グラフにも応用できる」という位置づけでよく使われます[Kumo.ai PyG Guide](https://kumo.ai/pyg/datasets/reddit)。


## データの可視化

このデータセットですが
- 非常に特徴量が多く(602次元)
- ラベルが多い(Reddit のコミュニティに相当で41クラス)

ちょっとイメージが難しい面があります。

### グラフ全体の統計情報の確認

パッとデータ全貌を確認するのであれば、以下コードを実行下さい。

```python
print(f'ノード数: {data.num_nodes}')
print(f'エッジ数: {data.num_edges}')
print(f'特徴量次元: {data.num_node_features}')
print(f'クラス数: {dataset.num_classes}')

if hasattr(data, 'train_mask'):
    print(f'train_mask shape: {data.train_mask.shape}')
    print(f'train ノード数: {data.train_mask.sum().item()}')
if hasattr(data, 'val_mask'):
    print(f'val_mask shape: {data.val_mask.shape}')
    print(f'val ノード数: {data.val_mask.sum().item()}')
if hasattr(data, 'test_mask'):
    print(f'test_mask shape: {data.test_mask.shape}')
    print(f'test ノード数: {data.test_mask.sum().item()}')

# エッジが無向か有向か
print(f'エッジインデックスの形状: {data.edge_index.shape}')
print(f'エッジ属性の有無: {hasattr(data, "edge_attr")}')
```

これにより、Reddit データセットが

- 約 23 万ノード
- 約 1.1 億エッジ

であることが分かります[PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.Reddit.html)[Kumo.ai PyG Guide](https://kumo.ai/pyg/datasets/reddit)。


### サブグラフの可視化（NetworkX を使う例）

グラフ全体は大きすぎるので、**学習用ノードの一部とその近傍だけを取り出して可視化**します。

```python
import networkx as nx
import matplotlib.pyplot as plt

# 学習ノードのうち最初の 1 つを選ぶ
train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
center_node = train_idx[0].item()

# 1-hop 近傍を取得（NeighborLoader の仕組みを真似る）
edge_index = data.edge_index
center_mask = (edge_index[0] == center_node)
neighbors = edge_index[1, center_mask].unique()

# 中心ノードとその近傍だけを含むノード集合
sub_nodes = torch.cat([torch.tensor([center_node]), neighbors])

# サブグラフのエッジを抽出
sub_mask = torch.isin(edge_index[0], sub_nodes) & torch.isin(edge_index[1], sub_nodes)
sub_edge_index = edge_index[:, sub_mask]

# NetworkX のグラフに変換
G = nx.Graph()
G.add_edges_from(sub_edge_index.t().tolist())

# 可視化
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_size=50, node_color='lightblue', edge_color='gray', width=0.5)
plt.title(f'Subgraph around training node {center_node}')
plt.show()
```

- これにより、ある投稿（ノード）と、その投稿にコメントした他の投稿（近傍ノード）のつながりを可視化できます。
- ノード数が多すぎる場合は、`neighbors = neighbors[:50]` のように近傍数を制限してください。

かなりエッジの多い構造であることが分かります。
これがRedditというSNSの特性かもしれません。

![1782529106554](image/readme/1782529106554.png)

### ラベルの例

```python
# 最初の 10 ノードのラベル
print("最初の 10 ノードのラベル（クラスID）:")
print(data.y[:10])

# 各クラスに属するノード数の分布（おおまか）
print("\n各クラスのノード数（上位 10 クラス）:")
unique, counts = torch.unique(data.y, return_counts=True)
for cls, cnt in zip(unique[:10], counts[:10]):
    print(f"クラス {cls.item()}: {cnt.item()} ノード")
```

- ラベルは 0〜40 の整数で、どのサブレディット（コミュニティ）に属するかを表します。


### 次元圧縮
次元圧縮して少しイメージしやすくなるかと期待してumapで3D散布図にしてみました。

![1782528892651](image/readme/1782528892651.png)

クラス0のみがかなり孤立していますが、それ以外は分離しているとは言いづらい状態です。
特徴量のみで分類ということが難しいということが伝わるようなデータセットです。

## 実験

Reddit データセットに GNN を適用して、ラベル(クラス)の実験をしてみようと思います。

### 1. Reddit データセットでよく使われるタスク

Reddit データセットでは、主に以下のようなタスクが考えられます。

1. **ノード分類（Node Classification）**  
   - 各ノード（投稿）がどのコミュニティ（サブレディット）に属するかを予測する。
   - 例：ある投稿が `r/aww` なのか `r/funny` なのかを予測。

2. **リンク予測（Link Prediction）**  
   - まだ存在しないエッジ（投稿間の共コメント関係）を予測する。
   - 例：「この 2 つの投稿は将来的に同じユーザーからコメントされるか？」を予測。

3. **グラフ分類（Graph Classification）**  
   - Reddit 全体ではなく、スレッド単位の小さなグラフを作り、そのグラフがどのカテゴリか（例：政治系・娯楽系）を分類する。

この中で、**ノード分類が最も直感的で、GNN の効果が分かりやすい**と思いました。

### 2. 手順
以下のフローでノード分類の学習の実験をしてみます。

1. データのロードと前処理  
Reddit データセットをロードし、GPU に転送。
2. ミニバッチ用サンプラの準備  
NeighborLoader で学習・検証・テスト用のミニバッチを作成。
3. GNN モデルの定義  
例：2 層 GCN。
4. 学習ループ  
ミニバッチごとに順伝播・損失計算・逆伝播。
5. テストセットでの評価  
学習済みモデルでテストノードの精度を計算。

## 実装
手順で記載した流れで学習してみます。
Colab でそのまま実行できる形でまとめます。


### 1. データのロードと前処理

必要なパッケージのインストールです。

```
# PyTorch 本体（Colab のデフォルトでも可）
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric 本体と依存ライブラリ
!pip install torch_geometric
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

データをロードしていきます。

```python
import torch
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F

# データセットのダウンロード
dataset = Reddit(root='/tmp/Reddit')
data = dataset[0]

# GPU に転送
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

print(f'ノード数: {data.num_nodes}')
print(f'エッジ数: {data.num_edges}')
print(f'特徴量次元: {data.num_node_features}')
print(f'クラス数: {dataset.num_classes}')
```

### 2. ミニバッチ用サンプラの準備（NeighborLoader）

Reddit は巨大なので、全グラフを一度に GPU に載せるのではなく、近傍サンプリングでミニバッチ学習します。

```python
# 学習用サンプラ
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],  # 1-hop で 10 ノード、2-hop で 5 ノードをサンプル
    batch_size=1024,
    input_nodes=data.train_mask,
    shuffle=True
)

# 検証用サンプラ
val_loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],
    batch_size=1024,
    input_nodes=data.val_mask
)

# テスト用サンプラ（後で使う）
test_loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],
    batch_size=1024,
    input_nodes=data.test_mask
)
```

- `num_neighbors` や `batch_size` は GPU メモリに応じて調整してください。

### 3. GNN モデルの定義（例：2 層 GCN）

前回も使ったGCNと呼ばれるシンプルなGNNを用います。

```python
from torch_geometric.nn import GCNConv

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
    hidden_channels=128,
    out_channels=dataset.num_classes
).to(device)
```


### 4. 学習ループの実装

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # 学習マスク部分のみで損失を計算
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
        # mask_name に応じてマスクを選択
        mask = getattr(batch, mask_name)
        correct += int((pred[mask] == batch.y[mask]).sum())
        total += int(mask.sum())
    return correct / total

# 学習実行（例: 50 エポック）
for epoch in range(1, 51):
    loss = train()
    val_acc = eval_loader(val_loader, mask_name='val_mask')
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
```

---

## 6. テストセットでの評価

学習が終わったら、テスト用サンプラで精度を確認します。

```python
test_acc = eval_loader(test_loader, mask_name='test_mask')
print(f'Test Accuracy: {test_acc:.4f}')
```

- Reddit データセットでは、GCN でおおよそ 93% 前後、GraphSAGE や ClusterGCN で 95〜97% 程度の精度が報告されています[Kumo.ai PyG Guide](https://kumo.ai/pyg/datasets/reddit)。



