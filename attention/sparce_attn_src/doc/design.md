以下では、**Sparse Attention における Local Attention × Global Attention 構成**について、
**設計思想 → 数学的定義 → 実装パターン → PyTorch 実装例**の順で体系的に説明します。
（Longformer / BigBird / LED 系の設計を一般化した説明です）

---

## 1. 背景と設計思想

### 1.1 なぜ Sparse Attention が必要か

通常の Self-Attention は

[
\mathcal{O}(N^2)
]

の計算・メモリコストを持つため、長系列（数万トークン）では実用的ではありません。

Sparse Attention は
**「重要な依存関係だけを残し、それ以外を切り捨てる」**
ことで、以下を同時に達成します。

* 計算量削減（ほぼ線形）
* 局所文脈の保持
* 文書全体のグローバルな情報集約

---

## 2. Local Attention × Global Attention の基本構造

### 2.1 Local Attention（局所注意）

各トークン *i* は、固定ウィンドウ幅 *w* 内の近傍トークンにのみ注意します。

[
\mathcal{A}_{local}(i) = {j \mid |i - j| \le w}
]

**特徴**

* CNN 的な局所文脈モデリング
* 計算量：(\mathcal{O}(N \cdot w))

---

### 2.2 Global Attention（グローバル注意）

特定のトークン集合 (G)（例：`[CLS]`、文頭、質問トークンなど）が
**全トークンと相互注意**を行います。

[
\mathcal{A}_{global}(i) =
\begin{cases}
{1, \dots, N} & i \in G 
G & i \notin G
\end{cases}
]

**特徴**

* 文書全体の集約・ブロードキャスト
* 情報のハブとして機能

---

### 2.3 両者の合成

最終的な注意集合：

[
\mathcal{A}(i) = \mathcal{A} *{local}(i) \cup \mathcal{A}* {global}(i)
]

これにより、

* **局所情報 → Local**
* **全体構造 → Global**

の両立が可能になります。

---

## 3. 実装パターン（実務で使われる3方式）

### パターン①：Attention Mask による制御（最も一般的）

* Full Attention を計算
* 許可されていない位置を `-inf` マスク

**メリット**

* 実装が単純
* Transformers 互換

**デメリット**

* 理論的には sparse だが、実装は dense

---

### パターン②：Block Sparse Attention（高速・本格派）

* Attention 行列を **ブロック単位で疎化**
* Triton / CUDA kernel を使用

**代表例**

* Longformer CUDA kernel
* xFormers block-sparse

---

### パターン③：Global token 専用 attention head

* Global token 用の head を分離
* 他トークンは local head のみ

**代表例**

* BigBird の一部構成

---

## 4. PyTorch による実装例（Mask方式）

### 4.1 Attention Mask の構築

```python
import torch

def build_local_global_mask(seq_len, window, global_indices):
    """
    seq_len: トークン長
    window: local attention 半径
    global_indices: global attention を持つトークン index
    """
    mask = torch.full((seq_len, seq_len), float("-inf"))

    for i in range(seq_len):
        # Local attention
        left = max(0, i - window)
        right = min(seq_len, i + window + 1)
        mask[i, left:right] = 0

        # Global attention
        mask[i, global_indices] = 0
        if i in global_indices:
            mask[i, :] = 0

    return mask
```

---

### 4.2 Self-Attention への適用

```python
import torch.nn.functional as F
import math

def sparse_attention(Q, K, V, attn_mask):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores + attn_mask
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)
```

---

### 4.3 使用例

```python
seq_len = 512
d_model = 768
window = 8
global_tokens = torch.tensor([0])  # CLS token

Q = torch.randn(seq_len, d_model)
K = torch.randn(seq_len, d_model)
V = torch.randn(seq_len, d_model)

mask = build_local_global_mask(seq_len, window, global_tokens)
output = sparse_attention(Q, K, V, mask)
```

---

## 5. 代表モデルとの対応関係

| モデル     | Local              | Global       | 備考            |
| ---------- | ------------------ | ------------ | --------------- |
| Longformer | Sliding Window     | 任意トークン | 実用性が高い    |
| BigBird    | Window + Random    | CLS          | 理論保証あり    |
| LED        | Encoder Longformer | Decoder Full | 要約特化        |
| GPT 系     | —                 | —           | Dense Attention |

---

## 6. 設計上の重要ポイント

### 6.1 Global token の選び方

* `[CLS]`
* 文頭 / 文末
* 質問トークン
* 学習可能な summary token

### 6.2 Window サイズのトレードオフ

* 小：高速・短距離特化
* 大：精度向上・計算増加

---

## 7. SSM・Mamba との比較（補足）

| 観点       | Sparse Attention    | SSM        |
| ---------- | ------------------- | ---------- |
| 長距離依存 | Global token で対応 | 自然に対応 |
| 並列性     | 高い                | 高い       |
| 解釈性     | 高い                | 低め       |

---

## 8. まとめ

* **Local × Global Attention** は Sparse Attention の中核設計
* 実装は
  * Mask ベース（簡単）
  * Block-sparse（高速）
    の二系統
* Longformer / BigBird の理解が最短ルート

---

必要であれば次に、

* Block-sparse 実装（xFormers / Triton）
* Attention 可視化（ヒートマップ）
* BigBird の random attention を含めた拡張

まで踏み込んで解説できます。
