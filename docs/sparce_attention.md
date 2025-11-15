# Sparse Attention

## 🧠 Sparse Attention Explained

**Sparse Attention** is a technique developed to address the computational and memory limitations of the **Dense Attention** (or **Full Attention**) mechanism, which is the core component of the Transformer architecture.

---

## 🛑 The Problem with Dense Attention (Full Attention)

In standard Dense Attention, every token in a sequence must calculate its relationship (attention score) with **every other token** in that same sequence.

1. **Computational Complexity**: If the sequence length is $N$, the time and memory complexity required to compute the Attention scores is **quadratic** in relation to the sequence length, denoted as $O(N^2)$.
2. **Scaling Limitation**: This $O(N^2)$ complexity makes it prohibitively expensive and often impossible to process **very long sequences** (e.g., long documents, high-resolution images, long audio tracks) because the computational cost and memory consumption grow too rapidly.

---

## 🎯 The Solution: Sparse Attention

Sparse Attention operates on the premise that not every token needs to interact with *all* other tokens to understand the context. It selectively **restricts (sparsifies)** the connections in the attention matrix to reduce the computational cost.

The goal is to reduce the complexity from $O(N^2)$ to something closer to **linear complexity** with respect to $N$, such as $O(N \cdot W)$ (where $W$ is a fixed window size) or $O(N \cdot \sqrt{N})$.

### Key Principle

Instead of calculating and storing the full $N \times N$ attention matrix, Sparse Attention only computes and focuses on a subset of the most relevant connections.

### Common Sparsification Patterns

Sparse Attention models implement different patterns to decide which connections to keep:

| Pattern                          | Principle                                                                                                                                                        | Complexity Example                          |
| :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------ |
| **Local Attention**        | Each token only attends to tokens within a**fixed, adjacent window** around itself. This mimics locality bias in CNNs.                                     | $O(N \cdot W)$                            |
| **Global Attention**       | A few designated**"global" tokens** (like the `[CLS]` token) attend to all tokens, while all other tokens only attend locally or to those global tokens. | $O(N \cdot k)$ or $O(N \cdot \sqrt{N})$ |
| **Random Attention**       | Each token randomly selects a small number of other tokens to attend to.                                                                                         | $O(N \cdot k)$ ($k$ is a small number)  |
| **Block Sparse Attention** | The attention matrix is divided into blocks, and only a chosen subset of these blocks is computed, often following a pre-defined or learned structure.           | $O(N \cdot \sqrt{N})$                     |

### Benefits

* **Longer Sequences**: Enables the processing of much longer input sequences that would otherwise crash due to memory limits.
* **Faster Training/Inference**: Reduces the number of necessary computations, leading to faster model training and prediction times.
* **Memory Efficiency**: Significant reduction in GPU memory consumption because the full attention matrix is never calculated or stored.

Sparse Attention is a crucial enabler for models designed to handle extensive data, such as **LongFormers** and other large-scale models in NLP, vision, and audio processing.

There isn't a single, universally accepted classification of "genres" in Sparse Attention, but the implementations and techniques generally fall into categories based on **how the attention connections are restricted (the sparsification pattern)**.

Here are the main categories of Sparse Attention patterns that exist, organized by their approach to limiting the quadratic complexity $O(N^2)$:

## 🌐 Main Categories of Sparse Attention

### 1. Fixed or Window-Based Sparsity

This is the simplest and most common form of sparsification, where the pattern of allowed connections is fixed and predetermined, usually based on proximity.

* **Local Attention (or Banded Attention)**:

  * **Principle**: Each token is restricted to attending only to tokens within a **fixed, adjacent window** of size $W$ around itself. Connections beyond this window are blocked.
  * **Benefit**: Reduces complexity to $O(N \cdot W)$, which is linear with respect to sequence length $N$ when $W$ is small.
  * **Use Case**: Effective for tasks where context is highly localized, such as short-range dependency modeling.
* **Dilated Attention**:

  * **Principle**: Similar to local attention, but the tokens within the window are not necessarily adjacent; instead, they are **sampled at a fixed interval** (or dilation rate).
  * **Benefit**: Allows the model to capture information from distant tokens without increasing the overall computational cost dramatically.

### 2. Global + Local Sparsity (Mixed Attention)

These models combine the benefits of local proximity with the need to capture critical long-range dependencies.

* **Global Attention**:
  * **Principle**: Designates a few **"global" or "special" tokens** (like the `[CLS]` token, or tokens at fixed intervals) that attend to *all* tokens, and are attended to by *all* tokens. The remaining tokens only attend locally or to the global set.
  * **Benefit**: The global tokens act as information hubs, effectively summarizing context for the entire sequence, thus bridging long distances.
  * **Examples**: Used in models like **Longformer**.

### 3. Data-Dependent or Adaptive Sparsity

These advanced methods allow the model to dynamically choose which connections are important based on the input data itself, rather than relying on a fixed pattern.

* **Learned Sparsity**:

  * **Principle**: The model learns a sparse mask during training. This might involve using a **gating mechanism** or a **pruning strategy** to dynamically zero out attention scores that are deemed unimportant for a given input.
  * **Benefit**: Highly flexible and potentially more efficient, as attention is only paid to truly relevant context.
* **Query/Key Clustering (e.g., Reformer)**:

  * **Principle**: Uses techniques like **Locality-Sensitive Hashing (LSH)** to group similar queries ($\mathbf{Q}$) and keys ($\mathbf{K}$) together. Attention is then only computed *within* these similar clusters.
  * **Benefit**: This effectively makes attention *sparse* by limiting it to semantically relevant neighbors, dramatically reducing complexity, often to $O(N \log N)$.

### 4. Hierarchical Sparsity

* **Principle**: The attention is structured across multiple levels of granularity. Tokens first attend locally, then the output is pooled to create a representation of a chunk (e.g., paragraph or block), and then these chunk representations attend globally.
* **Benefit**: Ideal for handling structured data like very long documents, allowing the model to focus on both fine-grained details and document-level context.

## Implementation
### Local Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalAttention(nn.Module):
    """
    ローカルアテンションのPyTorch実装。
    各トークンは、その周囲の 'window_size' 内のトークンにのみ注目します。
    """
    def __init__(self, d_model, window_size):
        super().__init__()
        self.d_model = d_model
        # window_sizeは奇数を推奨 (例: 5 -> 自分+前後2)
        self.window_size = window_size 
        
        # Q, K, V の線形変換層（シングルヘッドとして簡略化）
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x の形状: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # 1. Q, K, V の計算
        Q = self.query(x)  
        K = self.key(x)    
        V = self.value(x)  

        # 2. アテンションスコア（Q * K^T）の計算
        # scores の形状: (batch_size, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)

        # 3. ローカルアテンションマスクの作成
        
        # half_window: 各トークンの片側（左右）に注目するトークン数
        # 例: window_size=5 の場合, half_window = 2 (自分 + 前2 + 後2)
        half_window = (self.window_size - 1) // 2
        
        # 距離行列の作成: (seq_len, seq_len)
        # i行j列の値は |i - j| (インデックス i と j の距離)
        i = torch.arange(seq_len, device=x.device).unsqueeze(1)
        j = torch.arange(seq_len, device=x.device).unsqueeze(0)
        distance_matrix = torch.abs(i - j)
        
        # ローカルマスク: 距離が half_window 以下なら True (注目許可)
        # local_mask の形状: (seq_len, seq_len)
        local_mask = (distance_matrix <= half_window).to(x.device)

        # 4. マスクの適用
        # 注目不可な部分 (False) のスコアを負の無限大 (-torch.inf) に設定
        # これにより、Softmax適用後にその部分の重みがゼロになります。
        scores = scores.masked_fill(~local_mask, -torch.inf)

        # 5. Softmaxの適用
        attention_weights = F.softmax(scores, dim=-1) # (batch_size, seq_len, seq_len)
        
        # 6. 重みとVの積
        output = torch.matmul(attention_weights, V) # (batch_size, seq_len, d_model)

        return output

# パラメータ設定
d_model = 64     # 特徴次元
seq_len = 50     # シーケンス長
batch_size = 2   # バッチサイズ
window_size = 7  # ローカルアテンションの窓サイズ (例: 自分 + 前後3トークン)

# ダミー入力データ
input_data = torch.randn(batch_size, seq_len, d_model)

# モデルのインスタンス化
local_attn_layer = LocalAttention(d_model=d_model, window_size=window_size)

# 順伝播の実行
output = local_attn_layer(input_data)

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {output.shape}")

# 確認: アテンション重みの一部を表示して、疎結合になっているか確認
# (これはLocalAttentionクラスの内部でしかアクセスできないため、簡易的な確認のみ)
# 
# 仮に内部で計算されたスコアのマスク状態を確認したい場合:
# print(f"ローカルマスクの形状:\n{local_attn_layer.local_mask}")
# 実際に True (注目) の数が seq_len * window_size 程度になっているはずです。
```



## Related Work

[Paper Walkthrough - LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://storrs.io/paper-walkthrough-longnet-scaling-transformers-to-1-000-000-000-tokens/)
