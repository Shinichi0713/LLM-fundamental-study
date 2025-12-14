LLMにおける **MoE（Mixture of Experts）** を「有効かつ現実的」に理解できる実装例として、
**TransformerのFFNをMoEに置き換える最小構成**を示します。

目的は以下です。

* LLMで実際に使われる **MoEの本質構造を理解**
* 研究・試作・社内検証で **そのまま流用できる**
* 実装過剰にならない（Router + Expert + Load balancing）

---

## 実装方針（LLM実務に即した設計）

### MoEで置き換える場所

* **Self-AttentionはDenseのまま**
* **FFNのみをMoE化** （業界標準）

```
Transformer Block
 ├ Self-Attention (Dense)
 └ FFN → MoE FFN
```

---

## 1. MoE FFN の最小実装（PyTorch）

### 1.1 Expert（通常のFFN）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

```python
class Expert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
```

---

### 1.2 Router（Top-k Gating）

```python
class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: [batch, seq, d_model]
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        return probs
```

---

### 1.3 MoE FFN（Top-1 MoE）

```python
class MoEFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden) for _ in range(num_experts)]
        )
        self.router = Router(d_model, num_experts)

    def forward(self, x):
        # x: [batch, seq, d_model]
        gate_probs = self.router(x)  # [B, S, E]

        # Top-1 routing
        expert_idx = torch.argmax(gate_probs, dim=-1)  # [B, S]

        output = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                expert_input = x[mask]
                expert_output = expert(expert_input)
                output[mask] = expert_output

        return output
```

✔ **1トークンにつき1 Expertのみ実行**
✔ 計算量は Dense FFN とほぼ同等
✔ パラメータ容量は Expert数分増加

---

## 2. Transformer Block への統合

```python
class TransformerBlockMoE(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden, num_experts):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.moe_ffn = MoEFFN(d_model, d_hidden, num_experts)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.moe_ffn(x)
        x = self.norm2(x + ffn_out)

        return x
```

---

## 3. 実際のLLMで「有効」になる理由

### なぜこの実装が意味を持つのか

| 観点         | Dense FFN        | MoE FFN        |
| ------------ | ---------------- | -------------- |
| 実行計算量   | 常に全パラメータ | 一部Expertのみ |
| モデル容量   | 固定             | Expert数に比例 |
| 専門性       | 混在             | 自然分化       |
| スケーリング | 限界あり         | 非常に良い     |

---

## 4. 実運用で必須になる改良点（重要）

上記は **理解用の最小実装**です。
実際のLLMでは以下を追加します。

### 4.1 Top-2 Gating（Mixtral方式）

* 安定性向上
* 勾配消失対策

### 4.2 Load Balancing Loss

```python
# Expert利用率を均等化
loss_balance = (gate_probs.mean(dim=(0,1)) ** 2).sum()
```

---

### 4.3 Expert並列化

* GPU / ノード単位でExpert分散
* DeepSpeed MoE / Megatron-LM

---

## 5. 実務レベルでの代表例

| モデル             | MoE方式               |
| ------------------ | --------------------- |
| Switch Transformer | Top-1 MoE             |
| GLaM               | Top-2 MoE             |
| Mixtral 8x7B       | Decoder MoE           |
| DeepSeek-MoE       | Dense + Sparse Hybrid |

---

## 6. 意思決定用まとめ

### MoEを実装すべきタイミング

* 事前学習 or 大規模再学習
* パラメータ容量がボトルネック
* 計算資源に制約がある

### 実装しない方が良い場合

* 小規模FT
* 単一タスク
* 低レイテンシ必須推論

---

## 一言で整理すると

> **MoEは「FFNを疎にすることで、計算量を固定したまま知識容量を拡張する構造」**

PEFTや量子化が **後段最適化**なのに対し、
MoEは **モデル構造そのもののスケーリング戦略**です。

次に進むなら

* Mixtralの実装分解
* MoE × LoRA の併用
* Routerの学習安定化

まで一気に踏み込めます。
