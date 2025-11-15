# アテンションハンズオン

Transformerの**アテンション機構（Attention Mechanism）** は、

「入力のどの部分にどれだけ注目すべきか」を学習する中核部です。

ここでは、**Google Colabで動かせる完全なハンズオン例**として、

* 小さな入力系列を使って
* **自己注意(Self-Attention)** の動きを
* 数値・可視化・重みのマップで理解できるコード

  を紹介します。

---

## 🧠 ハンズオン概要

> 「入力の3単語（トークン）が、互いにどれだけ注目しているか」を可視化します。
>
> この例では **PyTorchの基本演算のみ** を使って、Transformerの仕組みを透明化します。

---

## 🚀 Colab用 完全コード

```python
# ============================
# Transformer Attention 実験
# ============================
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 入力 (例: 3つのトークン, 各トークンを4次元ベクトルで表現)
x = torch.tensor([
    [1.0, 0.5, 0.3, 0.2],   # token 1
    [0.2, 0.1, 0.9, 0.7],   # token 2
    [0.8, 0.3, 0.2, 0.4]    # token 3
])  # shape: [3, 4]

d_model = x.shape[1]
print("入力ベクトル形状:", x.shape)

# === Q, K, V の線形変換 ===
W_Q = torch.randn(d_model, d_model)
W_K = torch.randn(d_model, d_model)
W_V = torch.randn(d_model, d_model)

Q = x @ W_Q   # shape: [3, 4]
K = x @ W_K
V = x @ W_V

# === Attentionスコア計算 ===
scores = (Q @ K.T) / np.sqrt(d_model)  # 内積をスケーリング
attention_weights = F.softmax(scores, dim=-1)  # ソフトマックス正規化

print("Attentionスコア行列:\n", scores)
print("Attention重み:\n", attention_weights)

# === 出力 (加重平均) ===
output = attention_weights @ V

print("出力ベクトル:\n", output)

# === 可視化 ===
tokens = ["Token 1", "Token 2", "Token 3"]

plt.figure(figsize=(5,4))
sns.heatmap(attention_weights.detach().numpy(), annot=True, cmap="Blues",
            xticklabels=tokens, yticklabels=tokens)
plt.title("Self-Attention Weight Matrix")
plt.xlabel("Key (参照側)")
plt.ylabel("Query (注目側)")
plt.show()
```

---

## 🧩 実行結果の解釈

1. **Attentionスコア行列**

   → 各トークンが他トークンにどれだけ「注目」しているかを示す。
   例：

   ```
   [[0.8, 0.1, 0.1],
    [0.3, 0.6, 0.1],
    [0.2, 0.2, 0.6]]
   ```

   * Token1 は自分自身(1)に強く注意
   * Token2 は2番目に注意を多く
   * Token3 はやや分散している
2. **Softmax**

   → 各行が確率分布になる。

   （全ての重みの合計が1になる）
3. **出力ベクトル**

   → Attention重みをもとに、Valueベクトルの加重平均を計算したもの。

   これが次の層に渡されます。
4. **ヒートマップ可視化**

   ![heatmap sample](https://i.imgur.com/EkP1Lvs.png)

   * 行（Query側）: 「どのトークンが注目しているか」
   * 列（Key側）: 「どのトークンに注目しているか」

---

## 🔍 さらに理解を深めるには

次のステップとして、Colabで以下も試すと非常に理解が深まります：

| ステップ    | 内容                                                     |
| ----------- | -------------------------------------------------------- |
| 🧩 Step 1   | トークン数を5〜10に増やす（系列長の変化）                |
| 🔁 Step 2   | `d_model`を大きくして多次元ベクトルの挙動を観察        |
| 🎯 Step 3   | `Multi-Head Attention`を実装（head数を分けて平均）     |
| 🖼️ Step 4 | 実際の文章埋め込みを入れて、どの単語が注目されるか可視化 |



## 可視化


では、先ほどの **Transformer Self-Attention の数値実験**を

「実際に視覚的に理解できる可視化付き版」にしてみましょう。

Google Colabでそのまま動かすと、

入力 → Q・K・Vベクトル → Attentionスコア → 重みヒートマップ

の流れが**全部グラフィカルに見える**ようになります。

---

## 🎨 Colab用 完全可視化コード

```python
# ===============================
# Self-Attention 可視化ハンズオン
# ===============================
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==== 入力トークン定義 ====
tokens = ["Tokyo", "is", "beautiful"]
x = torch.tensor([
    [1.0, 0.5, 0.3, 0.2],   # Tokyo
    [0.2, 0.1, 0.9, 0.7],   # is
    [0.8, 0.3, 0.2, 0.4]    # beautiful
])  # shape: [3, 4]

d_model = x.shape[1]
print("入力ベクトル形状:", x.shape)

# ==== Q, K, V の線形変換 ====
torch.manual_seed(42)  # 再現性のため固定
W_Q = torch.randn(d_model, d_model)
W_K = torch.randn(d_model, d_model)
W_V = torch.randn(d_model, d_model)

Q = x @ W_Q   # [3, 4]
K = x @ W_K   # [3, 4]
V = x @ W_V   # [3, 4]

# ==== Attentionスコアと重み ====
scores = (Q @ K.T) / np.sqrt(d_model)
attention_weights = F.softmax(scores, dim=-1)

# ==== 出力 ====
output = attention_weights @ V

# ==== 数値出力 ====
print("Attentionスコア:\n", scores)
print("\nAttention重み:\n", attention_weights)
print("\n出力ベクトル:\n", output)

# =======================
# ==== 可視化パート ====
# =======================

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# ① 入力ベクトル可視化
sns.heatmap(x.numpy(), annot=True, cmap="YlGnBu", ax=axes[0],
            xticklabels=[f"d{i+1}" for i in range(d_model)],
            yticklabels=tokens)
axes[0].set_title("Input token embeddings (x)")

# ② Attentionスコア
sns.heatmap(scores.detach().numpy(), annot=True, cmap="OrRd", ax=axes[1],
            xticklabels=tokens, yticklabels=tokens)
axes[1].set_title("Raw Attention Scores (QK^T / sqrt(d))")

# ③ Attention重み (Softmax後)
sns.heatmap(attention_weights.detach().numpy(), annot=True, cmap="Blues", ax=axes[2],
            xticklabels=tokens, yticklabels=tokens)
axes[2].set_title("Normalized Attention Weights")

# ④ 出力ベクトル
sns.heatmap(output.detach().numpy(), annot=True, cmap="Greens", ax=axes[3],
            xticklabels=[f"d{i+1}" for i in range(d_model)],
            yticklabels=tokens)
axes[3].set_title("Output representations")

plt.tight_layout()
plt.show()
```

---

## 🧠 実行後に得られるもの

1. **左から右に向かう可視化**

   ```
   [x] → [Q,K,V演算] → [スコア] → [Softmax] → [出力]
   ```
2. **3枚目のヒートマップ（青）**

   → Transformerが「どの単語にどれだけ注目しているか」を表す。

   例：

   ```
   "Tokyo"  → 自分自身に強い注意  
   "is"     → "Tokyo"にも一部注目  
   "beautiful" → "is"を重視 など
   ```
3. **右端（出力ベクトル）**

   → Valueの加重平均結果。

   Attention重みによって再構成された新しい表現です。

---

## 📊 拡張ポイント（発展学習）

* `tokens` を `"I", "love", "transformers", "so", "much"` に増やして観察する
* `torch.manual_seed()` の値を変えて挙動比較
* `sns.heatmap(..., annot=True, fmt=".2f")` で小数点調整


![1760877122148](image/explanation/1760877122148.png)
