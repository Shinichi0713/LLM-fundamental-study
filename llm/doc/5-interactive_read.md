
BERTが文章を**双方向（bidirectional）に理解できる理由**は、
「Transformerだから」ではありません。
**学習タスク設計とAttentionの使い方を意図的に変えている**点が本質です。

結論を先に言うと、

> **BERTは「未来トークンを見ながら学習する」ことを、Masked Language Model（MLM）という実装で実現している**

これが双方向理解の正体です。

以下、実装レベルで分解します。

---

## 1. 双方向理解とは何を意味するのか

例文：

```
I went to the bank to deposit money.
```

* 左文脈のみ → 「川岸」か「銀行」か曖昧
* 右文脈を見る → 「銀行」と確定

👉 **単語が左右両方の文脈に依存して意味を持つ**

---

## 2. 従来モデル（GPT等）との決定的違い

### GPT（Decoder-only）

* **因果マスク（Causal Mask）**
* 未来トークンを見られない

```text
[t1] → [t2] → [t3] → ...
```

### BERT（Encoder-only）

* **マスクなしSelf-Attention**
* 全トークンが相互に参照可能

```text
[t1] ↔ [t2] ↔ [t3] ↔ ...
```

しかし
👉 **これだけでは不十分**

---

## 3. 核心①：Self-Attentionでの「双方向参照」

### 実装上の違い

#### GPT（因果マスクあり）

```python
attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```

#### BERT（マスクなし）

```python
attn_mask = None  # padding maskのみ
```

### 結果

* 各トークンは

  * 左も
  * 右も
  * 同時に参照可能

👉 **構造的双方向性**

---

## 4. 核心②：Masked Language Modeling（MLM）

### なぜMLMが必要か

もし単純に

```
入力: I went to the bank to deposit money
出力: 次の単語を予測
```

とすると
→ 未来トークンを見てしまうため **チート**

---

### MLMの実装

#### ステップ①：ランダムマスク

```text
I went to the [MASK] to deposit money
```

#### ステップ②：マスクされた単語を予測

```python
loss = CrossEntropy(
    prediction_at_mask,
    true_token
)
```

### 重要ポイント

* 予測対象トークンは

  * **左右の文脈を同時に利用**
* 未来情報のリークは発生しない

👉 **双方向文脈利用が成立**

---

## 5. MLMを実装レベルで見る

### マスク戦略（BERT原論文）

* 15%のトークンを選択

  * 80% → `[MASK]`
  * 10% → ランダムトークン
  * 10% → 元のまま

```python
if rand < 0.8:
    token = [MASK]
elif rand < 0.9:
    token = random_token()
else:
    token = original_token()
```

#### なぜこの工夫？

* `[MASK]` が推論時に存在しないため
* **事前学習と推論のギャップを減らす**

---

## 6. 核心③：学習時に「全位置を同時に最適化」

### GPT

* 左→右の逐次予測
* 一度に1位置

### BERT

* 文中の複数 `[MASK]` を同時予測

```python
loss = sum(loss_i for i in masked_positions)
```

👉 **文全体の意味整合性を強制**

---

## 7. 実装全体の流れ（擬似コード）

```python
tokens = tokenize(sentence)

masked_tokens, labels = apply_mlm_mask(tokens)

hidden = bert_encoder(masked_tokens)

loss = MLMHead(hidden[masked_positions], labels)
```

ここで重要なのは

* `bert_encoder` 内に **因果マスクが存在しない**
* `labels` が **一部位置のみ**

---

## 8. なぜ「双方向理解」が可能になるのか（要点）

| 要素             | 効果      |
| -------------- | ------- |
| マスクなしAttention | 左右参照可能  |
| MLM            | 未来リーク防止 |
| 同時最適化          | 文全体理解   |
| ノイズマスク         | 推論耐性    |

👉 **構造 × 学習目標の組み合わせ**

---

## 9. よくある誤解

### ❌「BERTは双方向Attentionだから賢い」

→ Attentionだけならチート

### ❌「MLMだけで双方向になる」

→ Attentionが因果マスクなしである必要

### ✔ 正解

> **MLM + 非因果Self-Attention**

---

## 10. 一言でまとめると

> **BERTは「未来を見ながら、未来を当てない」ように学習することで、双方向理解を実現している**

これは

* GPTのような生成モデル
* 単純なAutoencoder

では成立しない、**極めて設計的な工夫**です。

---

次に深掘りするなら

* NSPの役割と実装是非
* RoBERTaでNSPが削除された理由
* ELECTRAがMLMをどう置き換えたか
* Decoderに双方向性を擬似導入する方法

まで踏み込むと、設計思想が完全に腹落ちします。


