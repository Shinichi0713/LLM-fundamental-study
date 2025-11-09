# Modern BERTの工夫点

「 **Modern BERT** 」とは、初代 BERT（2018年）以降に登場した改良版 BERT 系モデルに共通する“最新の設計上の工夫”を指します。

研究の進展により、性能・学習効率・メモリ効率などが大幅に改善されています。

---

## 🌟 Modern BERT の主な工夫点

### 1️⃣ 相対位置埋め込み（Relative Positional Encoding）

従来のBERTは、固定の「絶対位置埋め込み（Sinusoidal PE）」を使用していました。

Modern BERTでは「 **相対位置表現（Relative Position Encoding / RoPE など）** 」が使われます。

* 目的：単語の**相対的な距離**を表現し、長文や文の再配置にも強くする
* 実装例：
  * **DeBERTa** → Relative + Disentangled PE
  * **RoFormer** → Rotary Positional Embedding（RoPE）

🧠効果：

「文の途中を入れ替えても意味が保たれる」「長文でも情報が減衰しにくい」

---

### 2️⃣ disentangled attention（分離アテンション）

（例： **DeBERTa** ）

* 従来：トークン埋め込み + 位置埋め込みを**単純加算**
* 改良：内容(content) と位置(position) の情報を**別々に扱う**
  ```
  Attention = Q_content × K_content^T + Q_content × K_position^T + Q_position × K_content^T
  ```
* 意味：単語の意味と位置関係を分離して処理することで、より豊かな文脈理解が可能。

🧠効果：

BERT-baseより小さいモデルでも同等性能を達成。

---

### 3️⃣ Pre-LayerNorm 構造

（例： **RoBERTa, DeBERTaV3, ModernBERT** ）

* 従来：Post-LayerNorm（Transformer block の出力後にLN）
* 改良：Pre-LayerNorm（ブロックの入力前にLN）に変更

🧠効果：

* 学習が安定する
* 高学習率でも発散しにくい
* より深い層まで安定して学習可能

---

### 4️⃣ 高効率化（パラメータシェア・軽量Attention）

（例： **ALBERT, MobileBERT, DistilBERT** ）

| 手法                 | 主な工夫                                | 効果                           |
| -------------------- | --------------------------------------- | ------------------------------ |
| **ALBERT**     | 層ごとの重み共有 + Factorized Embedding | パラメータ数を1/10以下に削減   |
| **MobileBERT** | Bottleneck構造 + Inverted Residual      | モバイル向け高効率             |
| **DistilBERT** | 知識蒸留                                | モデルを半分以下のサイズに圧縮 |

---

### 5️⃣ 学習データ・目的の改善

* **RoBERTa** : NSP（Next Sentence Prediction）を削除し、データ量を10倍に。
* **DeBERTaV3** : Masked LM ではなく、**MLM + replaced token detection (RTD)** の組み合わせを使用。
* **ModernBERT (Google 2024)** :
* コード + Web + 書籍など多様なコーパスで訓練
* 高速学習に適した **FlashAttention / XPos / RMSNorm** を採用

---

### 6️⃣ 高速化テクニック

* **FlashAttention** : GPUでアテンションを直接ストリーム計算し、高速かつ省メモリ化
* **RMSNorm** : LayerNormの簡略版（平方平均を使用）で軽量化
* **XPos** : 長文対応の拡張RoPE（相対位置のスケーリングを調整）

---

## 🧩 まとめ

工夫点は位置表現、RMSNormによる正規化、Disentangled Attention、FlashAttention

| 改良ポイント     | 技術                     | 効果                             |
| ---------------- | ------------------------ | -------------------------------- |
| 位置表現         | RoPE / XPos / 相対PE     | 長文に強く、文構造を理解しやすい |
| 正規化           | Pre-LN / RMSNorm         | 学習安定性・高速化               |
| アテンション構造 | Disentangled / Efficient | 精度向上・軽量化                 |
| 学習方式         | RoBERTa-style / RTD      | 汎化性能向上                     |
| 実装最適化       | FlashAttention           | GPUメモリ削減・高速化            |


wikipediaでMLM

![1762664182175](image/explanation/1762664182175.png)
