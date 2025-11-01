LLM（大規模言語モデル）で用いられる**Positional Encoding（位置表現）**には、Transformer登場以降さまざまな手法が提案されています。

以下に、主な方式を体系的にまとめます。

---

## ✅ Positional Encodingの主要カテゴリー

| 方式                  | 特徴                        | 代表モデル                     |
| --------------------- | --------------------------- | ------------------------------ |
| 固定（Fixed）         | 学習しない、数式ベース      | 初期Transformer、GPT-2など     |
| 学習型（Learned）     | 位置ベクトルを学習          | BERT、T5初期                   |
| 相対位置（Relative）  | 相対距離を学習・表現        | Transformer-XL、DeBERTa、T5    |
| 回転型（Rotary）      | 回転変換で相対位置を表現    | GPT-NeoX、LLaMA、Mistral、Qwen |
| 注意機構応用（ALiBi） | 距離に応じてAttentionを減衰 | BLOOM、DS-Trlモデルなど        |

---

## ✅ 各方式の具体例

### 1) 固定（Sinusoidal Positional Encoding）

Transformer論文（Vaswani et al. 2017）の方式

* 周期の異なる正弦波・余弦波を組合せ位置を表現
* **長いシーケンスにも拡張可能だが、性能は限定的**

PE(pos,2i)=sin(pos/10000(2i/d))PE(pos,2i+1)=cos(pos/10000(2i/d))PE(pos,2i)=sin(pos/10000^(2i/d))
PE(pos,2i+1)=cos(pos/10000^(2i/d))

**採用例**

初期Transformer、GPT-2

---

### 2) 学習型 Positional Embedding（Learned）

* 位置ごとのembeddingをパラメータとして学習
* 学習した長さ以上は基本伸ばせない

**採用例**

BERT、T5初期など

---

### 3) 相対位置情報（Relative Position Encoding）

#### (a) Shaw et al. (2018)

* query-keyスコアに相対距離の埋め込みを加算

#### (b) Transformer-XL

* 長文処理に強い
* メモリキャッシュとともに利用

#### (c) T5のRelative Bias

* attentionに**距離に応じたバイアス**を加算

Attention=QKT/sqrt(d)+bias(relativedistance)Attention = QK^T / sqrt(d) + bias(relative_distance)

**採用例**

Transformer-XL、T5、DeBERTa

DeBERTaは"disentangled attention"でさらに改良

---

### 4) RoPE（Rotary Positional Encoding）

* ベクトルに**回転変換**を付与して相対位置表現
* **長い文脈性能が良く、LLMで主流**
* NTKスケーリング・YaRNで長文対応拡張可能

**採用例**

GPT-NeoX、LLaMA、Mistral、Qwen、Phiなど

---

### 5) ALiBi (Attention Linear Bias)

* 距離に応じ注意重みを**線形にペナルティ**
* 位置ベクトルを持たない→**メモリ軽い**
* 推論時長さ拡張が容易

**採用例**

BLOOM、部分的に最新研究で併用されることも

---

### 6) その他

| 方式                               | 説明                                 |
| ---------------------------------- | ------------------------------------ |
| Axial Positional Embedding         | 2D構造に分解（画像/ViTで多い）       |
| Perceiver IO positional scheme     | 高次元構造対応                       |
| Hyena / Mamba 系 State-Space Model | 位置情報を暗黙にモデリング（PE不要） |

最近は**SSM（State-Space Models）**系が台頭しており、

直接Positional Encodingを使わない方向にも発展中（Mambaなど）。

---

## ✅ まとめ

| カテゴリ      | 例                 | 長文性能 | Notes              |
| ------------- | ------------------ | -------- | ------------------ |
| 固定          | Sinusoidal         | △       | 古典方式           |
| 学習型        | Learned embedding  | ×       | 固定長             |
| 相対位置      | T5, Transformer-XL | ◎       | 高精度だが計算重い |
| 回転          | RoPE               | ◎◎     | 現代LLM標準        |
| Attention補正 | ALiBi              | ◎       | 軽量で長文に強い   |
| SSM           | Mamba              | ◎       | PE不要新世代       |

---

## ✅ 現代の主流

> **RoPE（＋NTK/YaRNスケーリング）**

長文に強く、計算効率も良いためGPT-4やLLaMAシリーズでも採用。

---

必要なら、

* 各手法の数学的導出
* PyTorchでの実装例
* 比較評価グラフ
* 位置埋め込み視覚化

もお出しできます。どれを見たいですか？
