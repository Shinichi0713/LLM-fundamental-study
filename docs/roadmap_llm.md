## Roadmap of LLM

LLM（大規模言語モデル）の基礎を理解するために、体系的に押さえるべき項目を整理しました。**数理 → モデル構造 → 学習技術 → 応用**という流れで理解するとわかりやすいです。

---

## ✅ **LLMの基礎として理解すべき項目一覧**

### **1. 数理・基礎概念**

| 項目           | 内容                                       |
| -------------- | ------------------------------------------ |
| 確率・統計     | 期待値、確率分布、尤度、クロスエントロピー |
| 線形代数       | ベクトル、行列、内積、固有値、SVD          |
| 微積分・最適化 | 勾配、勾配降下法、損失関数                 |
| 情報理論       | エントロピー、KLダイバージェンス           |
| 分布表現       | Word2Vec、埋め込み（embedding）概念        |

---

### **2. ニューラルネットワーク基礎**

| 項目           | 内容                           |
| -------------- | ------------------------------ |
| パーセプトロン | ニューラルネットの基礎         |
| 活性化関数     | ReLU、Softmax                  |
| 損失関数       | Cross-Entropy、MSE             |
| トレーニング   | ミニバッチ学習、正則化、過学習 |

---

### **3. RNN → Transformer という歴史と進化**

| 世代        | 代表例              | ポイント              |
| ----------- | ------------------- | --------------------- |
| RNN         | SimpleRNN           | 長期依存性の課題      |
| LSTM / GRU  | LSTM, GRU           | 長い文脈に対応        |
| Attention   | Seq2Seq + Attention | Attentionで依存性改善 |
| Transformer | BERT, GPT           | 完全Attention構造     |

→ **LLM理解の中心はTransformer**

---

### **4. Transformerの中核概念**

| パーツ                     | 内容                          |
| -------------------------- | ----------------------------- |
| Self-Attention             | 重要語の重みづけ。Q/K/V       |
| Multi-Head Attention       | 複数の視点で情報を取得        |
| Positional Encoding        | 位置情報（Sinusoidal / RoPE） |
| FFN (Feed Forward Network) | 非線形変換                    |
| Layer Norm / Residual      | 学習安定化                    |
| Tokenization               | BPE, SentencePieceなど        |

---

### **5. 言語モデルの学習と推論**

| 項目         | 内容                            |
| ------------ | ------------------------------- |
| Pre-training | 大規模データで事前学習          |
| Fine-tuning  | 目的に特化して調整              |
| RLHF         | 人間フィードバック学習          |
| 推論         | greedy / sampling / beam search |
| KV Cache     | 推論高速化の要                  |

---

### **6. 実用アーキテクチャと代表モデル**

| 種類              | 例                  |
| ----------------- | ------------------- |
| Encoder型         | BERT                |
| Decoder型         | GPT                 |
| Encoder-Decoder型 | T5、FLAN            |
| ALiBi/RoPEなどPE  | 位置表現の工夫      |
| Sparse Attention  | Longformer、Mistral |

---

### **7. LLMに付随する技術領域**

| 分野               | 内容                |
| ------------------ | ------------------- |
| Retrieval (RAG)    | 検索 + LLM          |
| Prompt Engineering | 指示設計            |
| Agent              | 自律タスク遂行      |
| Memory機構         | 外部メモリ利用      |
| Tool Use           | API, DB, コード実行 |

---

### **8. セキュリティ・倫理**

| 項目           | 内容             |
| -------------- | ---------------- |
| Hallucination  | 幻覚対策         |
| Jailbreak      | ルール回避攻撃   |
| データバイアス | 公平性と説明責任 |

---

## 🎯 **理解の優先順位（ロードマップ）**

### ステップ1：入門

* ベクトルと行列
* Attentionの仕組み
* TokenとEmbedding

### ステップ2：構造理解

* Transformerのブロック構成
* Q/K/V・ポジションエンコーディング
* 誤差逆伝播・最適化

### ステップ3：実践

* BERTとGPTの違い
* RAGの動作
* 推論テクニック（sampling, temperature）

---

## 📘 **おすすめ深堀り順**

1. Self-Attentionを式レベルで理解
2. Positional Encoding (Sinusoidal & RoPE)
3. KV Cacheと高速化技術
4. RLHFとAlignment
5. Multi-modal拡張（VLM）

---

## ✨ まとめ

LLM理解の根幹は **Transformer + Attention + Embedding**

そこに **最適化/推論/アライメント/検索連携** が積み重なるイメージです。
