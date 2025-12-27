Google Colabで **EmbeddingGemma (embeddinggemma-300m)** を使い、文章内のトークン（単語の断片）同士がどのように注目し合っているか（アテンション）を可視化するコードを紹介します。

EmbeddingGemmaは、従来のGemmaと異なり**双方向アテンション（Bidirectional Attention）**を採用しているため、文の後半にある単語が前半にどう影響しているかを確認できるのが面白いポイントです。

---

## 🚀 実装の手順

可視化には、Transformersモデルのアテンション構造を直感的に表示できるライブラリ **`bertviz`** を使用します。

### 1. ライブラリのインストール

Google Colabのセルで以下を実行して、必要なパッケージをインストールします。

```python
!pip install -q transformers accelerate bertviz

```

### 2. モデルのロードと可視化コード

EmbeddingGemmaを使用するには、Hugging Faceへのログインが必要です（モデルの利用規約に同意している必要があります）。

```python
from transformers import AutoTokenizer, AutoModel
from bertviz import head_view
import torch

# 1. モデル名の指定
# 事前に Hugging Face で google/embeddinggemma-300m の利用規約に同意し、
# ログイン（huggingface-cli login）しておく必要があります。
model_id = "google/embeddinggemma-300m"

# 2. トークナイザとモデルの準備
# output_attentions=True を指定するのがポイントです
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, output_attentions=True, device_map="auto")

# 3. 可視化したいテキスト
# EmbeddingGemmaはプロンプト（"query: "など）を付けるのが推奨されています
text = "query: How do transformers pay attention to each word?"

# 4. 推論とアテンションの取得
inputs = tokenizer.encode_plus(text, return_tensors='pt').to(model.device)
outputs = model(**inputs)

# アテンション、トークン、インプットIDを取得
attention = outputs.attentions
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# 5. 可視化（BertVizのHead Viewを使用）
head_view(attention, tokens)

```

---

## 📝 コードのポイント解説

1. **`output_attentions=True`**:
通常、モデルは計算速度を優先してアテンションの重みを捨ててしまいます。このオプションを付けることで、可視化に必要な重み情報を保持して出力させます。
2. **双方向アテンションの確認**:
この可視化ツールでアテンションを動かすと、EmbeddingGemmaが**文の右側の単語から左側の単語へも強いアテンションを送っている**ことが確認できます。これはGPT（左から右のみ）との大きな違いです。
3. **トークンの分割**:
サブワード（SentencePieceなど）による分割具合も同時に確認できます。例えば「transformers」がどのように切り分けられているかも一目瞭然です。

---

**次に行うべきステップ:**
特定の層（Layer）やヘッド（Head）ごとの違いを詳しく見たい場合は、`bertviz` の **`model_view`** という関数も使えます。興味があればそちらの使い方もお伝えしましょうか？

[EmbeddingGemma Architecture and Recipe](https://www.youtube.com/watch?v=eMlx5fFNoYc)
このビデオでは、Transformerにおけるアテンションメカニズムが、どのように単語間の意味的関係を捉えるかをステップバイステップで視覚的に解説しており、EmbeddingGemmaの内部動作を理解するのに非常に役立ちます。