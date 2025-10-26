いいですね — アテンションの振る舞いを**見て、触って、実験する**のは LLM（トランスフォーマ）を理解する最も良い方法の一つです。以下は **実践的で再現可能な手順＋コード＋解説** を一式でまとめたものです。ローカルでも Colab でも動くように書いてあります。まずは小さめのモデルで始め、慣れたら大きいモデルで試してください。

---

# 準備（必要ライブラリ）

ローカルや Colab の Python 環境で以下をインストールしてください。

```bash
pip install transformers torch matplotlib seaborn tokenizers
# 対話的に可視化したければ（任意）
pip install bertviz
```

---

# 1) やってみる — BERT/GPT 系で attention を抽出してヒートマップ表示

以下は Hugging Face `transformers` を使った最小限の例です（`bert-base-uncased` を例に）。`output_attentions=True` で attention を取得し、各 layer・head の attention 行列をプロットします。

```python
# attention_viz.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

def plot_attention_matrix(attn, tokens, title="", vmax=None, save=None):
    """
    attn: (seq_len, seq_len) numpy array
    tokens: list of token strings (len seq_len)
    """
    plt.figure(figsize=(6,6))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
                square=True, cbar=True, vmax=vmax)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
    plt.show()

def visualize_attention(model_name="bert-base-uncased", text="The quick brown fox jumps over the lazy dog"):
    # Load
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]  # (1, seq_len)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Forward (get attentions)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.attentions: tuple of length num_layers, each (batch, num_heads, seq_len, seq_len)
    attentions = outputs.attentions

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    seq_len = attentions[0].shape[-1]

    print(f"Model: {model_name}, layers: {num_layers}, heads: {num_heads}, seq_len: {seq_len}")
    print("Tokens:", tokens)

    # Show a few heads (first layer, each head)
    # You can iterate layers/heads as you like
    for layer in range(min(4, num_layers)):  # show first 4 layers as demo
        for head in range(num_heads):
            mat = attentions[layer][0, head].cpu().numpy()  # (seq_len, seq_len)
            # optional normalization: already softmaxed in model -> sums to 1 across last dim
            plot_attention_matrix(mat, tokens, title=f"Layer {layer+1} Head {head+1}")

if __name__ == "__main__":
    visualize_attention()
```

**実行方法**

```bash
python attention_viz.py
```

（Colab ならセル実行でプロットが出ます）

---

# 2) 何を見れば良いか（解釈のヒント）

* 行列の**行 i**は「クエリがトークン i のとき、どのキーに重みを払っているか（どのトークンに注目しているか）」を示します。
* 主なパターン：
  * **対角優勢（diagonal）** ：トークンが主に自分自身/近傍に注目 → 局所的処理（言語モデルでよく見る）
  * **CLS / [EOS] に集中** ：要約・文全体の文脈取得（分類タスクでの特徴）
  * **句構造に沿った注目** ：動詞が目的語を強く参照、形容詞が被修飾語を参照、など（意味的関係）
  * **ヘッド間の分業** ：あるヘッドは局所、別ヘッドは長距離関係を拾うことがある

---

# 3) 見やすくまとめて表示（グリッドで各レイヤxヘッドを一気に表示）

大量の図を手で見るのは大変なので、1つのレイヤー内の全ヘッドをグリッド化するユーティリティも用意します。

```python
def plot_layer_heads(attentions, tokens, layer, vmax=None, cols=8, save=None):
    """
    attentions: outputs.attentions (tuple)
    layer: zero-based layer index
    """
    mat_layer = attentions[layer][0].cpu().numpy()  # (num_heads, seq_len, seq_len)
    num_heads = mat_layer.shape[0]
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    axes = axes.flatten()
    for h in range(len(axes)):
        ax = axes[h]
        if h < num_heads:
            sns.heatmap(mat_layer[h], ax=ax, cbar=False, vmax=vmax)
            ax.set_title(f"H{h}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    fig.suptitle(f"Layer {layer+1} Heads")
    if save:
        fig.savefig(save, dpi=200)
    plt.show()
```

呼び出し例：

```python
# after outputs = model(...)
plot_layer_heads(outputs.attentions, tokens, layer=0, cols=8)
```

---

# 4) よりインタラクティブに見るなら：`bertviz`

`bertviz` はトークンペアの注目をインタラクティブに探索できます（Jupyter/Colab向け）。

インストールしたら以下のパターンで簡単に可視化できます（公式 README を参照）：

```python
from bertviz import head_view
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer.encode_plus(text, return_tensors='pt')
outputs = model(**inputs)
attention = [att.cpu().numpy() for att in outputs.attentions]
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
head_view(attention, tokens)
```

（Colab や Jupyter ノートで動かすとブラウザ上でインタラクティブに探せます）

---

# 5) 実験アイデア（学習のためのハンズオン）

1. **部分語（subword）を含む単語で挙動を見る** ：BPE に分かれる単語で attention がどう分配されるか確認。
2. **文をわざと長くする** ：長距離依存をどのレイヤで拾うか。
3. **トークンを入れ替える（語順を変える）** ：一部ヘッドが語順に敏感かを確認。
4. **マスクしたときの挙動** （マスク言語モデル）と、**生成モデルの自己注意**での違いを比較。
5. **平均化 vs 特定ヘッド** ：全ヘッドを平均化して出る注意と、特定ヘッドの注意を比較して「どのヘッドが重要か」を評価。
6. **Attention rollout / attribution** ：層をまたいだ集約（入力→出力にどのトークンが寄与したか）を計算して可視化する（下記に簡単な実装案）。

---

# 6) Attention Rollout（簡易版）

層をまたいで注意を多段的に合成し、入力トークンが最終的にどれだけ影響を与えているかを可視化する手法（参考実装を簡略化）：

```python
def attention_rollout(attentions, start_layer=0, discard_ratio=0.0):
    # attentions: tuple of (batch, heads, seq, seq). We'll average over heads.
    num_layers = len(attentions)
    result = np.eye(attentions[0].shape[-1])
    for i in range(start_layer, num_layers):
        attn = attentions[i][0].mean(axis=0)  # avg over heads -> (seq, seq)
        # Optionally zero small weights
        if discard_ratio:
            flat = attn.flatten()
            cutoff = np.quantile(flat, discard_ratio)
            attn = np.where(attn < cutoff, 0, attn)
        attn = attn / attn.sum(axis=-1, keepdims=True)  # re-normalize
        result = attn @ result
    return result  # (seq, seq)

# usage:
roll = attention_rollout([a.cpu().numpy() for a in outputs.attentions])
plot_attention_matrix(roll, tokens, title="Attention Rollout (avg heads)")
```

---

# 7) 注意点・落とし穴（Interpretability）

* **Attention ≠ Explanation** ：Attention が高いから必ず重要という単純な解釈は危険です。研究では attention の重要度は場合により限定的とされています（ただし直感的な手掛かりにはなる）。
* **Softmax の性質** ：注意重みは正規化されるので、1箇所に極端に集中するか、分散するかで解釈が変わります。
* **トークン化の影響** ：サブワード分割が視覚的に“分裂”して見えるため、単語単位でまとめて見るとわかりやすい。
* **モデルごとの差** ：BERT（マスク）と GPT（自己回帰）では attention の役割が異なる（BERT は双方向文脈、GPT は左から右への因果注意）。

---

# 8) 実践ワークフロー（おすすめ）

1. Colab / ローカルで上のスクリプトを動かして、短文で全ヘッド・全レイヤを確認。
2. `bertviz` でインタラクティブに探索。
3. 興味が湧いたら attention-rollout、ヘッド除去実験（特定ヘッドをゼロにして下流タスク性能を見る）を試す。
4. 最後に「なぜそのヘッドが重要か？」を言語学的に解釈してみる（構文、修飾関係など）。



# BertViz

BERT の **Attention（アテンション）を可視化するための定番ツールが [BertViz](https://github.com/jessevig/bertviz)** です。

以下に、**Google Colab でそのまま動かせるセル例**をセットで示します👇

---

## 🎯 目的

BERT モデルのアテンションを可視化して、「どの単語がどの単語に注目しているか」を直感的に学ぶ。

---

## ✅ 手順構成

1️⃣ 依存パッケージのインストール

2️⃣ モデルとトークナイザーの読み込み

3️⃣ サンプル文の設定

4️⃣ BertViz の可視化

---

### 🧩 Colab セル例（コピーして順に実行できます）

#### ✅ セル1：セットアップ

```python
!pip install transformers==4.44.0 bertviz==1.4.0 torch
```

#### ✅ セル2：ライブラリ読み込み

```python
from transformers import BertTokenizer, BertModel
from bertviz import head_view, model_view
import torch
```

#### ✅ セル3：モデルとトークナイザーの読み込み

```python
# 英語モデルを例に（日本語版に変えることも可能）
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)
```

#### ✅ セル4：入力文の設定

```python
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"

inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
input_ids = inputs['input_ids']
attention = model(**inputs).attentions
```

#### ✅ セル5：BertVizによるHead View（単語ごとの注目を可視化）

```python
# 各アテンションヘッドがどの単語に注目しているかを可視化
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
head_view(attention, tokens)
```

#### ✅ セル6：BertVizによるModel View（層全体の関係を俯瞰）

```python
# 各層・ヘッド全体のアテンション構造をインタラクティブに表示
model_view(attention, tokens)
```

---

## 💡 補足

* Colab で実行後、インタラクティブなアテンションマップが開きます。

  （`head_view` は単語間の注目線を表示、`model_view` は層とヘッドの全体構造）
* 日本語モデルを使いたい場合：

  ```python
  model_name = "cl-tohoku/bert-base-japanese-v2"
  ```

---

## 🚀 応用例

* 入力文を変えて比較（例：「犬が走る」「猫が歩く」など）
* 各層・ヘッドがどの単語をどの程度注目しているかを観察し、

  BERTの内部構造を直感的に理解できます。



# LLMアテンション理解のテーマ

では、次の中からどの方向で掘り下げたいか教えてください。

どれも **「LLMのアテンションを理解する」ための実験テーマ**です👇

---

### 🧭 選択メニュー

#### 🟢 **① 日本語BERTでアテンションを可視化**

* モデル：`cl-tohoku/bert-base-japanese-v2`
* 文例：「犬が走る」と「猫が歩く」など
* Colabで日本語のAttention線を可視化

  🧩 → *英語例とほぼ同じ手順で動作*

---

#### 🔵 **② Attentionの重みを数値で分析**

* アテンションマップを**数値として抽出・描画**
* 「どの単語がどれに最も注目しているか」を表にして確認

  📊 → *head_viewの裏側を数値で理解する*

---

#### 🟣 **③ GPT系モデル（LLM）のアテンションを可視化**

* OpenAI系モデルやLLaMAなど、**Decoder-only型**の可視化
* Hugging Face経由でアテンションを取得し、

  *“どの過去トークンを参照しているか”* を見る

  🤖 → *LLMの生成挙動の理解に直結*

---

#### 🟠 **④ BertViz以外の可視化法**

* **Attention Rollout** （層をまたいだアテンション伝播）
* **TransformerLens** （OpenAI系モデルの内部構造解析）
* *研究寄りで、LLMの思考経路を追跡する方法*

---

どのテーマから進めてみたいですか？

（番号でも、「日本語BERT」などの言葉でもOKです）
