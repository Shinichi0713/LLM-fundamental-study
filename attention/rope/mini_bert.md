
近年のLLMはサイズが大きくなりすぎて要素レベルの改善がどの程度モデルの性能に寄与するかがみにくい点が悩みです。
そこで、LLMの要素単位の改善での効果を簡単に計測できるモデルがないかを情報探してみました。
結果出てきたのが2018年にGoogleが発表したBERTを更に簡略化した軽量モデルであるminiBERTなるものがありました。


本日テーマ：

>miniBERTがどのようなものか、miniBERTを簡単に触って説明してみようと思います。

## miniBERTとは

「miniBERT」という名前は、**BERTを大幅に簡略化した軽量モデル**を指す総称として複数の文脈で使われています。代表的なものを挙げると、以下の2つがあります。


### 1. LIUM（ル・マン大学）の「MiniBERT: a simple and explainable BERT model」

- **位置づけ**  
  BERTの「極端な簡略版（extreme simplification of the BERT model）」として設計されたモデルです[LIUM](https://lium.univ-lemans.fr/en/minibert-a-simple-and-explainable-bert-model)。

- **目的**  
  SNCF（フランス国鉄）向けのプロジェクト「PolysEmY」で、**技術文書中の多義語（polysemy）を扱うこと**と、**モデルをシンプルかつ説明可能にすること**を主目的としています[LIUM](https://lium.univ-lemans.fr/en/minibert-a-simple-and-explainable-bert-model)。

- **BERTとの違い**
  - BERTやGPT-3のような「巨大な汎用モデル」ではなく、**特定の技術コーパスとタスク（多義語の捕捉）に特化**しています。
  - パラメータ数を大幅に削減し、**シンプルさと説明可能性を優先**しています。
  - アテンション機構を使って「どの単語ペアが意味的に影響し合っているか」を明示的に扱う設計になっています。

- **特徴**
  - 一般のBERTに比べて**はるかに小規模**で、計算コストが低い。
  - アテンションの重みを見ることで「なぜその予測になったか」を**説明しやすい**ことが強調されています。

### 2. GitHub 上の「MiniBERT: Lightweight Emotion Classification Model」

- **位置づけ**  
  「compact, BERT-inspired language model for emotion classification」と説明される、**BERT風の軽量トランスフォーマー**です[GitHub - MiniBERT](https://github.com/Habiba-Abdelrehim/MiniBERT)。

- **目的**  
  **感情分類（emotion classification）**に特化したモデルで、GoEmotionsデータセットでファインチューニングされています[GitHub - MiniBERT](https://github.com/Habiba-Abdelrehim/MiniBERT)。

- **規模**
  - パラメータ数は約 **1,800万（18M）** と、標準的なBERT-base（約1.1億）より**大幅に少ない**軽量モデルです[GitHub - MiniBERT](https://github.com/Habiba-Abdelrehim/MiniBERT)。

- **BERTとの関係**
  - BERTのアーキテクチャを参考にしつつ、**独自に設計されたコンパクトなTransformer**であり、BERTの直接的な圧縮版というより「BERT風の軽量モデル」です。
  - 速度・効率を重視し、クラス不均衡などの課題を抱えつつも、標準BERTとの比較・ベンチマークを目指しています[GitHub - MiniBERT](https://github.com/Habiba-Abdelrehim/MiniBERT)。


## QucikStart
とりあえずパッと使ってみようと思います。
Hugging Faceに**軽量なBERT系モデル**として公開されている `boltuix/bert-mini` を「miniBERT」の代わりに使うのが実用的です。  
このモデルは約800万パラメータ・約15MBと非常に軽量で、MLM（Masked Language Modeling）がそのまま使えます[Hugging Face - boltuix/bert-mini](https://huggingface.co/boltuix/bert-mini)。


### 必要なライブラリ

```bash
pip install transformers torch
```

- `transformers` はHugging Faceのモデル・トークナイザを扱うライブラリです。
- `torch` はPyTorch本体です（GPUを使う場合はCUDA対応版を入れてください）。

### `AutoModelForMaskedLM` + `AutoTokenizer`

では動かしてみましょう。
CPUしかないPCで十分に動作します。

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# モデルとトークナイザのロード
model_name = "boltuix/bert-mini"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# 入力テキスト（[MASK] を含む）
text = "The lecture was held in the [MASK] hall."

# トークナイズ
inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

# 推論（勾配計算なし）
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# [MASK] 位置のロジットから上位5単語を取得
mask_logits = logits[0, mask_token_index, :]
top_5 = torch.topk(mask_logits, 5, dim=1)

# トークンIDを単語に戻して表示
for i, (token_id, score) in enumerate(zip(top_5.indices[0], top_5.values[0])):
    token = tokenizer.decode(token_id)
    print(f"{i+1}: {token} (score={score:.4f})")
```

**出力例**

上記のコードをコピー＆ペースの上実行下さい。
以下のようにHuggingFaceよりモデルをロードして出力が確認出来ます。

![1784447825138](image/mini_bert/1784447825138.png)

カーネギーホールが初めに出てきました。
こういうことが起こるのはデータのバイアスがあるからです。
兎にも角にもMLMは可能そうです。

```
1: carnegie (score=8.6277)
2: city (score=8.0210)
3: memorial (score=7.8888)
4: national (score=7.8867)
5: lecture (score=7.6196)
```

- `AutoModelForMaskedLM` を使うと、ロジットやスコアを直接取得できます。
- バッチ処理や独自のマスク戦略を組み込む場合にも拡張しやすいです。


### 日本語で試したい場合

`boltuix/bert-mini` は英語用ですが、日本語の軽量BERTモデル（例: `cl-tohoku/bert-base-japanese` やその小型版）を使えば、同じコードで日本語のMLMも可能です。  
その場合は `model_name` を日本語モデルに置き換えてください。

```python
# 例: 日本語BERT-base（MLM対応モデル）
model_name = "cl-tohoku/bert-base-japanese"
text = "今日は[MASK]に行きました。"
```

## モデルのコード作成

後はモデルの構造を解析しにいき、実態のコードを掴みに行きます。
上記を実行の後、以下を実行下さい。

```python
print(model)
```

するとモデルのアーキテクチャが得られます。
何のことはない、通常のBERTでした。


```
BertForMaskedLM(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 128, padding_idx=0)
      (position_embeddings): Embedding(512, 128)
      (token_type_embeddings): Embedding(2, 128)
      (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-3): 4 x BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=128, out_features=128, bias=True)
              (key): Linear(in_features=128, out_features=128, bias=True)
              (value): Linear(in_features=128, out_features=128, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=128, out_features=128, bias=True)
              (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=128, out_features=512, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=512, out_features=128, bias=True)
            (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (cls): BertOnlyMLMHead(
    (predictions): BertLMPredictionHead(
      (transform): BertPredictionHeadTransform(
        (dense): Linear(in_features=128, out_features=128, bias=True)
        (transform_act_fn): GELUActivation()
        (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
      )
      (decoder): Linear(in_features=128, out_features=30522, bias=True)
    )
  )
)
```

モデルのコードは以下レポジトリを参考下さい。
https://github.com/Shinichi0713/LLM-fundamental-study/tree/main/attention/rope/src

パラメータだけ合わしてHuggingFaceのパラメータをロードしてみます。
今回の小さいBERTのパラメータは以下通りでした。

```python
def load_hf_model_and_copy_weights(model_name="boltuix/bert-mini"):
    # Hugging Faceのモデルとトークナイザをロード
    hf_model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 自作モデルを指定されたconfigで初期化（すべてハードコーディング）
    custom_model = CustomBertForMaskedLM(
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=512,
        type_vocab_size=2,
        dropout_prob=0.1,  # hidden_dropout_prob=0.1 に対応
    )

    # 重みをコピー（名前が一致するもののみ）
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = custom_model.state_dict()

    # 名前マッピング（必要に応じて調整）
    name_map = {}
    for name, param in hf_state_dict.items():
        if name in custom_state_dict:
            name_map[name] = name

    # コピー実行
    for hf_name, custom_name in name_map.items():
        if hf_name in hf_state_dict and custom_name in custom_state_dict:
            if hf_state_dict[hf_name].shape == custom_state_dict[custom_name].shape:
                custom_state_dict[custom_name].copy_(hf_state_dict[hf_name])
            else:
                print(f"[WARN] Shape mismatch: {hf_name} {hf_state_dict[hf_name].shape} vs {custom_name} {custom_state_dict[custom_name].shape}")

    custom_model.load_state_dict(custom_state_dict, strict=False)  # strict=Falseで一部だけコピー
    return custom_model, tokenizer
```

ロードして実行すると以下のような結果が得られます。

![1784448543051](image/mini_bert/1784448543051.png)

先程と結果変わりありません。
ということで内部構造のベースはご紹介したコードで、何か検証するときはHuggingFaceからネットワークパラメータをロードして使うことで比較できるようになりました。

## 速度比較

同じ問題を100題、BERTとminiBERTに解かせてどの程度速度に差がつくかを確認してみます。

結果は次の通りでした。
ざっと速度は10倍miniBERTが早いという結果でした。

```
Warming up...
Measuring inference speed (100 runs)...
Model: bert-base-uncased
  Total time for 100 runs: 12.4765 s
  Average time per run: 0.124765 s
  Runs per second: 8.02

=== Comparison ===
miniBERT avg time: 0.012977 s
BERT-base avg time: 0.124765 s
Speedup (BERT-base / miniBERT): 9.61x
```

## 総括

ということで今回はLLMの要素技術を検証するためのベースとなるモデルの導入を行いました。
このモデルの良い点は
- CPUでも軽快に動作する
- ライセンスフリー
- ベースはTransformerなので、発見した技術を適用できる先が沢山存在する

ということです。
是非ご参考下さい。


最後に本のご紹介です。
著者はLLMからニューラルネットワークに本格的に入っていきました。
当初の勉強はLLMの基礎というよりいきなりTransformerの教科書から入っていきました。
また、昨年読んで基礎から最近のLLMまで網羅してある本も面白かったです。

以下の２冊が該当する教科書です。上が著者が初めて手にした教科書、下が昨年面白かった本です。
LLMの勉強を始めようという方、知識のアップデートを図ろうという方、是非ご覧ください。



<div class="shop-card">
    <div class="shop-card-image">
        <img src="https://m.media-amazon.com/images/I/71oOctdyAHS._SL1290_.jpg" alt="商品画像">
    </div>
    <div class="shop-card-content">
        <div class="shop-card-title">BERTによる自然言語処理入門: Transformersを使った実践プログラミング</div>
        <div class="shop-card-description">自然言語処理の標準モデル、BERTを使いこなせるようになる!
BERTはGoogleが2018年末に発表した自然言語処理モデルです。「文脈」を考慮した処理が特徴的であり、言語理解を評価する11個のタスクについて最高精度を達成し、今や標準的なモデルとしての地位を確立しています。
</div>
        <div class="shop-card-link">
            <a href="https://amzn.to/4b4t43U" target="_blank" rel="noopener">Amazonで詳細を見る</a>
        </div>
    </div>
</div>


<div class="shop-card">
    <div class="shop-card-image">
        <img src="https://m.media-amazon.com/images/I/91D8pDhoI5L._SL1500_.jpg" alt="商品画像">
    </div>
    <div class="shop-card-content">
        <div class="shop-card-title">大規模言語モデル入門</div>
        <div class="shop-card-description">ChatGPTに代表される大規模言語モデルが自然言語処理の幅広いタスクで高い性能を獲得し、大きな話題となっています。大規模言語モデルは、大規模なテキストデータで訓練された大規模なパラメータで構成されるニューラルネットワークです。2020年以降、自然言語処理や機械学習の知見をもとに、パラメータ数とテキストデータの拡大により、性能が飛躍的に向上しました。
Hugging Face社の""transformers""というPythonライブラリを利用することで、プログラミングの経験があれば、モデルの操作やデータからの学習がかんたんにできます。モデルを訓練するための日本語もしくは日本語を含む多言語のデータセットも充実してきており、すぐに業務に使える実用的なモデルを作ることが可能な時代がやってきました。
本書は、大規模言語モデルの理論と実装の両方を解説した入門書です。大規模言語モデルの技術や自然言語処理の課題について理解し、実際の問題に対処できるようになることを目指しています。以下のような構成によって、理論とプログラミングの双方の側面から、大規模言語モデルに関する情報を提供します。
</div>
        <div class="shop-card-link">
            <a href="https://amzn.to/4f6943e" target="_blank" rel="noopener">Amazonで詳細を見る</a>
        </div>
    </div>
</div>
