# 日本語のMLMデータセット
日本語のMLM（Masked Language Model）をゼロから、あるいは継続事前学習（Continuous Pre-training）する場合、**量と質（クリーニング済みか）のバランス**が非常に重要です。

2024年〜2025年現在のトレンドにおいて、最も推奨されるデータセットをカテゴリ別に紹介します。

-----

### 1\. 【イチオシ】CulturaX (Japanese subset)

現在、オープンな多言語データセットの中で**最も品質と量のバランスが良い**とされているものです。

  * **概要:** mC4（Google）とOSCARという2大巨大データセットを統合し、**重複排除（Deduplication）とクリーニング**を徹底的に行ったデータセットです。
  * **推奨理由:**
      * mC4などの生データはノイズ（広告、メニューバー、スパム）が多いですが、CulturaXはこれらがかなり除去されています。
      * MLMの学習において「同じ文章を何度も学習すること」は過学習の原因になりますが、重複排除されているため効率的です。
  * **Hugging Face:** `uonlp/CulturaX`

<!-- end list -->

```python
from datasets import load_dataset

# 日本語サブセットをストリーミングで読み込む（サイズが巨大なため）
dataset = load_dataset("uonlp/CulturaX", "ja", split="train", streaming=True)

# 中身の確認
print(next(iter(dataset)))
```

### 2\. 【高品質・知識ベース】Wikipedia (Japanese)

MLMには「事実知識」や「文法的な正しさ」を学習させたいので、Wikipediaは必須のコンポーネントです。

  * **概要:** 日本語Wikipediaの全記事。
  * **推奨理由:**
      * 文章が整形されており、文法が正しい。
      * 事実知識の密度が高い。
      * ただし、単体では量が少ない（数GB程度）ため、Webコーパス（CulturaXなど）と混ぜて使うのが鉄則です。
  * **Hugging Face:** `wikipedia` (指定日は最新のものを選ぶ)

<!-- end list -->

```python
dataset = load_dataset("wikipedia", "20231101.ja", split="train")
```

### 3\. 【圧倒的量】mC4 (Multilingual C4)

とにかく量が欲しい場合、あるいは自分でフィルタリングルールを厳密に決めたい場合に選択します。

  * **概要:** GoogleのT5モデルなどの学習に使われたCommon Crawlベースのデータセット。
  * **推奨理由:**
      * 日本語データだけでも圧倒的な量があります。
      * **注意点:** ノイズが多いため、そのまま学習に使うとモデルの性能が下がることがあります。`hojichar` などのツールでフィルタリングするか、前述の `CulturaX` を使う方が無難です。
  * **Hugging Face:** `google/mc4`

### 4\. 【国内トレンド】Swallow Corpus / LLM-jp Corpus

日本の研究機関（東工大やNIIなど）が整備しているデータセット群です。Hugging Face上ですぐに使える形（`load_dataset`一発）で公開されているものは限定的ですが、最新の動向として知っておくべきです。

  * **Swallow Corpus:** Llama-3-Swallowなどを開発したチームによるWebコーパス。
  * **LLM-jp Corpus:** 日本のLLM開発プロジェクトによるコーパス。

これらは直接Hugging FaceのDatasetsハブに全量が置かれていない場合がありますが（Common CrawlのURLリストとして提供されるなど）、加工済みのものとして `izumi-lab` などが公開しているデータセットが参考になります。

  * **例:** `izumi-lab/wikipedia-ja-20230720` など

-----

### 💡 おすすめのデータ構成レシピ

MLMを学習させる際は、単一のデータセットではなく、\*\*混合比率（Mix Ratio）\*\*を調整するのが一般的です。

**推奨レシピ例:**

1.  **Webテキスト (CulturaX)**: **70%**
      * 多様な表現、口語、最近のトピックを学習させるため。
2.  **Wikipedia**: **15%**
      * 高品質な知識ベースとして。エポック内で重複して出現させても良いくらい質が高いです。
3.  **CC-100 (日本語)**: **15%**
      * XLM-RoBERTaなどで使われた定番データ。CulturaXと被る可能性はありますが、安定しています。

### 🛠️ 実践的なアドバイス：フィルタリングツール

もし `mC4` や自前のWebクロールデータを使う場合は、日本のLLM開発でデファクトスタンダードになっている前処理ツール **HojiChar (ホジチャ)** の使用を強くおすすめします。

  * **HojiChar**: [https://github.com/HojiChar/HojiChar](https://github.com/HojiChar/HojiChar)
      * 日本語特有のノイズ（「詳細はこちら」「広告」など）を除去するフィルターや、重複排除機能が揃っています。

**結論:**
まずは **`uonlp/CulturaX` (ja)** をメインに据え、知識補強として **`wikipedia`** を混ぜる構成から始めるのが、現在最も手堅く、かつ高性能なMLMを作る近道です。
