# 目的

VLMを学習する前に、現状一般的なモデルや、学習用データ、注意点に関する情報を集める。

## モデル

### 人気のあるモデル

ローカルで動作するVLM（Vision-Language Model、視覚・言語モデル）において、最近人気のあるモデルには、日本語性能や軽量性を重視したものが含まれます。
主な人気モデルは以下の通りです。
Heron-NVILA-Lite-15B: 日本語VLMの新しい基準として注目されており、150億パラメータでクラス最高水準の性能を持ちながら、iPhone上でのローカル推論も実現しています。
Qwen (特にQwen2-VLシリーズ): Alibaba社が開発したモデルで、その効率性と性能のバランスから世界的に人気があります。2B、7B、72Bといった複数のパラメータサイズがあり、テキスト認識や動画理解能力が高いとされています。
Gemma 3: Googleがリリースしたモデルで、マルチモーダルをサポートしています。特に軽量なGemma 3 270Mは、基本的なタスクをこなすことができ、高い汎用性を持っています。
DeepSeek-VL2 / Phi-4 Multimodal / Pixtral: これらは特定の用途（科学分野、モバイルAI、ドローンなど）に特化しつつ、効率的なMoE（Mixture of Experts）設計や超軽量性を特徴としており、ローカル環境での実行に適しています。
LLaVA-OneVision: オープンソースコミュニティで人気があり、軽量でアクセスしやすいため、視覚的なQ&Aシステムなどの迅速なプロジェクトに適しています。
Sarashina2.2-Vision-3B: 日本語に特化したコンパクトかつ高性能なVLMとして、日本の企業によって公開されています。

### 様子見

③ Llama-3.1-Swallow-8B-Instruct-v0.2
MetaのLlama3.1モデルに、東京科学大学＆産総研が日本語チューニングしたモデルです。

huggingfaceのページはこちら、

tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2 · Hugging Face
We’re on a journey to advance and democratize artificial inte
huggingface.co
筆者が利用しているGGUFファイルの入手先はこちら、

mradermacher/Llama-3.1-Swallow-8B-Instruct-v0.2-i1-GGUF · Hugging Face
We’re on a journey to advance and democratize artificial inte
huggingface.co
パラメータ規模：8B
ベースモデル：Meta Llama 3.1
利用量子化サイズ：i1-Q4_K_M（imatrix版）
num_ctx：24576（モデルの上限は8192）

⑤ Gemma-2-2b-jpn-it
Googleが開発したgemma2の日本語チューニング2Bモデルです（Geminiのオープンソース版がGemmaシリーズ）。ELYZA等は、サードパーティーとしてLlamaベースモデルに日本語チューニングを行ったモデルですが、Gemma-jpnはGoogle謹製の日本語特化モデルです。

huggingfaceページはこちら、

google/gemma-2-2b-jpn-it · Hugging Face
We’re on a journey to advance and democratize artificial inte
huggingface.co
筆者が利用しているモデル

ollama pull 7shi/ezo-gemma-2-jpn:2b-instruct-q8_0

copy
パラメータ規模：2.6B
ベースモデル：gemma2
利用量子化サイズ：Q8_0
num_ctx：8192

エントリークラスのGPUや、モバイルGPUでも十分実用的な速度で動作します。現状で日本語をまともに扱えるSLMはgemm2-jpn（2B）一択です。

### Heron-NVIDIA-LITE

![1765697219499](image/1-introduction/1765697219499.png)

https://www.issoh.co.jp/tech/details/6742/#Heron-NVILA-Lite-15B

> 日本語という言語特有の構造や意味解釈に適応した新たな基準を提示しました。Heron-NVILA-Lite-15Bは、そうした取り組みの一環として、日本語と視覚情報を同時に扱えるマルチモーダルVLMとして設計されています。
> 日本語には助詞による文法構造や語順の自由度が高く、英語ベースのモデルでは文意を正確に理解できないケースも多々あります。こうした背景から、日本語を母語とするユーザーにとって違和感のない自然な応答、かつマルチモーダルに対応できるモデルの必要性が強く認識されてきました。

![1765697389891](image/1-introduction/1765697389891.png)

```
response = model.generate_content([image, "画像を説明してください。"])
print(response)
print("---" * 40)
```

```

画像には、2匹の猫がピンクのソファの上で寝ている様子が写っています。猫たちはリラックスした姿勢で、ソファの上でくつろいでいるようです。ソファの上には、2つのリモコンが置かれており、猫たちの近くに位置しています。

猫たちはそれぞれ異なる位置に寝ており、1匹はソファの中央付近で、もう1匹はその少し左側にいます。ソファの色合いがピンクであることから、猫たちの寝姿がとても可愛らしい印象を与えています。
```



#### 複数の画像とテキストを入力に渡すこともできる。

![1765697501007](image/1-introduction/1765697501007.png)

```
response = model.generate_content([
    images[0],
    "これは日本の画像です",
    images[1],
    "これはオーストリアの画像です",
    "各画像の違いを説明して"])
print(response)
print("---" * 40)
```

出力

```
この画像は日本の都市の風景で、交差点に立つ信号機が赤色に点灯しています。一方、オーストリアの画像では、信号機が緑色に点灯しており、自転車が横断歩道を渡っている様子が見えます。また、オーストリアの画像には、自転車の他に、歩行者も見えます。さらに、オーストリアの画像には、建物の外観や街灯、街路標識も確認できます。これらの違いから、日本の画像は都市の交差点の様子を示しており、オーストリアの画像は自転車や歩行者が交差点を渡る様子を示しています。
```



## ビジョン系モデルをローカルで扱う際のお勧めと注意点

ローカル系ビジョンモデル環境構築に関するお勧めと注意点です。

・テキスト系に比べてメモリとストレージをかなり消費するので、ColabやSageMaker等との併用がお勧めです。
・detasetのデータはParquetファイル等で圧縮し、HuggingFaceのdatasetsモジュールで読み込み、編集・加工、アップロードし、ローカルには極力保存しない方が良い。
・Llamaのような準標準的なモデルが無く、強いて言うとLlava系モデルが比較的多くなっているが、各モデルとも独自実装色が強いので、inferenceはまだしもFine Tuningになるとモデル毎の個別対応部分がかなりのウェイトを占めるようになる。
・テキスト系ではかなり便利なOllama等のプラットフォームもビジョン系で使えるモデルが現時点少なく、コードを自分で作成する必要が高くなるので、コード生成AIや先駆者の記事等を参考にした方が早い。
・DeepSpeed、flash_attention2、bitsandbytes、mixed_precision等の処理効率化や省メモリのツールが殆どMPS（Apple silicon）非対応で、mlx-lm/vlmも対象ビジョン系モデルが少ないので、多くのリソースを必要とするビジョン系モデルのファインチューニングはMacのMPSには向いて無いように思われます（間違っていたらご指摘下さい）。

## The Cauldron データセットに関して

The Cauldron データセットは、huggingface が idefics2 の公開と同時にリリースした VLM (Vision-Language Model) の fine-tuning 用の V&L (Vision & Language) データセットです。

VLM を fine-tuning する際には、Captioning, VQA, Chart/Figure understanding など様々なタスク指向の V&L データセットが使用されますが、これらの既存のデータセットは多くの場合、ファイル形式やデータ形式が異なっていることや、データセットの保存場所が点在してしまっているといったような問題を抱えています。

そのため、これまでは点在した V&L データセットを手元にダウンロードし、すべてのデータ形式をユーザー側で統一する必要がありました。

しかし、この作業には予想以上の手間がかかるので、VLM の fine-tuning によく使われる V&L データセットが同じデータ形式で１つの保存場所にまとまっていたらいいのになぁ … と思うわけです。
この願いを叶えたデータセットが、The Cauldron データセットです！
