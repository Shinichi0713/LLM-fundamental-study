LLM開発で実務・研究の両方で**頻出するライブラリ**を、役割別に体系的に整理します。
「何のために使うか」が分かるようにまとめています。


__1. 基盤ライブラリ（必須）__

__PyTorch__

ニューラルネットワーク構築する際に最も重要なライブラリです。
ニューラルネットワークに必要な処理を全て実装しているため、フレームワークと呼ばれています。

* 深層学習フレームワークの主流
* LLM研究・実装の事実上の標準

```python
import torch
```

__NumPy__

数理計算を扱うライブラリです。
データを読み出してから前処理と呼ばれるデータ処理によく使われます。

* 数値計算の基礎
* 前処理・評価で頻出

```python
import numpy as np
```

__2. LLMモデル・トークナイザ__

__Hugging Face Transformers__

BERT、GPT、Llamaなどの最先端モデルを数行のコードで読み込み、推論（実行）や追加学習（ファインチューニング）を行うためのメインライブラリです。

* 事前学習済みLLMの利用・学習・微調整
* GPT, BERT, LLaMA, T5 など

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

__Tokenizers__

人間が読む「文章」を、コンピューター（モデル）が計算できる「数値（ID）の列」に変換（トークン化）するライブラリです。

* 高速トークナイザ（Rust実装）
* 大規模データ前処理に必須

```python
from tokenizers import Tokenizer
```

__3. データセット・前処理__

__datasets（Hugging Face）__

世界中で公開されている膨大な機械学習用データセットを簡単にダウンロードし、メモリを節約しながら効率よく読み込むためのライブラリです。

* 大規模テキストデータの読み込み・管理

```python
from datasets import load_dataset
```

__pandas__

機械学習にかける前のデータの汚れを落としたり（クリーニング）、並び替えたり、統計をとったりする際に必須となるツールです。

* CSV / JSON データ操作

```python
import pandas as pd
```

__4. 学習効率化・分散学習__

__Accelerate__

単一のGPU、複数のGPU、あるいはTPUなど、実行環境が変わってもコードを書き換えずに並列処理を行えるようにするライブラリです。

* マルチGPU / TPU / 混在環境を簡単に制御

```python
from accelerate import Accelerator
```

__DeepSpeed__

「ZeRO」という技術により、モデルのデータを複数のGPUに賢く分散配置することで、限られたメモリで巨大なモデルを扱うことができます。

* 大規模モデル学習の定番
* ZeRO optimization

```python
import deepspeed
```

__FairScale__

* メモリ効率改善（FSDP）

__5. 量子化・軽量化（実務で重要）__

__bitsandbytes__

通常なら巨大なVRAM（ビデオメモリ）が必要なモデルを、家庭用のGPUでも動かせるほどメモリ消費を劇的に抑えることができます。

* 8bit / 4bit 量子化
* QLoRAで必須

```python
import bitsandbytes as bnb
```

__PEFT__

モデル全体を再学習するのではなく、LoRA（Low-Rank Adaptation）などの手法を用いて、ごく一部のパラメータだけを更新するライブラリです。

* LoRA / QLoRA / Prefix Tuning

```python
from peft import LoraConfig, get_peft_model
```

__6. 学習管理・実験管理__

__wandb__

学習中のロス（損失）や精度などの推移を、リアルタイムで美しいグラフにしてWeb上で確認できるツールです。

※本ライブラリを利用する場合、有償アカウントが必要となります。

※本書で使うライブラリにオプション機能で実装されていることが多いライブラリです。

* 実験ログ可視化

```python
import wandb
```

__TensorBoard__

* 学習曲線の確認

__7. 評価・ベンチマーク__

__evaluate__

様々な評価指標（メトリクス）を一つのライブラリで簡単に計算できるようにしたツールです。

* BLEU / ROUGE / Accuracy など

```python
import evaluate
```

__sacrebleu / rouge-score__

sacrebleu: 主に機械翻訳の精度を測る「BLEUスコア」を計算します。実装による計算のズレを防ぎ、世界共通の基準で比較できるように工夫されています。

* 生成モデル評価

__8. 推論・デプロイ__

__vLLM__

PagedAttentionというOSの仮想メモリのような技術を導入しており、GPUメモリを無駄なく使うことで、通常の数倍～数十倍の効率でリクエストを処理できます。

* 高速推論サーバ
* KV Cache最適化

```bash
pip install vllm
```

__Triton__

NVIDIAのGPUを動かすための「CUDA」という難しい言語を知らなくても、Pythonに近い書き方で高速なGPU処理（カーネル）を書けるようにする言語・コンパイラです。

* カスタムCUDAカーネル
* FlashAttentionの基盤

__9. 推論補助・アプリケーション層__

__LangChain__

LLM単体ではできないこと（記憶を持たせる、Web検索をさせる、PDFを読ませるなど）を実現するために、様々な機能を繋ぎ合わせる（Chain）フレームワークです。

* RAG / Agent 構築

```python
from langchain.llms import HuggingFacePipeline
```

__LlamaIndex__

膨大な社内文書やデータベースをLLMが扱いやすい形式（インデックス）に変換し、必要な情報を正確に検索してLLMに渡す仕組みを簡単に構築できます。

* 文書インデックス・検索

__10. VLM・マルチモーダル関連__

__torchvision__

PyTorchで画像認識（AI）を行うための標準的なライブラリです。

* 画像前処理

```python
import torchvision.transforms as T
```

__OpenCLIP__

OpenAIが開発した「CLIP（画像と説明文を同じ空間で理解する技術）」を、誰でも自由に使えるようにしたライブラリです。

* CLIP系モデル

```python
import open_clip
```

__11. 可視化・デバッグ__

__matplotlib / seaborn__

数値データを折れ線グラフ、棒グラフ、散布図、ヒートマップなどの「目に見える形」にするためのライブラリです。
AIの学習中の損失（ロス）の推移をグラフ化したり、認識結果の画像を表示したりする際に必須となります。

* 学習可視化

__tqdm__

時間がかかる計算やデータの読み込み（forループなど）の際、あとどれくらいで終わるかを視覚的なバーで表示してくれるライブラリです。

* 進捗バー
