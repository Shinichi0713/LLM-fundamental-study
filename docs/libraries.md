LLM開発で実務・研究の両方で**頻出するライブラリ**を、役割別に体系的に整理します。
「何のために使うか」が分かるようにまとめています。

---

## 1. 基盤ライブラリ（必須）

__PyTorch__

* 深層学習フレームワークの主流
* LLM研究・実装の事実上の標準

```python
import torch
```

__NumPy__

* 数値計算の基礎
* 前処理・評価で頻出

```python
import numpy as np
```

---

## 2. LLMモデル・トークナイザ

### Hugging Face Transformers

* 事前学習済みLLMの利用・学習・微調整
* GPT, BERT, LLaMA, T5 など

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

### Tokenizers

* 高速トークナイザ（Rust実装）
* 大規模データ前処理に必須

```python
from tokenizers import Tokenizer
```

---

## 3. データセット・前処理

### datasets（Hugging Face）

* 大規模テキストデータの読み込み・管理

```python
from datasets import load_dataset
```

### pandas

* CSV / JSON データ操作

```python
import pandas as pd
```

---

## 4. 学習効率化・分散学習

### Accelerate

* マルチGPU / TPU / 混在環境を簡単に制御

```python
from accelerate import Accelerator
```

### DeepSpeed

* 大規模モデル学習の定番
* ZeRO optimization

```python
import deepspeed
```

### FairScale

* メモリ効率改善（FSDP）

---

## 5. 量子化・軽量化（実務で重要）

### bitsandbytes

* 8bit / 4bit 量子化
* QLoRAで必須

```python
import bitsandbytes as bnb
```

### PEFT

* LoRA / QLoRA / Prefix Tuning

```python
from peft import LoraConfig, get_peft_model
```

---

## 6. 学習管理・実験管理

### wandb

* 実験ログ可視化

```python
import wandb
```

### TensorBoard

* 学習曲線の確認

---

## 7. 評価・ベンチマーク

### evaluate

* BLEU / ROUGE / Accuracy など

```python
import evaluate
```

### sacrebleu / rouge-score

* 生成モデル評価

---

## 8. 推論・デプロイ

### vLLM

* 高速推論サーバ
* KV Cache最適化

```bash
pip install vllm
```

### Triton

* カスタムCUDAカーネル
* FlashAttentionの基盤

---

## 9. 推論補助・アプリケーション層

### LangChain

* RAG / Agent 構築

```python
from langchain.llms import HuggingFacePipeline
```

### LlamaIndex

* 文書インデックス・検索

---

## 10. VLM・マルチモーダル関連

### torchvision

* 画像前処理

```python
import torchvision.transforms as T
```

### OpenCLIP

* CLIP系モデル

```python
import open_clip
```

---

## 11. 可視化・デバッグ

### matplotlib / seaborn

* 学習可視化

### tqdm

* 進捗バー

---

## 12. 代表的な構成例（実務テンプレ）

```text
PyTorch
├─ Transformers
├─ Datasets
├─ Accelerate
├─ PEFT
├─ bitsandbytes
├─ wandb
└─ vLLM
```

---

## まとめ（重要）

LLM開発では、

* **基盤**：PyTorch + Transformers
* **効率化**：Accelerate / DeepSpeed / PEFT
* **実務**：bitsandbytes / vLLM / LangChain

この3層を押さえるのが王道です。

次に進むなら、

* フルスクラッチ学習 vs 微調整
* QLoRA構成の実装例
* VLM構成でのLLMの役割

などを具体コード付きで整理できます。
