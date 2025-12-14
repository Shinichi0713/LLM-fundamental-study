Unslothでファインチューニングする場合、モデルに対して行われる処理は多くありません。
ただし裏側では、**「LLMを壊さず・軽く・速く学習するための一連の下処理」**が自動で行われています。
初心者にも分かるように、**流れとして**説明します。

---

## 全体像を一文で

> **Unslothのファインチューニングでは、
> 「既存のLLMを軽量化 → 学習してはいけない部分を固定 → 学習用の小さな部品（LoRA）を取り付ける」
> という処理が行われます。**

---

## 1. モデル読み込み時に行われる処理

### ① 量子化されたモデルとして読み込む（重要）

Unslothでは多くの場合、

* **4bit量子化モデル** （bnb-4bit）を使用
* モデルの重みを軽い形式でメモリに配置

**何が起きているか**

* 精度を極力保ったまま
* GPUメモリ消費を大幅削減

初心者視点では、

> 「大きすぎて載らないモデルを、載る形にしている」

と理解すれば十分です。

---

### ② Unsloth専用の高速実装に差し替え

* Attention
* FFN
* 出力処理

などが、**Unsloth用に最適化された実装**に置き換えられます。

**効果**

* 学習が速くなる
* メモリ使用量が減る
* OOMが起きにくくなる

この段階で、
**普通にTransformersで読み込むモデルとは中身が少し違う状態**になります。

---

## 2. ファインチューニング前に行われる処理

### ③ 元のモデルの重みをすべて固定（freeze）

Unslothでは、

* 事前学習済みの重み
  → **更新しない**
* 勾配計算もしない

**意味**

* もとの賢さを壊さない
* 壊滅的忘却を防ぐ

初心者向けに言うと、

> 「賢い脳みそには触らない」

---

### ④ LoRAアダプタを追加する

次に行われるのが **PEFT（LoRA）** の処理です。

* Attention層などに
* **小さな学習用パラメータ（LoRA）**を追加
* 学習対象はここだけ

**重要**

* モデル全体のごく一部しか学習しない
* 学習が軽く、安全

ここで初めて、

> 「新しい知識を覚える場所」

が作られます。

---

## 3. 学習中に行われる処理

### ⑤ 学習はLoRA部分のみ

学習ループ中は、

* Baseモデル
  → 計算に使うだけ
* LoRA部分
  → 重みが更新される

**結果**

* 学習が速い
* GPUメモリをほとんど使わない
* モデルが壊れにくい

---

### ⑥ メモリ節約と高速化の工夫（内部処理）

Unslothは学習中に以下を自動調整します。

* 不要な中間結果を保存しない
* 勾配チェックポイントを最小化
* Attention計算をまとめて処理

初心者は意識しなくてよく、

> 「勝手に軽くしてくれる」

という理解で問題ありません。

---

## 4. 学習後に残るもの

ファインチューニング後に得られるのは、

* **元のモデル（変更なし）**
* **LoRAアダプタ（学習結果）**

です。

つまり、

* ベースモデルは再利用可能
* タスクごとにLoRAを差し替えられる

という構造になります。

---

## 5. 処理の流れまとめ（図解イメージ）

```
① 量子化モデル読み込み
        ↓
② Unsloth高速化処理
        ↓
③ Baseモデルを固定
        ↓
④ LoRAを追加
        ↓
⑤ LoRAのみ学習
        ↓
⑥ Adapterとして保存
```

---

## 6. 初心者向け一文まとめ

> **Unslothでのファインチューニングでは、
> モデルを軽くして、安全な場所（LoRA）だけに学習させる処理が自動で行われる。
> そのため「壊れにくく・速く・少ないGPUで」学習できる。**

---

次の段階としては、

* 「どの層にLoRAが入っているのか」
* 「学習後のモデルをどう使うのか」
* 「うまく学習できたかをどう確認するか」

を理解すると、LLMファインチューニングの全体像が完成します。


# 実装法

以下は、先ほどのフロー

```
① 量子化モデル読み込み
② Unsloth高速化処理
③ Baseモデルを固定
④ LoRAを追加
⑤ LoRAのみ学習
⑥ Adapterとして保存
```

を **Unslothでそのまま実現する、最小かつ実務的なPython実装例**です。
初心者が「まず成功する」ことを重視した構成にしています。

---

## モデル選定

**unsloth/gemma-2b-bnb-4bit**

理由:

* 2Bで軽量
* 4bit量子化済み
* Unsloth公式対応
* ファインチューニング入門に最適

---

## 前提

```bash
pip install unsloth transformers datasets accelerate
```

GPU: 1枚（12〜24GB想定）

---

## 実装例（コメント付き）

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import torch
```

---

### ① 量子化モデル読み込み ＋ ② Unsloth高速化処理

```python
model_name = "unsloth/gemma-2b-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    load_in_4bit = True,     # 4bit量子化モデルを使用
    dtype = None,            # 自動で最適dtypeを選択
)
```

👉 この時点で

* 量子化
* 高速Attention
* メモリ最適化

が **すべて内部で適用済み** です。

---

### ③ Baseモデルを固定 ＋ ④ LoRAを追加

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,                                # LoRAランク（初心者向け）
    target_modules = ["q_proj", "v_proj"],# 触る層を最小限に
    lora_alpha = 8,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
)
```

ここで起きていること:

* Baseモデルの重み → **完全に固定**
* LoRA Adapter → **追加**
* 学習対象 → **LoRAのみ**

---

### 学習データ準備（簡易例）

```python
dataset = load_dataset("json", data_files="train.jsonl")

def format_prompt(example):
    return {
        "text": f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
    }

dataset = dataset.map(format_prompt)
```

※ `instruction` / `output` は任意の指示データ形式

---

### ⑤ LoRAのみ学習

```python
training_args = TrainingArguments(
    output_dir = "./outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    learning_rate = 2e-4,
    num_train_epochs = 3,
    fp16 = True,
    logging_steps = 10,
    save_steps = 500,
    save_total_limit = 2,
    optim = "adamw_8bit",
    report_to = "none",
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset["train"],
    tokenizer = tokenizer,
)

trainer.train()
```

👉 学習中に更新されるのは **LoRAパラメータのみ**
👉 Baseモデルは一切変更されません

---

### ⑥ Adapterとして保存

```python
model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")
```

保存されるのは:

* 数MB〜数十MBの **LoRA Adapter**
* ベースモデルは別途そのまま利用可能

---

## フローとの対応関係

| フロー                  | 実装箇所                               |
| ----------------------- | -------------------------------------- |
| ① 量子化モデル読み込み | `from_pretrained(load_in_4bit=True)` |
| ② Unsloth高速化処理    | `FastLanguageModel.from_pretrained`  |
| ③ Baseモデルを固定     | `get_peft_model`内部                 |
| ④ LoRAを追加           | `get_peft_model`                     |
| ⑤ LoRAのみ学習         | `Trainer.train()`                    |
| ⑥ Adapterとして保存    | `save_pretrained()`                  |

---

## 初心者向け一文まとめ

> **Unslothでは、
> 「量子化・高速化・凍結・LoRA追加」という面倒な前処理を
> ほぼ数行で安全に完了できるため、
> LLMファインチューニングを“実際に成功させる”ことに集中できる。**

---

次に進むなら、

* 学習後のAdapterを使った推論例
* LoRAをマージする／しない判断
* 学習が成功したかの確認方法

まで一気に整理できます。
