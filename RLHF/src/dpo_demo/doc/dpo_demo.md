はい、Google ColabでDPO（Direct Preference Optimization）の実演は可能です。ただし、**無料版のT4 GPU（VRAM約15GB）では、モデルサイズを小さめに選ぶ必要があります**。

以下に、Google Colabでそのまま実行できるコード例と注意点をまとめます。

---

## 1. 実行可能なモデルサイズの目安

| モデルサイズ | LoRA不使用 | LoRA使用 |
|-------------|-----------|----------|
| ~1Bパラメータ | 可能（VRAM余裕あり） | 非常に楽 |
| ~3Bパラメータ | 厳しい（バッチサイズ1でもギリギリ） | 可能 |
| ~7Bパラメータ | 不可 | バッチサイズ1で調整可能 |

**推奨**: まずは `Qwen2-0.5B-Instruct`（約0.5B）や `gpt2`（約124M）で動作を確認し、慣れてきたら `LoRA` を使って `Phi-2`（2.7B）や `Qwen2.5-3B` などに挑戦するのが現実的です。

---

## 2. Google Colabでの準備

まず、ランタイム → ランタイムのタイプを変更 → ハードウェアアクセラレータを **T4 GPU** に設定してください。

以下のセルを順番に実行していきます。

### ライブラリのインストール

```python
!pip install -q trl transformers datasets accelerate peft
```

### 必要なモジュールのインポート

```python
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
```

---

## 3. 最小限のDPO実装コード

### 3.1 モデルとトークナイザーの読み込み

```python
model_name = "Qwen/Qwen2-0.5B-Instruct"  # 軽量でColab向き

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# パディングトークンの設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
```

### 3.2 選好データの準備

DPOには **「prompt（質問）」「chosen（好ましい回答）」「rejected（好ましくない回答）」** の3つ組が必要です。

```python
# サンプルデータ（実際には数百～数千件の選好データが必要）
preference_data = [
    {
        "prompt": "AIとは何ですか？",
        "chosen": "AI（人工知能）は、人間の知能を模倣して学習・推論・問題解決を行うコンピューターシステムです。",
        "rejected": "AIとは、単なる自動化プログラムのことです。"
    },
    {
        "prompt": "日本の首都はどこですか？",
        "chosen": "日本の首都は東京都です。",
        "rejected": "日本の首都は大阪です。"
    },
    {
        "prompt": "Pythonでリストの重複を削除するには？",
        "chosen": "set()を使って重複を削除し、必要に応じてlist()で再変換します。例：list(set(my_list))",
        "rejected": "for文で一つずつ確認して新しいリストに追加するしかありません。"
    },
]

dataset = Dataset.from_list(preference_data)
```

### 3.3 LoRAの設定（推奨）

フルファインチューニングよりVRAMを大幅に節約できます。

```python
peft_config = LoraConfig(
    r=16,                    # LoRAのランク
    lora_alpha=32,           # スケーリング係数
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]  # モデル構造に応じて調整
)
```

### 3.4 トレーニング設定

```python
training_args = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                      # DPOの温度パラメータ（通常0.1～0.5）
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # 実質バッチサイズ4
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=10,
    fp16=True,                     # T4はfloat16を推奨
    report_to="none"               # WandB等を使わない場合
)
```

### 3.5 DPOトレーニングの実行

```python
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Noneにすると、modelのコピーを内部で作成（PEFT使用時は自動処理）
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config
)

trainer.train()
```

### 3.6 学習後の推論テスト

```python
prompt = "AIとは何ですか？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 4. 実際にColabで実行する際の重要なポイント

### VRAM管理のコツ
- `device_map="auto"` を使い、モデルを自動的にGPUに配置する
- `torch_dtype=torch.float16` で半精度にする（T4はbfloat16非対応）
- `gradient_checkpointing` を有効にするとさらにVRAMを節約できます（`training_args`に `gradient_checkpointing=True` を追加）

### ref_modelについて
- DPOは「学習前の参照モデル（ref_model）」と「学習中のモデル（model）」の出力確率の差を最大化します。
- `ref_model=None` にすると、TRLが自動的に参照モデルを作成しますが、これでVRAMが2倍必要になる場合があります。
- **PEFT（LoRA）を使う場合、ref_model=Noneが推奨**され、PEFTアダプタを外した状態が自動的に参照モデルとして使われるためVRAM効率が良くなります。

### データセット
- 実用的な選好データが必要な場合、Hugging Faceの `trl-lib/ultrafeedback_binarized` などの公開選好データセットが利用できます。
- ただし、データセットの形式が `prompt/chosen/rejected` になっていることを確認してください。

---

## 5. まとめ

Google Colabの無料版T4 GPUでも、**0.5B～1Bクラスのモデル + LoRA** であればDPOの実演は十分に可能です。上記のコードをそのままColabのセルに貼り付けて実行することで、選好データからの直接最適化（DPO）の流れを体験できます。

もし7Bクラスのモデルで試したい場合は、以下の対策が必要です。
- LoRAのランクを下げる（r=8やr=4）
- バッチサイズを1にして、gradient_accumulation_stepsを増やす
- シーケンス長（max_length）を短くする（256や512トークンに制限）

## データセット

はい、DPOトレーニングに利用できる公開データセットは複数存在します。以下、主なデータセットをまとめます。

---

## 1. UltraFeedback（最も広く使われている）

| 項目 | 内容 |
|------|------|
| **データセット名** | `trl-lib/ultrafeedback_binarized` |
| **規模** | 約6万ペア（binarized版） |
| **特徴** | TRLライブラリの公式推奨データセット |
| **形式** | prompt / chosen / rejected の3カラム |
| **内容** | 多様な指示に対する複数モデルの回答を、GPT-4等で評価・選好付けしたデータ |

TRLのDPOTrainerと最も親和性が高く、DPOの実装例で最も頻繁に参照されるデータセットです。<source-chip title="Hugging Face" url="https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized" />

---

## 2. Anthropic HH-RLHF

| 項目 | 内容 |
|------|------|
| **データセット名** | `Anthropic/hh-rlhf` |
| **規模** | 約17万ペア（helpful + harmless） |
| **特徴** | RLHF研究の先駆け的データセット |
| **内容** | 「有用性（helpful）」と「無害性（harmless）」の2つの観点から人間が選好を付けた対話データ |

元々はRLHF（PPO）用に作成されましたが、DPOでもそのまま利用可能です。DPO用に整形された版として `Columbia-NLP/DPO-hh-rlhf` も公開されています。<source-chip title="Hugging Face" url="https://huggingface.co/datasets/Anthropic/hh-rlhf" />

---

## 3. Stanford Human Preferences（SHP）

| 項目 | 内容 |
|------|------|
| **データセット名** | `stanfordnlp/SHP` / `stanfordnlp/SHP-2` |
| **規模** | 38.5万件（v1）/ 480万件（v2） |
| **特徴** | Reddit等のコミュニティ回答に対する人間の選好データ |
| **内容** | 料理、法律相談、プログラミングなど18～129の分野にわたる回答の選好 |

大規模で多様なドメインをカバーしており、汎用的な選好学習に適しています。<source-chip title="Hugging Face" url="https://huggingface.co/datasets/stanfordnlp/SHP-2" />

---

## 4. Tulu 2.5 Preference Data

| 項目 | 内容 |
|------|------|
| **データセット名** | `allenai/tulu-2.5-preference-data` |
| **規模** | 複数の選好データセットを統合・クリーニング |
| **特徴** | AllenAIがDPO・PPOのベストプラクティス研究のために統一フォーマット化 |
| **内容** | 複数の公開選好データセットを同じ形式に整形したもの |

DPOとPPOの比較研究のために作成されたデータセットで、フォーマットの統一性が高いのが特徴です。<source-chip title="Hugging Face" url="https://huggingface.co/datasets/allenai/tulu-2.5-preference-data" />

---

## 5. HelpSteer3-Preference（NVIDIA）

| 項目 | 内容 |
|------|------|
| **データセット名** | `nvidia/HelpSteer3-preference` |
| **規模** | 多様なタスク・言語にわたる人間による選好注釈 |
| **特徴** | 多言語対応、人間による直接の選好注釈 |
| **内容** | 複数のタスク領域にわたる選好データ |

NeurIPS 2025のDatasets & Benchmarks Trackで発表された比較的新しいデータセットです。<source-chip title="NeurIPS" url="https://papers.nips.cc/paper_files/paper/2025/file/3e0271cf7df2cdb3b91565ad1f525f3a-Paper-Datasets_and_Benchmarks_Track.pdf" />

---

## 6. その他のデータセット

| データセット名 | 規模 | 特徴 |
|-------------|------|------|
| `OpenRLHF/preference_700K` | 70万件 | 複数の選好データセットを統合した大規模データ |
| `openbmb/UltraFeedback` | 大規模 | 中国の研究チームによるUltraFeedbackの拡張版 |
| `HuggingFaceH4/ultrafeedback_binarized` | 約6万件 | HuggingFace H4チームによる整形版 |

---

## Google Colabで使う際のコード例

```python
from datasets import load_dataset

# UltraFeedback（最も推奨）
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# またはHH-RLHF
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# データの中身を確認
print(dataset[0])
# {'prompt': '...', 'chosen': '...', 'rejected': '...'}
```

---

## データセット選択のガイドライン

| 目的 | 推奨データセット |
|------|---------------|
| DPOの動作確認・実験 | `trl-lib/ultrafeedback_binarized` |
| 対話エージェントの改善 | `Anthropic/hh-rlhf` |
| 大規模・多様なドメイン | `stanfordnlp/SHP-2` |
| 研究・再現性重視 | `allenai/tulu-2.5-preference-data` |
| 多言語対応 | `nvidia/HelpSteer3-preference` |

---

## 注意点

DPOトレーニングでは、データセットの形式が **prompt / chosen / rejected** の3カラムになっていることが必要です。TRLのDPOTrainerはこの形式を標準で受け付けますが、データセットによっては `conversations` 形式や `history` 形式で提供されている場合があるため、必要に応じて前処理（フォーマット変換）が必要です。TRLのドキュメントに各Trainerが期待するデータセット形式が詳しく記載されていますので、参照されることをお勧めします。<source-chip title="TRL Documentation" url="https://huggingface.co/docs/trl/en/dpo_trainer" />