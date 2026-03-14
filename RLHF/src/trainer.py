# 必要なライブラリのインストール
!pip install -q trl peft transformers datasets accelerate

import torch
from datasets import Dataset
from trl import DPOTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# 1. モデルの準備 (QwenやTinyLlamaなど、Colabのメモリに収まるもの)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)

# 2. 学習用データセットの準備 (DPOは「好ましい回答」と「好ましくない回答」のペアを使う)
data = [
    {
        "prompt": "こんにちは、元気ですか？",
        "chosen": "はい、元気です！あなたはどうですか？",
        "rejected": "私はAIなので感情はありません。"
    },
    # ... データセットをここに追加
]
dataset = Dataset.from_list(data)

# 3. LoRA設定 (VRAM節約のため必須)
peft_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
)

# 4. トレーニング設定
training_args = TrainingArguments(
    output_dir="./dpo_results",
    per_device_train_batch_size=1, # Colab向けに極小化
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    max_steps=100,
    logging_steps=10,
)

# 5. DPOトレーナーの実行
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None, # Noneにすると自動的にモデルをコピーしてリファレンスとして使用
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    args=training_args,
)

dpo_trainer.train()