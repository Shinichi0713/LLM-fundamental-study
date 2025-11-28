from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# ------------------------------
# 1. 4bit量子化の設定
# ------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,             # 4bit 量子化
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",     # 通常は "nf4"
    bnb_4bit_compute_dtype=torch.bfloat16,  # 計算精度
)

# ------------------------------
# 2. モデルを4bit量子化して読み込み
# ------------------------------
model_name = "meta-llama/Llama-3-8b-instruct"  # ← 任意のモデルに変更OK

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",             # GPUへ自動配置
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# 3. LoRA の設定（重要）
# ------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# ------------------------------
# 4. データを読み込む
# ------------------------------
dataset = load_dataset("ybelkada/ultrachat_200k", split="train[:2000]")

def format_example(example):
    prompt = example["prompt"]
    answer = example["answer"]
    text = f"<s>Instruction: {prompt}\nAnswer: {answer}</s>"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(format_example)

# ------------------------------
# 5. 学習設定
# ------------------------------
training_args = TrainingArguments(
    output_dir="qlora-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",  # メモリ効率化
    logging_steps=10,
    save_steps=200,
    ddp_find_unused_parameters=False,
)

# ------------------------------
# 6. Trainer で学習開始
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# ------------------------------
# 7. LoRA のみを保存
# ------------------------------
model.save_pretrained("qlora-lora-adapter")
