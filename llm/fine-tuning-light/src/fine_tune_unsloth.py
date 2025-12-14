from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import torch

model_name = "unsloth/gemma-2b-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    load_in_4bit = True,     # 4bit量子化モデルを使用
    dtype = None,            # 自動で最適dtypeを選択
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,                                # LoRAランク（初心者向け）
    target_modules = ["q_proj", "v_proj"],# 触る層を最小限に
    lora_alpha = 8,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
)

dataset = load_dataset("json", data_files="train.jsonl")

def format_prompt(example):
    return {
        "text": f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
    }

dataset = dataset.map(format_prompt)

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

model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")

