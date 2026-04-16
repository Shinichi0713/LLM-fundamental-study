
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

model_name = "mistralai/Mistral-7B-v0.1"  # 例：7Bモデル

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA設定（オプションだが推奨）
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


from datasets import Dataset

# ダミーデータ（実際にはUltraFeedbackなどを使う）
dummy_data = {
    "prompt": [
        "Explain IPO.",
        "What is RLHF?",
    ],
    "chosen": [
        "IPO stands for Identity Preference Optimization...",
        "RLHF is Reinforcement Learning from Human Feedback...",
    ],
    "rejected": [
        "IPO is a financial term meaning Initial Public Offering...",
        "RLHF is not important for LLM alignment.",
    ],
}

dataset = Dataset.from_dict(dummy_data)


from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer

# DPOConfigでIPO用の設定をまとめる
ipo_config = DPOConfig(
    output_dir="./ipo-stablelm-zephyr-3b",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    eval_steps=50,
    save_steps=50,
    logging_steps=10,
    remove_unused_columns=False,
    # fp16=True,  # GPUに応じて必要なら有効化

    # IPO用の設定
    beta=0.01,          # IPOでは小さめのbetaが推奨
    loss_type="ipo",    # IPO用の損失関数
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=ipo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  # ← tokenizer ではなく processing_class に変更
)

trainer.train()

trainer.save_model("./ipo-stablelm-zephyr-3b-final")
tokenizer.save_pretrained("./ipo-stablelm-zephyr-3b-final")

