from datasets import Dataset
from trl import DPOTrainer
from transformers import TrainingArguments

# データセットの準備
dataset_dict = {
    "prompt": ["Explain IPO.", "What is RLHF?"],
    "chosen": ["IPO is...", "RLHF is..."],
    "rejected": ["IPO is a finance term...", "RLHF is not important..."],
}
dataset = Dataset.from_dict(dataset_dict)

# トレーニング引数
training_args = TrainingArguments(
    output_dir="./ipo_output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
)

# IPOで学習
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,        # 必要に応じて参照モデルを指定
    args=training_args,
    beta=0.01,            # IPOのbeta
    train_dataset=dataset,
    tokenizer=tokenizer,
    loss_type="ipo",       # ここが重要
)

dpo_trainer.train()