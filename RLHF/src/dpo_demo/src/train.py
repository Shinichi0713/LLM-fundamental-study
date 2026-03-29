from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,                  # DPOの温度パラメータ
    learning_rate=5e-5,
    per_device_train_batch_size=1,  # Colabのメモリ制限のため小さめ
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    eval_steps=50,
    save_steps=50,
    logging_steps=10,
    output_dir="./dpo-stablelm-zephyr-3b",
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # tokenizer_name_or_path=model_name,  # ← ここを修正
)

trainer.train()

trainer.save_model("./dpo-stablelm-zephyr-3b-final")
tokenizer.save_pretrained("./dpo-stablelm-zephyr-3b-final")