def tokenize_dpo_sample(sample):
    """
    DPO用サンプル (prompt, chosen, rejected) をトークナイズ
    """
    # プロンプト＋応答の形式に整形
    chosen_text = f"{sample['prompt']}\nAssistant: {sample['chosen']}"
    rejected_text = f"{sample['prompt']}\nAssistant: {sample['rejected']}"

    # トークナイズ
    chosen_tokens = tokenizer(
        chosen_text,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    rejected_tokens = tokenizer(
        rejected_text,
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    return {
        "input_ids": chosen_tokens["input_ids"],
        "attention_mask": chosen_tokens["attention_mask"],
        "chosen_labels": chosen_tokens["input_ids"],
        "rejected_labels": rejected_tokens["input_ids"],
    }

# データセットをトークナイズ
from datasets import Dataset

train_dataset = Dataset.from_list(train_dpo).map(tokenize_dpo_sample, batched=False)
eval_dataset = Dataset.from_list(eval_dpo).map(tokenize_dpo_sample, batched=False)