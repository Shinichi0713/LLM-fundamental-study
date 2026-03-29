from datasets import load_dataset

# helpful-base サブセットをロード
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpless-base")
# または helpful-base / helpful-online なども使えます
# dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")


import re

def split_dialogue(text):
    """
    "Human: ...\nAssistant: ..." 形式のテキストを
    (prompt, response) に分割する簡易関数
    """
    # Human: と Assistant: で分割
    parts = re.split(r"\nAssistant:\s*", text, maxsplit=1)
    if len(parts) == 2:
        prompt = parts[0].replace("Human:", "").strip()
        response = parts[1].strip()
        return prompt, response
    else:
        # 分割できない場合はそのまま返す（あまりないはず）
        return text, ""

def prepare_dpo_dataset(raw_dataset, num_samples=1000):
    """
    HH-RLHF の (chosen, rejected) を
    DPO用の (prompt, chosen, rejected) 形式に変換
    """
    dpo_data = []
    for i, example in enumerate(raw_dataset):
        if i >= num_samples:
            break
        chosen_text = example["chosen"]
        rejected_text = example["rejected"]

        # chosen 側からプロンプトを抽出
        prompt, chosen_response = split_dialogue(chosen_text)
        _, rejected_response = split_dialogue(rejected_text)

        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        })
    return dpo_data

# 学習用・評価用に分割
train_raw = dataset["train"]
eval_raw = dataset["test"] if "test" in dataset else dataset["train"].select(range(100, 200))

# サンプル数を制限（Colabのメモリ制限のため）
train_dpo = prepare_dpo_dataset(train_raw, num_samples=500)
eval_dpo = prepare_dpo_dataset(eval_raw, num_samples=100)

print(f"学習データ数: {len(train_dpo)}")
print(f"評価データ数: {len(eval_dpo)}")
print("例:")
print(train_dpo[0])

