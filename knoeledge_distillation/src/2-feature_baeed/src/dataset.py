from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

# 1. データのロード
raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

# 2. トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    # 空行を除外
    texts = [t for t in examples["text"] if len(t) > 0 and not t.isspace()]
    return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

# 先にトークナイズ処理を行う
tokenized_datasets = raw_datasets.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# --- ここから分割処理 ---

# 3. trainデータセットをさらに分割 (例: 90% を学習用, 10% を検証用)
# ※ wikitext-2の元の "validation" セットを使っても良いですが、
# 　 独自に分割する手法をここでは示します。
split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)

# 分割後の名前を分かりやすく定義
train_dataset = split_datasets["train"]
val_dataset = split_datasets["test"]  # test_sizeで指定した分が "test" キーに入ります

# 4. MLM用のデータコレクター
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# 5. 各DataLoaderの作成
train_dataloader = DataLoader(
    train_dataset, 
    shuffle=True, 
    batch_size=8, 
    collate_fn=data_collator
)

val_dataloader = DataLoader(
    val_dataset, 
    shuffle=False,  # 検証時はシャッフル不要
    batch_size=8, 
    collate_fn=data_collator
)

print(f"学習データ数: {len(train_dataset)}")
print(f"検証データ数: {len(val_dataset)}")