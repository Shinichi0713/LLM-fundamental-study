from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

# 1. データのロード (WikiText-2)
raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

# 2. トークナイザーの準備 (教師モデルに合わせる)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    # 空行を除外するための処理
    texts = [t for t in examples["text"] if len(t) > 0 and not t.isspace()]
    return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

tokenized_datasets = raw_datasets.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# 3. MLM用のデータコレクター (自動で[MASK]を作成してくれる)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# 4. DataLoaderの作成
train_dataloader = DataLoader(
    tokenized_datasets["train"], 
    shuffle=True, 
    batch_size=8, 
    collate_fn=data_collator
)