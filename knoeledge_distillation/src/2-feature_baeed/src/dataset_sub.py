from datasets import load_dataset
from transformers import BertTokenizer

# 1. 小さめのWikiText-2をロード
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# 2. 教師モデルと同じトークナイザーを準備
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 3. 前処理関数の定義
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# 4. データセット全体に適用
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 最初の1000件だけを使う
small_train_dataset = tokenized_datasets.select(range(1000))