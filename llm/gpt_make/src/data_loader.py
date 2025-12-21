from datasets import load_dataset
from transformers import AutoTokenizer
import torch

# 1. 設定
MODEL_NAME = "rinna/japanese-gpt-1b"
DATASET_NAME = "allenai/c4"
SEQ_LEN = 128  # 今回は短めに設定

# 2. トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. Japanese mC4 をストリーミングモードで読み込み
# split="train" で訓練データ、languages=["ja"] で日本語を指定
dataset = load_dataset(DATASET_NAME, "ja", split="train", streaming=True)

def tokenize_function(examples):
    # テキストをトークナイズ
    return tokenizer(examples["text"], truncation=False)

def group_texts(examples):
    # 全てのトークンを連結し、SEQ_LEN ごとに分割する（効率的な学習のため）
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # SEQ_LEN の倍数になるように切り捨て
    total_length = (total_length // SEQ_LEN) * SEQ_LEN
    
    result = {
        k: [t[i : i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)]
        for k, t in concatenated_examples.items()
    }
    # 自己回帰モデルの正解（labels）は input_ids と同じ（内部でずらされる）
    result["labels"] = result["input_ids"].copy()
    return result

# 4. 前処理パイプライン
# map関数で逐次処理を適用
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text", "timestamp", "url"]
)
lm_dataset = tokenized_dataset.map(group_texts, batched=True)

# 5. データの取り出しテスト
print("Fetching processed data...")
for i, batch in enumerate(lm_dataset.take(3)):
    print(f"\n--- Batch {i} ---")
    print(f"Decoded text: {tokenizer.decode(batch['input_ids'])[:100]}...")
    print(f"Shape: {len(batch['input_ids'])} tokens")