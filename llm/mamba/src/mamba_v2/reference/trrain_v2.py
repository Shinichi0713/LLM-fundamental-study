
import torch
from transformers import AutoTokenizer

# ここが要注意。本物のMamba-2は日本語対応のトークナイザを使うとドボン
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small", use_fast=False)

from transformers import Mamba2Config, Mamba2ForCausalLM
import torch

# 130mの構造（次元数）を保ちつつ、層の数だけを削ってVRAMを節約する設定
config = Mamba2Config(
    num_hidden_layers=4,      # 軽量化
    hidden_size=512,          # 128の倍数で安定
    num_heads=8,              # エラーが求めている 8 に合わせる
    head_dim=128,             
    expand=2,
    n_groups=8,               # 明示的に指定して安全策をとる
    vocab_size=tokenizer.vocab_size
)

model = Mamba2ForCausalLM(config)
print("Tiny Mamba-2 model successfully loaded on GPU!")

from transformers import Mamba2Config, Mamba2ForCausalLM
import torch

# 130mの構造（次元数）を保ちつつ、層の数だけを削ってVRAMを節約する設定
safe_config = Mamba2Config(
    num_hidden_layers=6,      # 元の24層から削減
    hidden_size=768,          # 130m相当の次元
    num_heads=12,             # 768 * 2 / 128 = 12
    head_dim=128,             # 標準的な設定
    expand=2,
    vocab_size=tokenizer.vocab_size
)

model = Mamba2ForCausalLM(safe_config)
print("Tiny Mamba-2 model successfully loaded on GPU!")


from datasets import Dataset

# 1. 学習用の生テキストデータ
# エンジニアリングやAIに関する文章をリストとして用意します
raw_texts = [
    "Mamba is a new state space model architecture.",
    "State space models process sequences in linear time.",
    "Deep learning requires efficient hardware and algorithms.",
    "Kyoto University is known for its research in artificial intelligence.",
    "Large language models are transforming the way we write code.",
    "Computer vision allows machines to interpret visual information.",
    "Reinforcement learning is used to train autonomous agents.",
    "The transformer architecture uses attention mechanisms."
] * 50 # データを増やして学習を安定させます

# 2. Dataset オブジェクトの作成
raw_dataset = Dataset.from_dict({"text": raw_texts})

# 3. トークナイズ処理の定義
def tokenize_function(examples):
    # テキストをトークンIDに変換
    # truncation=True で最大長を超えた分をカット、padding="max_length" で長さを揃えます
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=64,
        padding="max_length"
    )

# 4. マッピング処理の実行
# batched=True で高速に処理し、不要になった元の "text" カラムを削除します
tokenized_dataset = raw_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"]
)

# 中身の確認（最初の1件を表示）
print(f"Dataset samples: {len(tokenized_dataset)}")
print(f"First sample input_ids: {tokenized_dataset[0]['input_ids'][:10]}...")

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW 

# 1. データローダーの準備（型を Long に固定）
def collate_fn(batch):
    # 各データの input_ids を Long 型のテンソルとして取得
    input_ids_list = []
    for item in batch:
        ids = item["input_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        else:
            ids = ids.to(torch.long)
        input_ids_list.append(ids)

    max_len = 64 
    padded_ids = []
    for ids in input_ids_list:
        if len(ids) >= max_len:
            new_ids = ids[:max_len]
        else:
            # パディング部分を明示的に Long 型 (dtype=torch.long) で作成
            padding = torch.zeros(max_len - len(ids), dtype=torch.long)
            new_ids = torch.cat([ids, padding])
        padded_ids.append(new_ids)
    
    input_ids = torch.stack(padded_ids)
    # labels も同様に Long 型であることを保証
    return {"input_ids": input_ids, "labels": input_ids.clone()}

train_loader = DataLoader(tokenized_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 2. 学習ループ（ここは前回と同じですが、念のため再実行）
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

print("Starting manual training loop with LongTensor...")
model.train()

for epoch in range(2):
    for step, batch in enumerate(train_loader):
        # input_ids と labels を GPU へ転送（型は Long のまま）
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

print("Training Complete!")