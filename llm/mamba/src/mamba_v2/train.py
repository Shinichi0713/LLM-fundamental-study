import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from your_module import MambaLikeLM  # 上記クラスを別ファイルにまとめた想定

# デバイス
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# トークナイザ（MambaLikeLM のデフォルトと同じ GPT2 トークナイザ）
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# データセットのロード（wikitext-2 を例に）
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,  # 適宜調整
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
)

tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

train_loader = DataLoader(
    tokenized_dataset,
    batch_size=8,  # 適宜調整
    shuffle=True,
)

# モデル初期化
vocab_size = tokenizer.vocab_size
d_model = 512
n_layers = 6

model = MambaLikeLM(
    vocab_size=vocab_size,
    tokenizer=tokenizer,
    d_model=d_model,
    n_layers=n_layers,
).to(device)

# 最適化器と損失関数
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# デバイス
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データローダー（例）
train_loader = DataLoader(
    tokenized_dataset,
    batch_size=8,
    shuffle=True,
)

# モデル初期化
vocab_size = tokenizer.vocab_size
d_model = 512
n_layers = 6

model = MambaLikeLM(
    vocab_size=vocab_size,
    tokenizer=tokenizer,
    d_model=d_model,
    n_layers=n_layers,
).to(device)

# 最適化器と損失関数
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # 学習率を少し下げる
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 学習ループ
num_epochs = 3
max_grad_norm = 1.0

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    nan_occurred = False

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        x = input_ids[:, :-1]
        targets = input_ids[:, 1:].contiguous()

        optimizer.zero_grad()
        logits = model(x)  # [B, L-1, vocab_size]

        # nan/inf チェック
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("WARNING: logits contains nan/inf, skipping step")
            nan_occurred = True
            break

        loss = criterion(
            logits.view(-1, vocab_size),
            targets.view(-1),
        )

        if torch.isnan(loss):
            print("WARNING: loss is nan, skipping step")
            nan_occurred = True
            break

        loss.backward()

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
            norm_type=2.0
        )

        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

    if nan_occurred:
        print("nan detected during epoch, reinitializing optimizer and skipping epoch")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # optimizer 再初期化
        continue

    # epoch 終了時の平均損失
    avg_epoch_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished, Average Loss: {avg_epoch_loss:.4f}")

    # epoch 終了時にモデルを保存
    save_path = f"mamba_like_wikitext_epoch{epoch+1}.pth"
    model.save_model(save_path)  # あなたのクラスに定義済みのメソッド

# 最終モデル保存（任意）
model.save_model("mamba_like_wikitext_final.pth")

