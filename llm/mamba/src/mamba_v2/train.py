import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# ここでは GPT-2 のトークナイザを使う例（vocab_size=50257）
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = tokenizer.vocab_size

# 上で実装した MambaLM を再定義（必要に応じて別ファイルから import しても可）
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=False,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state + d_model, bias=False)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :residual.size(1)]
        x = x.transpose(1, 2)

        ssm_params = self.x_proj(x)
        B, C = ssm_params.split([self.d_state, self.d_model], dim=-1)
        A = -torch.exp(self.A_log)

        h = torch.einsum("bln,nd->bld", x, A) + torch.einsum("bln,bln->bld", B, x)
        y = torch.einsum("bld,nd->bln", h, C) + self.D * x
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = y + residual
        return y


class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, d_state, d_conv, expand=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# モデルインスタンス作成（例：Mamba-130M相当の小さめ設定）
d_model = 768
n_layer = 12
d_state = 16
d_conv = 4
expand = 2

model = MambaLM(
    vocab_size=vocab_size,
    d_model=d_model,
    n_layer=n_layer,
    d_state=d_state,
    d_conv=d_conv,
    expand=expand,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}")


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,        # 最大長を固定
        padding="max_length",  # ここを追加：最大長までパディング
    )
    return tokenized

# wikitext-2-raw-v1 をロード
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# トークナイズ
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# 学習用データのみ使用
train_dataset = tokenized_datasets["train"]

# データローダー作成
train_loader = DataLoader(
    train_dataset,
    batch_size=4,  # ColabのGPUメモリに合わせて調整
    shuffle=True,
)

from torch.optim import AdamW

# オプティマイザと損失関数
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 学習エポック数（Colabの制限に合わせて調整）
num_epochs = 3

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        # 入力トークン
        input_ids = batch["input_ids"].to(device)  # (B, L)

        # 次トークンをターゲットとして使用（言語モデリング）
        # 入力: input_ids[:, :-1], ターゲット: input_ids[:, 1:]
        if input_ids.size(1) < 2:
            continue  # 長さ1のシーケンスはスキップ

        inputs = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()

        # 順伝播
        logits = model(inputs)  # (B, L-1, vocab_size)

        # 損失計算
        loss = criterion(
            logits.view(-1, vocab_size),
            targets.view(-1),
        )

        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished, Average Loss: {avg_loss:.4f}")


def generate_text(model, tokenizer, prompt, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)  # (1, L, vocab_size)
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# 例
prompt = "The history of artificial intelligence"
generated = generate_text(model, tokenizer, prompt)
print("Generated:", generated)