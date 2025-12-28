import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

TASKS = ["arith", "logic", "code", "nl"]

class SimpleTokenizer:
    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+=?., ")
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        self.vocab = [self.pad_token, self.unk_token] + chars
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text, max_len):
        ids = [self.stoi.get(c, self.stoi[self.unk_token]) for c in text]
        ids = ids[:max_len]
        pad_len = max_len - len(ids)
        ids += [self.stoi[self.pad_token]] * pad_len
        return ids

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i != self.stoi[self.pad_token])


class MixedTaskDataset(Dataset):
    def __init__(self, n_samples=10000, vocab=None, max_len=64):
        self.n = n_samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return self.n

    def _arith(self):
        x = random.randint(1, 10)
        y = random.randint(1, 10)
        q = f"{x} + {y} = ?"
        a = str(x + y)
        return q, a

    def _logic(self):
        q = "All A are B. C is A. Is C B?"
        a = "Yes"
        return q, a

    def _code(self):
        q = "What is sum of even numbers less than 6?"
        a = "6"
        return q, a

    def _nl(self):
        q = "Taro is older than Hanako. Who is older?"
        a = "Taro"
        return q, a

    def __getitem__(self, idx):
        task = random.choice(TASKS)
        if task == "arith":
            q, a = self._arith()
        elif task == "logic":
            q, a = self._logic()
        elif task == "code":
            q, a = self._code()
        else:
            q, a = self._nl()

        return {
            "input": q,
            "target": a,
            "task": task
        }



class MoE(nn.Module):
    def __init__(self, d_model, d_hidden, n_experts=4):
        super().__init__()
        self.n_experts = n_experts

        self.router = nn.Linear(d_model, n_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_model)
            )
            for _ in range(n_experts)
        ])

    def forward(self, x):
        B, T, D = x.shape

        logits = self.router(x)                 # [B, T, E]
        probs = F.softmax(logits, dim=-1)

        expert_idx = probs.argmax(dim=-1)       # [B, T]

        out = torch.zeros_like(x)

        for e in range(self.n_experts):
            mask = expert_idx == e
            if mask.any():
                out[mask] = self.experts[e](x[mask])

        return out, probs, expert_idx


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_hidden, use_moe=False, n_experts=4):
        super().__init__()
        self.use_moe = use_moe

        self.attn = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if use_moe:
            self.ffn = MoE(d_model, d_hidden, n_experts)
        else:
            self.ffn = None
            self.dense_ffn = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_model)
            )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        if self.use_moe:
            ffn_out, router_probs, expert_idx = self.ffn(x)
        else:
            ffn_out = self.dense_ffn(x)
            router_probs, expert_idx = None, None

        x = self.norm2(x + ffn_out)
        return x, router_probs, expert_idx



class LitModel(pl.LightningModule):
    def __init__(self, vocab_size=128, d_model=128, use_moe=False):
        super().__init__()
        self.save_hyperparameters()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.block = TransformerBlock(
            d_model=d_model,
            d_hidden=256,
            use_moe=use_moe,
            n_experts=4
        )
        self.head = nn.Linear(d_model, vocab_size)

        # ★ expert_idx 保存用
        self.expert_idx_buffer = []

    def forward(self, x):
        x = self.embed(x)
        x, router_probs, expert_idx = self.block(x)
        logits = self.head(x[:, -1])
        return logits, router_probs, expert_idx

    def training_step(self, batch, batch_idx):
        x = batch["input_ids"]
        y = batch["labels"]

        logits, router_probs, expert_idx = self(x)
        loss = F.cross_entropy(logits, y)

        if router_probs is not None:
            entropy = -(router_probs * torch.log(router_probs + 1e-9)).sum(-1).mean()
            self.log("router_entropy", entropy)

        self.log("loss", loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-4)


    def validation_step(self, batch, batch_idx):
        x = batch["input_ids"]
        y = batch["labels"]

        logits, router_probs, expert_idx = self(x)
        loss = F.cross_entropy(logits, y)

        # MoE のときのみ収集
        if expert_idx is not None:
            # [B, T] → flatten
            self.expert_idx_buffer.append(
                expert_idx.detach().flatten().cpu()
            )

        self.log("val_loss", loss, prog_bar=True)


    def on_validation_epoch_end(self):
        if len(self.expert_idx_buffer) == 0:
            return

        expert_idx_all = torch.cat(self.expert_idx_buffer)  # [N_tokens]

        # TensorBoard Histogram
        self.logger.experiment.add_histogram(
            tag="MoE/expert_idx",
            values=expert_idx_all,
            global_step=self.current_epoch
        )

        # バッファクリア
        self.expert_idx_buffer.clear()

class MixedTaskDataset(Dataset):
    def __init__(self, tokenizer, n_samples=10000, max_len=64):
        self.tokenizer = tokenizer
        self.n = n_samples
        self.max_len = max_len

    def __len__(self):
        return self.n

    def _arith(self):
        x = random.randint(1, 9)
        y = random.randint(1, 9)
        q = f"{x} + {y} = ?"
        a = str(x + y)
        return q, a, "arith"

    def _logic(self):
        q = "All A are B. C is A. Is C B?"
        a = "Yes"
        return q, a, "logic"

    def _code(self):
        q = "What is sum of even numbers less than 6?"
        a = "6"
        return q, a, "code"

    def _nl(self):
        q = "Taro is older than Hanako. Who is older?"
        a = "Taro"
        return q, a, "nl"

    def __getitem__(self, idx):
        task = random.choice(TASKS)

        if task == "arith":
            q, a, task = self._arith()
        elif task == "logic":
            q, a, task = self._logic()
        elif task == "code":
            q, a, task = self._code()
        else:
            q, a, task = self._nl()

        input_ids = self.tokenizer.encode(q, self.max_len)
        label_ids = self.tokenizer.encode(a, max_len=8)

        # 次トークン予測ではなく分類扱い（最後の1文字）
        label = label_ids[0]

        return {
            "input_ids": input_ids,
            "label": label,
            "task": task
        }

def collate_fn(batch):
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    tasks = [b["task"] for b in batch]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "task": tasks
    }


tokenizer = SimpleTokenizer()

train_dataset = MixedTaskDataset(
    tokenizer=tokenizer,
    n_samples=20000,
    max_len=64
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

val_dataset = MixedTaskDataset(
    tokenizer=tokenizer,
    n_samples=2000,
    max_len=64
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)


model = LitModel(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    use_moe=True
)

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    log_every_n_steps=50
)

trainer.fit(model, train_loader, val_loader)



import matplotlib.pyplot as plt
import numpy as np

def plot_token_expert_heatmap(router_probs, tokens):
    """
    router_probs: [T, E]
    tokens: List[str]
    """
    probs = router_probs.detach().cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.imshow(probs.T, aspect="auto")
    plt.colorbar(label="Routing probability")
    plt.yticks(range(probs.shape[1]), [f"Expert {i}" for i in range(probs.shape[1])])
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.xlabel("Token")
    plt.ylabel("Expert")
    plt.title("Token-level Expert Routing")
    plt.tight_layout()
    plt.show()







