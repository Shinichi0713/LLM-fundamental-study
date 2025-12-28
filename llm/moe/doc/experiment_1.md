今回はMoEについて有無による性能の差をなんとか知ることが出来ないかと考えました。

以下の実験で性能を確認していきます。


## 1. 実験設計方針

実験を始める前にMoEの優位性が出る条件について整理します。

1. **入力分布が明確に多峰性（heterogeneous）**
2. **サブタスク間で必要な内部表現・推論様式が大きく異なる**
3. **1サンプルあたりでは単純だが、全体としては混在している**
4. Denseモデルでは

   * パラメータ共有による**表現干渉（interference）**
   * 容量不足 or 勾配競合
     が発生する

→ MoEは「**条件付き計算＋専門化**」によってこれを回避できます。

ということで問題の範囲を多様にして通常のDenseモデルとの性能差を確認していこうと思います。


## 2. 例題①：推論様式が異なる数理・論理・言語混在QA
次にモデルに解かせる問題を考えます。

### 問題設定

以下の4タイプの問題が**ランダムに混在**して出題されるようにします。
これでモデルにとっては異なる問題が出題されるという状況を作り出します。

| タイプ | 内容             | 必要能力         |
| ------ | ---------------- | ---------------- |
| A      | 逐次的な数値計算 | 算術・状態更新   |
| B      | 論理パズル       | 記号推論         |
| C      | プログラム理解   | 構文・制御フロー |
| D      | 自然言語読解     | 意味表現         |

### サンプル問題

#### A: 数値状態推論

```
x=3から開始する。
以下を順に実行せよ。
1. xを2倍する
2. xに5を足す
3. xを3で割った余りを求める
最終結果は？
```

#### B: 論理推論

```
AはBより背が高い。
CはAより背が低い。
DはBより背が高い。
最も背が高い可能性があるのは誰か？
```

#### C: コード理解

```python
def f(x):
    y = 0
    for i in range(x):
        if i % 2 == 0:
            y += i
    return y

f(6)の戻り値は？
```

#### D: 言語読解

```
太郎は花子より年上である。
花子は次郎より年下である。
この情報から必ず言えることは何か？
```


### なぜMoEが有利か

* **Denseモデル**

  * 1つの内部表現で
    数値計算 / 論理 / 構文 / 意味
    をすべて処理 → 勾配干渉
* **MoEモデル**

  * Expert 1: 算術・状態遷移
  * Expert 2: 論理関係
  * Expert 3: コード・制御構造
  * Expert 4: 言語意味
  * Routerが入力特徴（token分布・構文）で分岐

MoEモデルは問題によって対応を変えることで状況に応じた回答力がつくと期待されます。
結果、**全パラメータ数を揃えても、MoEの方が正答率が高くなる**のではと想定されます。


## 3. 例題②：タスク切替を伴う逐次入力（MoEが最も差を出しやすい）

### 問題設定

1つの入力シーケンス内で**タスクが切り替わる**。

```
Q1: 17 + 24 はいくつか？
Q2: 以下の文の論理的誤りを指摘せよ。
「すべての鳥は飛べる。ペンギンは鳥である。よってペンギンは飛べる。」
Q3: 次のコードの出力は？
print(sum([i for i in range(10) if i%3==0]))
```

### 評価方法

* **全問正解率**
* **後半問題での精度劣化**


## 4. 実験コードの実装

ここまで考えた構成で実装していきます。
モデルのラッパーとしてpytorch_lightning、評価はtensor boardを使います。

```python
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

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

dense_model = LitModel(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    use_moe=False
)
moe_model = LitModel(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    use_moe=True
)

dense_logger = TensorBoardLogger(
    save_dir="/content/tb_logs",
    name="dense"
)
moe_logger = TensorBoardLogger(
    save_dir="/content/tb_logs",
    name="moe"
)

def make_trainer(logger):
    return pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=1,
        logger=logger,
        log_every_n_steps=50
    )

# Dense
dense_trainer = make_trainer(dense_logger)
dense_trainer.fit(dense_model, train_loader, val_loader)

# MoE
moe_trainer = make_trainer(moe_logger)
moe_trainer.fit(moe_model, train_loader, val_loader)
```


__結果__

学習ロスと検証ロスの推移を比較します。
縦軸がロス値、横軸は学習のエポック数です。

結果、MoEの方が早く収束することが確認されました。

![1766880278379](image/experiment_1/1766880278379.png)

![1766880252534](image/experiment_1/1766880252534.png)


肝心の正解率ですが。

両手法とも1.0=全問正解。。。
少し問題が簡単だったかもしれません。。。

![1766880421808](image/experiment_1/1766880421808.png)

見たかった後半の精度劣化も確認できませんでした。。。

次回は少し難しい問題で実験します。


