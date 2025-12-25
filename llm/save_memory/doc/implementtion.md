以下では **「KVキャッシュの有無による差を“体感”できる」**ことを主目的に、
**再現性が高く・実装量が少なく・差が明確に出る**実験を体系立てて提案します。

単なる速度比較ではなく、

* 何が無駄になっているか
* どこがボトルネックか
* なぜスケールで差が爆発するか

が理解できる構成にしています。

---

# 実験の全体像（結論先出し）

| 実験  | 体感できる差                  |
| --- | ----------------------- |
| 実験1 | **計算量の差（O(T²) → O(T))** |
| 実験2 | **Wall-clock time の差**  |
| 実験3 | **GPUメモリ使用量の差**         |
| 実験4 | **トークン長スケール時の破綻**       |

---

# 実験1：Attention 計算回数の可視化（最重要）

## 目的

> **「同じことを何回やっているか」**を数で理解する

---

## 条件

* Decoder-only Transformer
* 1トークンずつ生成
* 比較対象：

  * ❌ KVキャッシュなし
  * ✅ KVキャッシュあり

---

## 計測指標

```text
Attention score 計算回数
= QK^T の総要素数
```

---

## 期待される理論値

| トークン数 T | KVなし    | KVあり |
| ------- | ------- | ---- |
| 10      | 55      | 10   |
| 100     | 5050    | 100  |
| 1000    | 500,500 | 1000 |

👉 **O(T²) vs O(T)**

---

## 実装アイデア（最小）

```python
counter = 0

def count_matmul(q, k):
    global counter
    counter += q.size(-2) * k.size(-2)
    return torch.matmul(q, k.transpose(-2, -1))
```

* KVなし：毎回 `k.size = t`
* KVあり：常に `q.size = 1`

**→ 数字で衝撃を受ける**

---

# 実験2：生成時間の実測（体感）

## 目的

> **「長文生成で“待てなくなる”体験」**

---

## 条件

* モデル：小型GPT（2–4層でOK）
* 生成長：256 / 512 / 1024 tokens
* 同一入力・同一環境

---

## 計測コード例

```python
import time

start = time.time()
for _ in range(gen_len):
    logits = model(input_ids)  # or model(..., kv_cache)
end = time.time()

print(f"Generation time: {end - start:.2f}s")
```

---

## 期待結果（CPUでも差が出る）

| 長さ   | KVなし | KVあり |
| ---- | ---- | ---- |
| 256  | 〇    | ◎    |
| 512  | △    | ◎    |
| 1024 | ❌    | 〇    |

👉 **「途中から露骨に遅くなる」**

---

# 実験3：GPUメモリ使用量の比較（視覚的）

## 目的

> **なぜ「長文＝OOM」になるのかを理解**

---

## 測定

```python
torch.cuda.reset_peak_memory_stats()
...
print(torch.cuda.max_memory_allocated() / 1024**2, "MB")
```

---

## 観察ポイント

* KVなし：

  * 毎ステップで巨大な QK^T 行列
* KVあり：

  * KVは線形増加
  * 中間テンソルが小さい

---

## 体感

> 「KVキャッシュがないと
> **計算より先にメモリが死ぬ**」

---

# 実験4：スケール破綻点の確認（決定打）

## 目的

> **「この設計は実用にならない」ラインを見る**

---

## 手順

* 生成長を段階的に増やす
* どこで破綻するかを比較

---

## 結果イメージ

| トークン数 | KVなし | KVあり |
| ----- | ---- | ---- |
| 512   | OK   | OK   |
| 1024  | 遅い   | OK   |
| 2048  | ほぼ停止 | OK   |
| 4096  | OOM  | △    |

👉 **LLMがKVキャッシュ前提で設計されている理由が腑に落ちる**

---

# 実験5（応用）：プロファイラで「無駄」を見る

## PyTorch profiler

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True
) as prof:
    model(...)
print(prof.key_averages().table())
```

---

## 見えるもの

* ❌ KVなし：

  * matmul が指数的に増殖
* ✅ KVあり：

  * matmul が一定

---

# 実験まとめ（理解の段階）

| 段階   | 理解         |
| ---- | ---------- |
| 数    | 計算量が違いすぎる  |
| 時間   | 実用可否が分かれる  |
| メモリ  | なぜOOMするか   |
| スケール | KVキャッシュは必須 |

---

# 一文で言うと

> **KVキャッシュは
> 「推論を“逐次計算”から“状態遷移”に変える仕組み」**

---

## KV offloading（CPU / NVMe）

以下では **KVキャッシュを GPU → CPU → NVMe にオフロードする最小構成の PyTorch 実装例**を示します。
**理解重視・実験用**のコードであり、vLLM 等の本番実装を簡略化したものです。

---

# 0. 前提と設計方針

## 何をするか

* **GPU**：直近トークンの計算
* **CPU**：過去 KV キャッシュの保持
* **NVMe**：さらに古い KV をディスク退避（numpy memmap）

## 割り切り

* 単一レイヤ・単一ヘッド（拡張は容易）
* Decoder-only / 自己回帰生成
* 性能最適化より **仕組みの理解** を優先

---

# 1. KV Cache Manager（GPU / CPU / NVMe）

```python
import torch
import numpy as np
import os

class KVCacheOffloader:
    def __init__(self, head_dim, max_tokens, device="cuda",
                 cpu_limit=256, nvme_path="kv_cache"):
        """
        cpu_limit : CPUに保持する最大トークン数
        """
        self.device = device
        self.head_dim = head_dim
        self.cpu_limit = cpu_limit

        self.gpu_k = []
        self.gpu_v = []

        self.cpu_k = []
        self.cpu_v = []

        os.makedirs(nvme_path, exist_ok=True)
        self.nvme_k = np.memmap(
            f"{nvme_path}/k.dat",
            dtype=np.float32,
            mode="w+",
            shape=(max_tokens, head_dim),
        )
        self.nvme_v = np.memmap(
            f"{nvme_path}/v.dat",
            dtype=np.float32,
            mode="w+",
            shape=(max_tokens, head_dim),
        )

        self.nvme_ptr = 0

    def append(self, k, v):
        """
        k, v: (1, head_dim) on GPU
        """
        self.gpu_k.append(k)
        self.gpu_v.append(v)

        # GPU → CPU オフロード
        if len(self.gpu_k) > 1:
            old_k = self.gpu_k.pop(0).cpu()
            old_v = self.gpu_v.pop(0).cpu()
            self.cpu_k.append(old_k)
            self.cpu_v.append(old_v)

        # CPU → NVMe オフロード
        if len(self.cpu_k) > self.cpu_limit:
            ck = self.cpu_k.pop(0).numpy()
            cv = self.cpu_v.pop(0).numpy()
            self.nvme_k[self.nvme_ptr] = ck
            self.nvme_v[self.nvme_ptr] = cv
            self.nvme_ptr += 1

    def get_all_kv(self):
        """
        全KVをGPUに集約（Attention用）
        """
        kv = []

        # NVMe → CPU → GPU
        if self.nvme_ptr > 0:
            nk = torch.from_numpy(self.nvme_k[:self.nvme_ptr]).to(self.device)
            nv = torch.from_numpy(self.nvme_v[:self.nvme_ptr]).to(self.device)
            kv.append((nk, nv))

        if self.cpu_k:
            ck = torch.cat(self.cpu_k).to(self.device)
            cv = torch.cat(self.cpu_v).to(self.device)
            kv.append((ck, cv))

        if self.gpu_k:
            gk = torch.cat(self.gpu_k)
            gv = torch.cat(self.gpu_v)
            kv.append((gk, gv))

        K = torch.cat([x[0] for x in kv], dim=0)
        V = torch.cat([x[1] for x in kv], dim=0)
        return K, V
```

---

# 2. KV Offloading 対応 Attention

```python
class SimpleAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, kv_cache: KVCacheOffloader):
        """
        x: (1, dim)
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        kv_cache.append(k, v)

        K, V = kv_cache.get_all_kv()  # 全履歴取得

        attn = (q @ K.T) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = attn @ V
        return out
```

---

# 3. 生成ループ（Offloading体感用）

```python
def generate(model, steps=512, dim=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    kv_cache = KVCacheOffloader(
        head_dim=dim,
        max_tokens=steps,
        device=device,
        cpu_limit=64,
    )

    x = torch.zeros(1, dim, device=device)

    for i in range(steps):
        with torch.no_grad():
            x = model(x, kv_cache)

        if i % 50 == 0:
            print(f"step {i}, KV total = {i+1}")

    print("Generation done")
```

---

# 4. 実行

```python
attn = SimpleAttention(dim=64)
generate(attn, steps=300)
```

---

# 5. この実装で「体感できること」

## ✔ 見える変化

| 項目     | 体感        |
| ------ | --------- |
| GPUメモリ | ほぼ一定      |
| CPUメモリ | 緩やかに増加    |
| NVMe   | I/O発生（遅延） |
| 長文生成   | OOMしない    |

---

# 6. 実運用（vLLM / llama.cpp）との対応関係

| 本実装         | 実運用          |
| ----------- | ------------ |
| Python list | Paged KV     |
| memmap      | NVMe paging  |
| 全取得         | Block table  |
| 同期I/O       | 非同期 prefetch |

---

# 7. 重要な理解ポイント

> **KV offloading は「速くする技術」ではない
> →「破綻しないための技術」**



