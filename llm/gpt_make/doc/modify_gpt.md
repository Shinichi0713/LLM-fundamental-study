これは **バグではなく仕様どおりの挙動** です。
結論から言うと、

> **GPT-2 トークナイザで日本語 C4(ja) を扱うと、デコード表示は高確率で文字化けします**

ただし重要なのは👇
**学習自体が壊れているわけではありません。**

以下、
1️⃣ なぜ起きるか
2️⃣ 学習上の影響
3️⃣ 正しい対処法（目的別）
を順に説明します。

---

## 1. なぜ文字化けするのか（本質）

### GPT-2 tokenizer の正体

GPT-2 は **Byte-level BPE** です。

* 入力は **UTF-8のバイト列**
* 「文字」ではなく「バイト」をトークン化
* 英語前提で設計

### 日本語の場合に起きていること

日本語は UTF-8 で **1文字 = 3バイト** が多いです。

例：

```
生  →  e7 94 9f
```

GPT-2 tokenizer はこれを

```
[e7] [94] [9f]
```

のような **意味を持たないバイト断片** として扱います。

その結果：

* トークン列自体は正しい
* しかし途中で切って `decode()` すると
  → **不完全なUTF-8列**
  → `�`（replacement character）が出る

---

## 2. 重要：学習は壊れていないのか？

### ✅ 学習は壊れていません

LLMの学習では：

* **decode結果の可読性は不要**
* モデルは一貫したトークン列を学習

GPT-2 tokenizer × 日本語は：

| 観点                 | 状態        |
| ------------------ | --------- |
| forward / backward | 正常        |
| loss               | 正常        |
| 勾配                 | 正常        |
| 表示                 | ❌ 人間に読めない |

👉 **「人間が読みにくい」だけ**

---

## 3. では何が問題になるか？

### 実務・研究的には以下が問題

1. **語彙効率が極端に悪い**

   * 1文字 = 3〜4トークン
2. **SEQ_LENを大量に消費**
3. **文法・形態素構造を学習しづらい**
4. **収束が遅い**

👉 日本語GPTとしては **性能が出にくい**

---

## 4. 正しい対処法（目的別）

### 🔹 目的A：

**「とりあえずGPTの仕組み理解・実験」**

➡ 今のままでOK
文字化けは **無視して良い**

```python
# 表示用に skip_special_tokens を付ける程度
tokenizer.decode(batch['input_ids'][0][:20], skip_special_tokens=True)
```

※ 完全には直りません

---

### 🔹 目的B：

**「日本語GPTをまともに学習したい」**

➡ **日本語向け tokenizer に変更する必要あり**

#### 推奨①：SentencePiece（日本語向け）

```python
MODEL_NAME = "rinna/japanese-gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)
```

✔ 日本語サブワード
✔ 文字化けしない
✔ GPT構造と相性良い

---

#### 推奨②：LLaMA系（Unigram）

```python
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

✔ 最新構成
✔ RoPE前提
✔ 日本語性能が高い

---

### 🔹 目的C：

**「自作GPT × 自作Tokenizer」**

最も理想的です。

#### 手順概要

1. SentencePiece で tokenizer 学習
2. vocab_size を GPT に合わせる
3. decode 表示も正常

```bash
spm_train \
 --input=ja_corpus.txt \
 --model_prefix=ja_sp \
 --vocab_size=32000 \
 --character_coverage=0.9995 \
 --model_type=unigram
```

---

## 5. 今回のログの正体を解説

### Batch 1（正常に見える理由）

```
生八つ橋のタグまとめ | エキサイト
```

→ 偶然 UTF-8 境界が揃った

---

### Batch 2, 3（文字化け）

```
��！私は大好きです！ま�
```

→ SEQ_LEN=48 で **途中切断**
→ UTF-8 不完全

---

## 6. まとめ（重要）

### 今起きていること

> GPT-2 tokenizer は
> **日本語を「意味のある文字」として扱っていない**

### 結論

| 状況            | 評価   |
| ------------- | ---- |
| 学習が壊れている      | ❌    |
| tokenizerが不適切 | ✅    |
| 修正が必要か        | 目的次第 |

---

## 次にやると良いこと（おすすめ順）

1️⃣ tokenizer を日本語向けに変更
2️⃣ SEQ_LEN を実質文字長ベースで再設計
3️⃣ vocab_size を 32k〜50k に
4️⃣ 既に実装済みの **RoPE + SwiGLU + RMSNorm** を活かす

ここまでやると
**「現代的な日本語GPT」** になります。



了解です。
では **あなたの GPT 実装を「現代LLM標準構成」へ段階的に進化**させます。

対応内容は以下の3点です。

---

## 今回対応する改善点（全体像）

| 改善                       | 効果            |
| ------------------------ | ------------- |
| **LayerNorm → RMSNorm**  | 学習安定・高速化      |
| **FFN → SwiGLU**         | 表現力・収束性向上     |
| **Attention scaling 調整** | 深層化・MoE併用時の安定 |

これらは **LLaMA / PaLM / Mistral / Qwen 系の中核設計**です。

---

# 1. RMSNorm の実装（LayerNorm置換）

## 1.1 RMSNormとは（要点）

LayerNorm：
[
\frac{x - \mu}{\sigma}
]

RMSNorm：
[
\frac{x}{\sqrt{\text{mean}(x^2)}}
]

* **平均との差分を取らない**
* 計算軽量
* 大規模モデルで安定

---

## 1.2 RMSNorm 実装

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x
```

---

## 1.3 DecoderBlock で LayerNorm を置換

```python
self.norm1 = RMSNorm(embed_dim)
self.norm2 = RMSNorm(embed_dim)
```

---

# 2. FFN → SwiGLU への変更

## 2.1 SwiGLUとは（直感）

従来FFN：

```text
Linear → GELU → Linear
```

SwiGLU：

```text
(xW1 ⊙ SiLU(xW2))W3
```

* ゲート構造
* 勾配が通りやすい
* **表現力が大幅に向上**

---

## 2.2 SwiGLU 実装

```python
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

---

## 2.3 DecoderBlock に組み込み

```python
if use_moe:
    self.ffn_or_moe = MoELayer(embed_dim, num_experts, top_k, expert_hidden_dim=ffn_hidden_dim)
else:
    self.ffn_or_moe = SwiGLU(embed_dim, ffn_hidden_dim)
```

※ MoE の Expert 内部も同様に置き換えるとさらに良いです（後述）

---

# 3. Attention Scaling の調整

## 3.1 標準の問題点

```python
scores = QKᵀ / sqrt(head_dim)
```

* 深層化
* MoE併用
* 長文

で **Attentionが過度に尖る**

---

## 3.2 改善案①：スケールを固定値で緩和

### SelfAttention.**init** に追加

```python
self.scale = self.head_dim ** -0.5
```

### forward で使用

```python
scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
```

（※除算ではなく乗算が一般的）

---

## 3.3 改善案②（よりモダン）：学習可能スケール

```python
self.scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5))
```

👉 MoE + 深層で特に安定します。

---

# 4. MoE Expert も SwiGLU 化（推奨）

```python
class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

---

# 5. DecoderBlock 最終形（要点）

```python
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim,
                 use_moe=False, num_experts=None, top_k=None):
        super().__init__()

        self.norm1 = RMSNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm2 = RMSNorm(embed_dim)

        self.use_moe = use_moe
        if use_moe:
            self.ffn_or_moe = MoELayer(embed_dim, num_experts, top_k, ffn_hidden_dim)
        else:
            self.ffn_or_moe = SwiGLU(embed_dim, ffn_hidden_dim)

    def forward(self, x, mask):
        x = x + self.attention(self.norm1(x), mask)

        if self.use_moe:
            ffn_out, moe_loss = self.ffn_or_moe(self.norm2(x))
            x = x + ffn_out
            return x, moe_loss
        else:
            x = x + self.ffn_or_moe(self.norm2(x))
            return x, None
```

---

# 6. ここまでで何が変わったか（設計的まとめ）

あなたの GPT は今：

* ✅ RoPE
* ✅ RMSNorm
* ✅ SwiGLU
* ✅ 改良Attention scaling
* ✅ MoE対応

→ **LLaMA系Decoderの思想とほぼ同一**

---

## 次にやると「完成形」に近づきます

優先度順：

1. **KV Cache（推論速度10倍）**
2. **FlashAttention**
3. **Grouped Query Attention (GQA)**
4. **RoPE extrapolation（NTK / YaRN）**
5. **Weight tying（Embedding ↔ LM Head）**

どこまで行きますか？

