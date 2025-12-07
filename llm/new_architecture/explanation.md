# Modern BERTの工夫点

「 **Modern BERT** 」とは、初代 BERT（2018年）以降に登場した改良版 BERT 系モデルに共通する“最新の設計上の工夫”を指します。

研究の進展により、性能・学習効率・メモリ効率などが大幅に改善されています。

---

## 🌟 Modern BERT の主な工夫点

### 1️⃣ 相対位置埋め込み（Relative Positional Encoding）

従来のBERTは、固定の「絶対位置埋め込み（Sinusoidal PE）」を使用していました。

Modern BERTでは「 **相対位置表現（Relative Position Encoding / RoPE など）** 」が使われます。

* 目的：単語の**相対的な距離**を表現し、長文や文の再配置にも強くする
* 実装例：
  * **DeBERTa** → Relative + Disentangled PE
  * **RoFormer** → Rotary Positional Embedding（RoPE）

🧠効果：

「文の途中を入れ替えても意味が保たれる」「長文でも情報が減衰しにくい」

---

### 2️⃣ disentangled attention（分離アテンション）

（例： **DeBERTa** ）

* 従来：トークン埋め込み + 位置埋め込みを**単純加算**
* 改良：内容(content) と位置(position) の情報を**別々に扱う**
  ```
  Attention = Q_content × K_content^T + Q_content × K_position^T + Q_position × K_content^T
  ```
* 意味：単語の意味と位置関係を分離して処理することで、より豊かな文脈理解が可能。

🧠効果：

BERT-baseより小さいモデルでも同等性能を達成。

---

### 3️⃣ Pre-LayerNorm 構造

（例： **RoBERTa, DeBERTaV3, ModernBERT** ）

* 従来：Post-LayerNorm（Transformer block の出力後にLN）
* 改良：Pre-LayerNorm（ブロックの入力前にLN）に変更

🧠効果：

* 学習が安定する
* 高学習率でも発散しにくい
* より深い層まで安定して学習可能

---

### 4️⃣ 高効率化（パラメータシェア・軽量Attention）

（例： **ALBERT, MobileBERT, DistilBERT** ）

| 手法                 | 主な工夫                                | 効果                           |
| -------------------- | --------------------------------------- | ------------------------------ |
| **ALBERT**     | 層ごとの重み共有 + Factorized Embedding | パラメータ数を1/10以下に削減   |
| **MobileBERT** | Bottleneck構造 + Inverted Residual      | モバイル向け高効率             |
| **DistilBERT** | 知識蒸留                                | モデルを半分以下のサイズに圧縮 |

---

### 5️⃣ 学習データ・目的の改善

* **RoBERTa** : NSP（Next Sentence Prediction）を削除し、データ量を10倍に。
* **DeBERTaV3** : Masked LM ではなく、**MLM + replaced token detection (RTD)** の組み合わせを使用。
* **ModernBERT (Google 2024)** :
* コード + Web + 書籍など多様なコーパスで訓練
* 高速学習に適した **FlashAttention / XPos / RMSNorm** を採用

---

### 6️⃣ 高速化テクニック

* **FlashAttention** : GPUでアテンションを直接ストリーム計算し、高速かつ省メモリ化
* **RMSNorm** : LayerNormの簡略版（平方平均を使用）で軽量化
* **XPos** : 長文対応の拡張RoPE（相対位置のスケーリングを調整）

---

## 🧩 まとめ

工夫点は位置表現、RMSNormによる正規化、Disentangled Attention、FlashAttention

| 改良ポイント     | 技術                     | 効果                             |
| ---------------- | ------------------------ | -------------------------------- |
| 位置表現         | RoPE / XPos / 相対PE     | 長文に強く、文構造を理解しやすい |
| 正規化           | Pre-LN / RMSNorm         | 学習安定性・高速化               |
| アテンション構造 | Disentangled / Efficient | 精度向上・軽量化                 |
| 学習方式         | RoBERTa-style / RTD      | 汎化性能向上                     |
| 実装最適化       | FlashAttention           | GPUメモリ削減・高速化            |

wikipediaでMLM

![1762664182175](image/explanation/1762664182175.png)

以下は **RoPE を取り入れた Hybrid (Local + Global) Sparse Attention** の **実行可能な PyTorch 実装コード**です。
特徴：

* RoPE（Rotary Positional Embedding）を Q/K に適用して相対位置情報を導入します。
* Local (sliding window) Attention は `unfold` ベースで高速に抽出します。
* Global tokens（`global_mask` が True の位置）は全トークンと相互 attention します。
* 出力として `(out, full_attn)` を返し、`full_attn` は可視化用の擬似フル注意行列 `(B, H, T, T)` です（※可視化目的のみ。Tが大きいとメモリ高）。

コピペで実行できるようテストスニペットも付けました。

> 注意：`head_dim` は偶数（`dim % num_heads == 0` かつ `(dim/num_heads) % 2 == 0`）である必要があります（RoPE の偶数分割のため）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities: unfold_kv (same as earlier)
# -----------------------------
def unfold_kv(x: torch.Tensor, kernel_size: int, padding: int = 0):
    """
    x: (B, H, T, D)
    returns: (B, H, T, kernel_size, D)
    """
    B, H, T, D = x.shape
    x_img = x.permute(0, 1, 3, 2).reshape(B * H, D, 1, T)
    x_unf = F.unfold(x_img, kernel_size=(1, kernel_size), padding=(0, padding), stride=(1, 1))
    x_unf = x_unf.view(B * H, D, kernel_size, T)
    x_unf = x_unf.permute(0, 3, 2, 1).reshape(B, H, T, kernel_size, D)
    return x_unf

# -----------------------------
# RoPE helpers
# -----------------------------
def build_rope_cache(seq_len: int, dim: int, device=None, dtype=torch.float32):
    """
    Build cos and sin caches for RoPE.
    Returns:
      cos: (seq_len, dim//2)
      sin: (seq_len, dim//2)
    Note: dim must be even (we treat pairs).
    """
    assert dim % 2 == 0, "RoPE head dim must be even"
    half = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=dtype, device=device) / half))
    positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)  # (seq_len,1)
    angles = positions * inv_freq.unsqueeze(0)  # (seq_len, half)
    cos = torch.cos(angles)  # (seq_len, half)
    sin = torch.sin(angles)
    return cos, sin

def apply_rope_to_qk(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    x: (B, H, T, D) where D is even
    cos, sin: (T, D//2)
    returns rotated x of same shape
    """
    B, H, T, D = x.shape
    half = D // 2
    # Split interleaved: even/odd positions along last dim
    x1 = x[..., :D:2]  # (B,H,T,half)
    x2 = x[..., 1:D:2]  # (B,H,T,half)
    # cos/sin -> (1,1,T,half) for broadcasting
    cos_b = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,half)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    # rotate
    x1c = x1 * cos_b - x2 * sin_b
    x2c = x1 * sin_b + x2 * cos_b
    # interleave back: [x1c0, x2c0, x1c1, x2c1, ...]
    x_rot = torch.stack([x1c, x2c], dim=-1).reshape(B, H, T, D)
    return x_rot

# -----------------------------
# Hybrid Sparse Attention with RoPE
# -----------------------------
class RoPEHybridSparseAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, window: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window = window
        self.kernel_size = 2 * window + 1

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # caches for RoPE will be created on forward based on seq_len

    def forward(self, x: torch.Tensor, global_mask: torch.Tensor = None):
        """
        x: (B, T, D)
        global_mask: (B, T) bool
        returns: out (B, T, D), full_attn (B, H, T, T)  # full_attn is for visualization
        """
        B, T, D = x.shape
        device = x.device
        # 1) project
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B,H,T,dh)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 1.5) build RoPE cache and apply to q,k
        cos, sin = build_rope_cache(T, self.head_dim, device=device, dtype=q.dtype)  # (T, dh/2)
        q = apply_rope_to_qk(q, cos, sin)
        k = apply_rope_to_qk(k, cos, sin)

        # 2) extract local windows
        K_windows = unfold_kv(k, kernel_size=self.kernel_size, padding=self.window)  # (B,H,T,win,dh)
        V_windows = unfold_kv(v, kernel_size=self.kernel_size, padding=self.window)  # (B,H,T,win,dh)

        # 3) local scores
        scores_local = torch.einsum("bhtd,bhtwd->bhtw", q, K_windows) / (self.head_dim ** 0.5)  # (B,H,T,win)

        # 4) global part
        if global_mask is None:
            scores_global = None
            K_global = None
            V_global = None
            global_idx_list = [torch.empty(0, dtype=torch.long, device=device) for _ in range(B)]
            global_token_mask = None
        else:
            global_idx_list = []
            maxG = 0
            for b in range(B):
                idx = torch.nonzero(global_mask[b], as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    idx = torch.empty(0, dtype=torch.long, device=device)
                global_idx_list.append(idx)
                if idx.numel() > maxG:
                    maxG = idx.numel()

            if maxG == 0:
                scores_global = None
                K_global = None
                V_global = None
                global_token_mask = None
            else:
                # pad to maxG
                K_global = torch.zeros(B, self.num_heads, maxG, self.head_dim, device=device, dtype=q.dtype)
                V_global = torch.zeros(B, self.num_heads, maxG, self.head_dim, device=device, dtype=q.dtype)
                global_token_mask = torch.zeros(B, maxG, dtype=torch.bool, device=device)
                for b in range(B):
                    idx = global_idx_list[b]
                    if idx.numel() == 0:
                        continue
                    kg = k[b, :, idx, :]  # (H, G_b, dh)
                    vg = v[b, :, idx, :]
                    G_b = kg.shape[1]
                    K_global[b, :, :G_b, :] = kg
                    V_global[b, :, :G_b, :] = vg
                    global_token_mask[b, :G_b] = True

                scores_global = torch.einsum("bhtd,bhgd->bhtg", q, K_global) / (self.head_dim ** 0.5)
                # mask padded later

        # 5) combine local and global
        if scores_global is None:
            attn_local = F.softmax(scores_local, dim=-1)
            ctx_local = torch.einsum("bhtw,bhtwd->bhtd", attn_local, V_windows)
            out_heads = ctx_local  # (B,H,T,dh)
            full_attn = torch.zeros(B, self.num_heads, T, T, device=device, dtype=q.dtype)
            # fill local-only full_attn
            for t in range(T):
                left = max(0, t - self.window)
                right = min(T, t + self.window + 1)
                win_len = right - left
                # attn_local[..., t, :win_len] -> place at positions left:right
                full_attn[..., t, left:right] = attn_local[..., t, :win_len]
        else:
            # mask padded global slots
            gmask = global_token_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,G)
            scores_global = scores_global.masked_fill(~gmask, float("-1e9"))

            scores_cat = torch.cat([scores_local, scores_global], dim=-1)  # (B,H,T, win+G)
            attn_cat = F.softmax(scores_cat, dim=-1)
            attn_cat = self.dropout(attn_cat)

            w_local = attn_cat[..., : self.kernel_size]
            w_global = attn_cat[..., self.kernel_size :]

            ctx_local = torch.einsum("bhtw,bhtwd->bhtd", w_local, V_windows)
            ctx_global = torch.einsum("bhtg,bhgd->bhtd", w_global, V_global)
            out_heads = ctx_local + ctx_global

            # build full_attn for visualization
            full_attn = torch.zeros(B, self.num_heads, T, T, device=device, dtype=q.dtype)
            for b in range(B):
                gidx = global_idx_list[b]
                for t in range(T):
                    left = max(0, t - self.window)
                    right = min(T, t + self.window + 1)
                    win_len = right - left
                    # local part
                    full_attn[b, :, t, left:right] = w_local[b, :, t, :win_len]
                    # global part -> assign per actual indices
                    if gidx.numel() > 0:
                        G_b = gidx.numel()
                        full_attn[b, :, t, gidx] += w_global[b, :, t, :G_b]

        # 6) merge heads & out proj
        out = out_heads.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return out, full_attn


# -----------------------------
# Quick test snippet
# -----------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    B, T, D = 1, 48, 128
    H = 8
    window = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RoPEHybridSparseAttention(dim=D, num_heads=H, window=window).to(device)
    x = torch.randn(B, T, D, device=device)

    # set a couple of global tokens
    global_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    global_mask[0, 0] = True
    global_mask[0, 12] = True

    out, full_attn = model(x, global_mask)
    print("out.shape=", out.shape)            # (B,T,D)
    print("full_attn.shape=", full_attn.shape)  # (B,H,T,T)

    # visualize head 0
    att = full_attn[0, 0].detach().cpu().numpy()  # (T,T)
    plt.figure(figsize=(6,6))
    plt.imshow(att, aspect="auto")
    plt.colorbar()
    plt.title("RoPE + Hybrid Sparse Attention (head 0)")
    plt.show()
```

---

### 解説（短く）

* `build_rope_cache` と `apply_rope_to_qk` で RoPE を Q/K に適用。これにより attention の点積が相対位置 (i - j) に敏感になります。
* `unfold_kv` で K/V のローカルウィンドウをまとめて抽出（GPUで効率的）。
* `scores_local` と `scores_global` を作り、同じ softmax 空間で結合 → local と global が競合して重みづけされる。
* `full_attn` は可視化用で、ウィンドウ外は 0、global は実際の global indices にだけ値が入る。


結果

今回作成したLLMでMASK部を予測させた結果を示します。

[MASK]となっている個所をMASKして、→の部分がモデルが予測した結果、()の内部が正解データです。

```
式 会 社 兵 庫 共 融 銀行 [MASK→（](()}); ) 明 [MASK→治](治) 2 [MASK→2](2) 年
```

```
京都 支 店 ： 京 都市 下 [MASK→京](京) 区 河 原 町 松 原 ２ 丁 目 富 永 町 ３ ４ ８ 
```

```
可能性 がある 頭 部 付 属 肢 ・ 背 板 と 関 節 肢 を [MASK→も](も) [MASK→た](た) ない 胴
```

```
食品 関 連 事業 [MASK→者](者) [MASK→による](による) 食品 循 環 資源 の 有効 利用 を 促 進 する 
```
