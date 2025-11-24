RoPEスケーリングは、「RoPE（Rotary Positional Encoding）を長いシーケンスでもうまく働かせるための拡張・改良テクニック」のことです。

---

## ✅ そもそもRoPEとは？

Transformerで使われる**位置情報の埋め込み（Positional Encoding）**の一種で、

トークンを複素数平面上の回転として表現します。

### 簡単に言うと

* 従来の学習可能位置埋め込み：位置を直接ベクトルで表す
* RoPE：**トークンベクトルに “回転角” を与えて位置関係を表現**

これにより、

| 特徴                   | 効果                               |
| ---------------------- | ---------------------------------- |
| 相対位置関係を保持     | → 長文で意味関係が崩れにくい      |
| 数式的にシンプル       | → GPU計算と相性が良い             |
| 推論時に長文へ拡張可能 | → 長い上下文処理に有効（LLM必須） |

---

## ✅ RoPEスケーリングとは？

RoPEは高い性能を持ちますが、 **本来は学習時に使った最大長近くまでしか安定しません** 。

> 例えば学習時 4K tokens → 推論時 128K tokens → 精度が低下

この問題を解決し、**学習より遥かに長いシーケンスで性能を維持する技術**がRoPEスケーリングです。

---

## ✅ スケーリングの主な手法

| 手法                                        | 概要                                           | 使われ例                       |
| ------------------------------------------- | ---------------------------------------------- | ------------------------------ |
| **NTKスケーリング**                   | 周波数を非線形に調整し、長距離ロバスト性を向上 | LLaMA系、Mistral               |
| **Dynamic NTK**                       | 入力長に応じ動的にスケール                     | 最新研究モデル                 |
| **YaRN (Yet another RoPE extension)** | 線形＋非線形補正で安定性大                     | LLaMA2・LLaMA3・Qwenなどで主流 |

---

## ✅ イメージ図

RoPEは周波数(回転速度)がトークン位置と共に増加するが、

長文では高周波成分が壊れる → スケールして緩和

```
無スケーリング:
位置上昇 → 角度増加 → 破綻

RoPEスケーリング:
位置上昇 → スケール係数で角度補正 → 安定
```

---

## ✅ 数式のイメージ

RoPEの角周波数 ω を、スケール係数 `s` で変換：

```
θ = pos / s
```

YaRNはさらに非線形補正を組み合わせる方式です。

---

## ✅ 結果：何が良くなるか？

| 改善点           | 効果                          |
| ---------------- | ----------------------------- |
| 長文性能アップ   | 100K〜1M tokensなど超長文対応 |
| 精度劣化の回避   | 長文でも意味が飛ばない        |
| 少ない追加コスト | 実装が軽い                    |

---

## ✅ まとめ

| ポイント         | 内容                            |
| ---------------- | ------------------------------- |
| RoPE             | 回転で位置表現                  |
| 課題             | 長文で劣化                      |
| RoPEスケーリング | 長文でも安定化                  |
| 採用例           | LLaMA, Qwen, Mistralなど主流LLM |

---

**RoPE (Rotary Positional Embedding)** の実装は、数式上では行列の掛け算ですが、実際のコード（PyTorchなど）では計算効率を最大化するために**要素ごとの積（Element-wise multiplication）**として実装されます。

現代のLLM（LLaMA, Mistral, PaLMなど）で標準的に採用されている実装スタイルを、ステップバイステップで解説します。

---

## 🛠️ RoPE実装の全体像

RoPEの実装は、主に以下の3つのパートに分かれます。

1.  **周波数（角度）の事前計算**: `Standard`なPEと同様に、$\sin$ と $\cos$ のテーブルを事前に作ってキャッシュします。
2.  **回転用のヘルパー関数**: ベクトルを効率的に回転させるための関数です。
3.  **適応関数**: クエリ($Q$)とキー($K$)に回転を適用します。

### 📌 重要な最適化テクニック
数式通りの行列演算 $\mathbf{R}\mathbf{x}$ を行うと計算量が多いため、以下のような変形を利用します。

2次元ベクトル $\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$ を角度 $\theta$ で回転させる場合：
$$
\begin{pmatrix} x_1 \cos\theta - x_2 \sin\theta \\ x_2 \cos\theta + x_1 \sin\theta \end{pmatrix} = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \otimes \begin{pmatrix} \cos\theta \\ \cos\theta \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \end{pmatrix} \otimes \begin{pmatrix} \sin\theta \\ \sin\theta \end{pmatrix}
$$
※ $\otimes$ は要素ごとの積。

この式を使うことで、行列演算なしで回転を実現します。

---

## 💻 PyTorchによる実装コード

このコードは、Hugging Faceの `transformers` ライブラリやLLaMAの実装に近い形式です。

```python
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        
        # 1. 周波数 (theta) の計算
        # dim は head_dim (d_model / num_heads)
        # 2つずつペアにするため、計算するのは dim/2 個の周波数
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # モデルの状態として保存しない（勾配計算不要）バッファとして登録
        self.register_buffer("inv_freq", inv_freq)
        
        # cos/sinのキャッシュを初期化
        self.max_seq_len_cached = max_seq_len
        t = torch.arange(max_seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        
        # 外積を使って (seq_len, dim/2) の角度グリッドを作成
        freqs = torch.outer(t, self.inv_freq)
        
        # LLaMAスタイル: ベクトルの前半と後半をペアにするため、同じ角度を2回繰り返して結合
        # emb: (seq_len, dim) -> [theta_0, theta_1, ..., theta_0, theta_1, ...]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # cos, sin を計算してキャッシュ (batch, head次元のためにunsqueezeしておく)
        # shape: (1, 1, seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        # x: (batch, n_heads, seq_len, head_dim)
        if seq_len > self.max_seq_len_cached:
            # キャッシュサイズを超えた場合の再計算ロジック（省略可だが実用的には必要）
            self._update_cache(seq_len, x.device)
            
        # 入力シーケンス長に合わせてスライスして返す
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    def _update_cache(self, seq_len, device):
        # キャッシュ更新用メソッド（__init__と同じ処理）
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])


def rotate_half(x):
    """
    ベクトルを半分に分割し、符号を反転させて入れ替える関数。
    これが (-x2, x1) の部分に相当します。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    クエリとキーにRoPEを適用する関数。
    数式: (x * cos) + (rotate_half(x) * sin)
    """
    # q, k の形状: (batch, n_heads, seq_len, head_dim)
    # cos, sin の形状: (1, 1, seq_len, head_dim) -> ブロードキャストされる
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

```

---

## 🚀 使用例

```python
# パラメータ設定
batch_size = 2
n_heads = 8
seq_len = 10
head_dim = 64  # d_model / n_heads

# 1. インスタンス化
rope = RotaryEmbedding(dim=head_dim, max_seq_len=2048)

# 2. ダミーのQueryとKeyを作成 (Transformer内部での計算を想定)
q = torch.randn(batch_size, n_heads, seq_len, head_dim)
k = torch.randn(batch_size, n_heads, seq_len, head_dim)

# 3. 現在のシーケンス長に対応する cos, sin を取得
cos, sin = rope(q, seq_len=seq_len)

# 4. 回転を適用
q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)

print(f"Original Q shape: {q.shape}")
print(f"Rotated Q shape:  {q_rotated.shape}")
# -> 形状は変わらず、値だけが回転されている
```

## 🔍 実装のポイント解説

### 1. ペアの作り方 (`torch.cat` vs Interleaved)
RoPEの論文や初期の実装では、隣り合う要素 $(x_0, x_1), (x_2, x_3)$ をペアにしていました。しかし、現代の効率的な実装（LLaMAなど）では、**ベクトルを前半と後半にバッサリ分割**してペアにします。

* ベクトル: $[x_0, x_1, \dots, x_{d/2-1}, \dots, x_{d-1}]$
* ペア: $(x_0, x_{d/2}), (x_1, x_{d/2+1}), \dots$

コード中の `emb = torch.cat((freqs, freqs), dim=-1)` と `rotate_half` 関数はこの「前半・後半分割」方式に対応しています。これにより、`tensor[::2]`のようなストライド操作よりもメモリアクセスが連続し、効率が良くなります。

### 2. `rotate_half` の意味
数式 $\begin{pmatrix} -x_2 \\ x_1 \end{pmatrix}$ を実現しています。
```python
x1, x2 = x.chunk(2, dim=-1) # ベクトルを前半(x1)と後半(x2)に分割
return torch.cat((-x2, x1), dim=-1) # (-x2, x1) の順で結合
```

### 3. ブロードキャスト
`cos` と `sin` の形状を `(1, 1, seq_len, dim)` にしています。
入力 `q` は `(batch, n_heads, seq_len, dim)` なので、PyTorchのブロードキャスト機能により、自動的に全バッチ・全ヘッドに対して同じ回転行列（位置に依存）が適用されます。

この実装は非常に高速で、GPU上でも効率的に動作するため、現在のLLM開発におけるデファクトスタンダードとなっています。