 **RMSNorm** （Root Mean Square Layer Normalization）は、2019年に提案された手法で、従来のLayerNormをさらに軽量化・高速化した正規化手法です。Llama 3やGemma、そしてModernBERTなど、最新のほぼすべてのLLMで採用されています。

一言で言うと、**「平均の計算をサボっても、精度を落とさず高速化できる」**という発見に基づいた技術です。

---

### 1. 従来の LayerNorm との数学的な違い

従来の **LayerNorm** は、データの「平均」と「分散」の両方を計算して正規化します。

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
$$

($\mu$: 平均, $\sigma$: 標準偏差, $\gamma$: スケール, $\beta$: バイアス)

これに対し、**RMSNorm** は「平均 (**$\mu$**)」を引くプロセスと「バイアス (**$\beta$**)」を完全に省略し、**「二乗平均平方根 (RMS)」**のみで割ります。

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}} \cdot \gamma
$$

---

### 2. なぜ RMSNorm が優れているのか？（メリット）

#### ① 計算コストの削減（高速化）

LayerNormでは「平均」と「分散（平均との差の二乗和）」の2つの統計量を計算する必要があります。

RMSNormは「二乗の和」を計算するだけで済むため、計算ステップが減り、GPU上での実行速度が向上します。わずかな差に見えますが、何百層もあるモデルではこの積み重ねが大きな速度差（スループットの向上）に繋がります。

#### ② 数値的な安定性

研究の結果、Transformerにおいて重要なのは「平均を0にすること（再中心化）」ではなく、**「入力のスケールを一定に保つこと（再スケーリング）」**であることが分かりました。RMSNormはスケーリングに特化しているため、LayerNormと同等の学習安定性を維持できます。

#### ③ パラメータの削減

バイアス項 (**$\beta$**) を使わないことが多いため、学習対象のパラメータ数がわずかに減り、メモリ節約にも寄与します。

---

### 3. ModernBERT における RMSNorm

ModernBERTでもRMSNormが採用されていますが、ここには**「ハードウェアへの最適化」**という意図も含まれています。

* **カーネルフュージョン** : RMSNormの計算は非常にシンプルなため、GPUのカーネル（計算の塊）として他の演算と結合しやすく、メモリの読み書きをさらに減らすことができます。
* **Flash Attentionとの相性** : Pre-Norm構造でRMSNormを使用することで、Attention層に入る直前の数値を高速に整えることができ、全体的な推論レイテンシの削減に貢献しています。

---

### 4. まとめ：LayerNorm vs RMSNorm

| **特徴**                           | **LayerNorm (従来)**                 | **RMSNorm (現代)**               |
| ---------------------------------------- | ------------------------------------------ | -------------------------------------- |
| **計算要素**                       | 平均**$\mu$**と 分散**$\sigma$** | **二乗平均平方根 (RMS) のみ**    |
| **バイアス (**$\beta$**)** | あり                                       | **なし (省略されることが多い)**  |
| **計算負荷**                       | 標準的                                     | **低い (高速)**                  |
| **学習の安定性**                   | 高い                                       | **高い (同等)**                  |
| **主な採用例**                     | BERT, GPT-2                                | **Llama 3, ModernBERT, Mistral** |

---

### 実装（PyTorchによる簡易版）

**Python**

```
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 入力 x の二乗平均の平方根 (RMS) を計算
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # 正規化してスケールを適用
        return x / rms * self.weight
```

RMSNormは、**「本質的でない計算を削ぎ落とした、実利的な最適化」**の代表例と言えます。



## VS LayerNorm

RMSNormと通常のLayerNormの最大の違いは、 **「データの中心（平均）をゼロに合わせる処理を行うかどうか」** にあります。

LayerNormは「平均」と「分散」の両方を使いますが、RMSNormは「二乗平均（RMS）」のみを使います。これにより、計算がシンプルになり、速度が向上します。

---

### 1. 数学的な処理の違い

| **手法**      | **処理内容**                            | **数式（簡略化）**                                                                    |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **LayerNorm** | 平均を0にし、分散を1にする                    | **$y = \frac{x - \text{mean}}{\sqrt{\text{var} + \epsilon}} \cdot \gamma + \beta$** |
| **RMSNorm**   | **平均を引かず** 、二乗平均平方根で割る | **$y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$**                 |

---

### 2. PyTorchによる実装例での比較

実際に両方のロジックをPython（PyTorch）で実装して、内部でどのような計算が行われているかを確認してみましょう。

**Python**

```
import torch
import torch.nn as nn

# サンプルデータ（バッチサイズ1, 次元数4）
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
dim = x.shape[-1]
eps = 1e-6

# --- 1. LayerNorm の手動実装 ---
mean = x.mean(dim=-1, keepdim=True)        # 平均: (1+2+3+4)/4 = 2.5
var = x.var(dim=-1, keepdim=True, unbiased=False) # 分散
x_ln = (x - mean) / torch.sqrt(var + eps)  # 平均を引いてから割る

# --- 2. RMSNorm の手動実装 ---
# 平均を計算するプロセスがない！
rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps) # 二乗平均の平方根
x_rmsn = x / rms                           # そのまま割る

print(f"Original:  {x}")
print(f"LayerNorm: {x_ln}")
print(f"RMSNorm:   {x_rmsn}")
```

#### 実装からわかる重要な違い

1. **平均引き算（Re-centering）の有無** :

* LayerNormは `x - mean` を行います。これによりデータは0を中心に分布します。
* RMSNormは `x` をそのまま使います。データの中心位置は動かさず、**「大きさ（スケーリング）」だけを整えます。**

1. **パラメータの簡略化** :

* LayerNormには通常、重み（**$\gamma$**）とバイアス（**$\beta$**）の2つがあります。
* RMSNormは通常、**バイアス（**$\beta$**）を使いません。** これにより学習パラメータも削減されます。

---

### 3. なぜRMSNormが使われるのか？

近年の研究（特にTransformer系）では、**「平均を0にすること」よりも「入力のスケールを一定に保つこと」の方が学習の安定に重要である**ことがわかってきました。

* **高速化** : 平均を計算しなくていいため、計算ステップが減り、GPUでの処理が高速になります。
* **省メモリ** : バイアス項を持たないため、パラメータ数が減ります。
* **実績** : Llama 3、Mistral、ModernBERTなど、主要な最新モデルのほぼすべてがこのRMSNormを採用しています。

### まとめ

* **LayerNorm** : 「平均を引く＋分散で割る」の2ステップ（丁寧だが少し重い）。
* **RMSNorm** : 「二乗平均で割る」の1ステップ（シンプルで速い）。
