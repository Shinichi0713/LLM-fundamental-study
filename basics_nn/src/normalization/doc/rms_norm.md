**RMSNorm (Root Mean Square Layer Normalization)** は、Transformerモデル（特に Llama 3 や Mistral などの最新LLM）で標準的に採用されている**レイヤー正規化（Layer Normalization）の改良版**です。

従来の LayerNorm を簡略化し、**「計算量を減らしつつ、同等以上の学習安定性を確保する」**ことを目的としています。

__1. 従来の LayerNorm との違い__

従来の **Layer Norm (LN)** は、データの「平均（Mean）」と「分散（Variance）」の両方を使って正規化を行っていました。

対して **RMSNorm** は、**「平均を引く計算を省略し、二乗平均平方根（RMS）だけで割る」**という非常にシンプルな設計になっています。

* ** (Gains):** 学習可能なスケーリング・パラメータ。
* **:** ゼロ除算を防ぐための微小な値。
* **:** ベクトルの次元数。

__2. なぜ平均を引くか__

RMSNorm の提案論文によると、LayerNorm が成功している主な理由は「平均を引いて中心を合わせること」ではなく、**「入力のスケール（大きさ）を一定に保つこと」**にあることが分かりました。

* **再スケーリング不変性:** 重みのスケールが変わっても出力が安定する性質です。
* **計算の効率化:** 平均を計算するステップがなくなるため、計算コストが **10%〜40% 削減** されます。
* **微分の簡略化:** 逆伝播（バックプロパゲーション）時の計算もシンプルになり、学習が高速化します。

__PyTorch での実装例__

LLMの実装でよく使われる、シンプルで効率的な RMSNorm のコードです。

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 学習可能なパラメータ γ (Gains) のみを持つ
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x: [batch, seq_len, dim]
        # 二乗平均平方根 (RMS) で割る。平均を引く工程がない。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 正規化した後、γを掛けてスケールを調整
        return self._norm(x.float()).type_as(x) * self.weight

```

---

### 4. まとめ：なぜ現代のLLMに必須なのか

| 特徴 | Layer Normalization | RMSNorm |
| --- | --- | --- |
| **計算要素** | 平均 + 分散 | **二乗平均平方根のみ** |
| **パラメータ** |  (Scale) と  (Shift) | ** (Scale) のみ** |
| **計算負荷** | 標準 | **軽い（高速）** |
| **採用モデル** | BERT, GPT-2 | **Llama, Mistral, PaLM, Gemma** |

現代のLLMでは、モデルが非常に深くなるため、各層での計算コストのわずかな差が全体の学習時間に大きく影響します。RMSNorm は「効果を落とさずに無駄を削ぎ落とした」非常に洗練された工夫といえます。

**この RMSNorm が、モデルのどの位置（Pre-norm か Post-norm か）に置かれるのが主流か、その配置の重要性についても興味はありますか？**