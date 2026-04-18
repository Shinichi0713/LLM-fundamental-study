以下、Anthropicの **「Tracing the thoughts of a large language model」** における**Circuit Tracing**を、BERTなど他モデルで再現する手順を整理します。


## 1. Circuit Tracingの概要（Anthropic）

AnthropicのCircuit Tracingは、LLM内部の **「特徴量（features）」同士の接続を可視化** し、**どの特徴がどの出力に因果的に寄与しているか**を明らかにする手法です[Anthropic: Tracing the thoughts of a large language model](https://www.anthropic.com/research/tracing-thoughts-language-model)。

主な構成要素：

1. **Cross-Layer Transcoders (CLT)**  
   - MLP出力を**スパースな特徴量（SAE features）**で再構成する線形近似器。
2. **Local Replacement Model**  
   - Attentionパターン・LayerNorm分母を固定し、モデルを**線形化**した近似モデル。
3. **Attribution Graphs**  
   - 特徴量・トークン・出力logitをノードとし、**線形効果（edge weight）**で接続した計算グラフ。
4. **Feature Inhibition / Patching**  
   - 特定特徴を抑制・注入し、**因果関係**を検証する介入実験。

BERTもTransformerベースなので、**同じ枠組みで再現可能**です[Circuit Tracing: Revealing computational graphs in language models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)。

---

## 2. BERTでCircuit Tracingを再現する手順

### ステップ1：BERTのMLP層にSparse Autoencoder（SAE）を学習

**目的**：MLPの活性を**スパースな特徴量**に分解し、「意味のある概念」を抽出します。

- BERTのMLP出力：  
  `MLP_out = MLP(LayerNorm(x))`
- SAEの構造（1層目に適用）：
  - `a = JumpReLU(W_enc * x)`（スパース活性）
  - `x_recon = W_dec * a`（再構成）

**PyTorchコード例（1層目MLPに対するSAE）**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, n_features, sparsity_lambda=1e-3):
        super().__init__()
        self.W_enc = nn.Linear(d_model, n_features, bias=False)
        self.W_dec = nn.Linear(n_features, d_model, bias=False)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        # x: (B, seq_len, d_model)
        a = F.relu(self.W_enc(x))  # (B, seq_len, n_features)
        x_recon = self.W_dec(a)   # (B, seq_len, d_model)
        # L1スパース正則化
        loss_recon = F.mse_loss(x_recon, x)
        loss_sparse = self.sparsity_lambda * a.abs().mean()
        loss = loss_recon + loss_sparse
        return x_recon, a, loss
```

**学習**：
- BERTのMLP出力（`MLP_out`）を入力として、`x_recon ≈ MLP_out`になるようSAEを学習。
- 各層ごとにSAEを用意し、**層ごとの特徴量**を抽出します。

---

### ステップ2：Cross-Layer Transcoders（CLT）の構築

**目的**：ある層のMLP出力を、**下位層の特徴量の線形和**で再構成し、**層間の情報フロー**を線形近似します。

- 層`l`のMLP出力：`y^l`
- 下位層`l'`の特徴量：`a^{l'}`
- CLT再構成：
  `y_hat^l = Σ_{l'=1..l} W_dec^{l'→l} * a^{l'}`

**PyTorchコード例（層`l`に対するCLT）**：

```python
class CrossLayerTranscoder(nn.Module):
    def __init__(self, n_layers, d_model, n_features_per_layer):
        super().__init__()
        # W_dec^{l'→l}: 層l'の特徴 → 層lの再構成
        self.W_dec_list = nn.ModuleList([
            nn.Linear(n_features_per_layer[l_prime], d_model, bias=False)
            for l_prime in range(n_layers)
        ])

    def forward(self, features_list):
        # features_list[l_prime]: (B, seq_len, n_features[l_prime])
        y_hat = 0.0
        for l_prime, a_lprime in enumerate(features_list):
            y_hat += self.W_dec_list[l_prime](a_lprime)
        return y_hat
```

**学習**：
- BERTのMLP出力`y^l`を教師信号として、`y_hat^l ≈ y^l`になるようCLTを学習。
- これにより、**「どの層のどの特徴が、上層のどの出力に寄与しているか」**を線形近似できます。

---

### ステップ3：Local Replacement Modelの構築

**目的**：特定プロンプトに対して、Attentionパターン・LayerNorm分母を固定し、モデルを**線形化**します。

- Attentionパターン・LayerNorm分母は**非線形**ですが、これらを**固定**することで、
  - 残りの部分（MLP＋CLT）が**線形近似**として扱えるようになります。
- BERTの場合：
  - Self-Attentionの`softmax(QK^T)`を固定（勾配を止める）。
  - LayerNormの分母（標準偏差）を固定。

**PyTorchコード例（簡略化）**：

```python
def forward_with_frozen_attention(model, input_ids):
    # 通常のforward
    outputs = model(input_ids, output_hidden_states=True)
    # Attentionパターン・LayerNorm分母を固定（ここでは概念例）
    # 実際には、各層のAttention/LayerNormの出力をdetach()する
    # 例：layer.attention.self.value = layer.attention.self.value.detach()
    return outputs
```

**Local Replacement Model**：
- 固定したAttention/LayerNormの上に、**CLTで再構成したMLP出力**を流し込む。
- これにより、**「プロンプト依存の非線形部分を固定し、残りを線形近似」**したモデルが得られます。

---

### ステップ4：Attribution Graphsの計算

**目的**：**どの特徴がどの出力logitにどれだけ寄与しているか**を、**線形効果**として計算します。

- ノード：
  - 特徴量`a_i`（SAEの活性）
  - トークン埋め込み
  - 出力logit
- エッジ重み：
  - `A_{s→t} = a_s * w_{s→t}`
  - `w_{s→t}`は、**後ろ向きヤコビアン**（`J_backwards`）で近似：
    - `A_{s→t} = a_s * Σ(W_dec^T * J_backwards * W_enc)`

**PyTorchコード例（概念）**：

```python
def compute_attribution_graph(model, sae_list, input_ids, target_logit_idx):
    # 1. 特徴量a_iを取得（各層のSAE活性）
    features_list = []
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        for l, sae in enumerate(sae_list):
            _, a_l, _ = sae(outputs.hidden_states[l])
            features_list.append(a_l)

    # 2. ターゲットlogitに対するヤコビアンを計算
    #    （PyTorchのautograd.gradで、target_logitに対する各特徴の勾配を取得）
    # 3. Attribution: A_{s→t} = a_s * (∂logit_t / ∂a_s)
    #    ここでは概念例として、勾配を重みとして扱う
    attribution_weights = []
    for a_l in features_list:
        a_l.requires_grad_(True)
        # model forward with CLT/local replacement
        # logit_t = model(...)[target_logit_idx]
        # grad = torch.autograd.grad(logit_t, a_l, retain_graph=True)[0]
        # attribution = a_l * grad
        # attribution_weights.append(attribution)
        pass  # 実装はモデル構造に依存

    return attribution_weights
```

**実装上のポイント**：
- BERTは**双方向Attention**なので、Attribution計算時に**全トークン間の影響**を考慮する必要があります。
- 実際には、Anthropicの論文にある**backwards Jacobian**の式に従い、`stop_grad`で非線形部分を無視しながら計算します[Circuit Tracing: Revealing computational graphs in language models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)。

---

### ステップ5：Graph Pruning（重要ノード・エッジの抽出）

**目的**：Attribution Graphが大きすぎるため、**重要度の高いノード・エッジのみを残す**。

- 各エッジの重み`|A_{s→t}|`でソートし、上位k%のみを保持。
- ターゲットlogitへの**累積寄与度**が高いパスを抽出。

**PyTorchコード例（概念）**：

```python
def prune_attribution_graph(attribution_weights, top_k_ratio=0.1):
    # attribution_weights: 各層の(B, seq_len, n_features)テンソル
    all_weights = torch.cat([w.flatten() for w in attribution_weights])
    threshold = torch.quantile(all_weights.abs(), 1.0 - top_k_ratio)
    pruned = [w * (w.abs() >= threshold).float() for w in attribution_weights]
    return pruned
```

---

### ステップ6：Feature Inhibition（因果検証）

**目的**：特定特徴を**抑制・注入**し、出力がどう変わるかを見ることで、**因果的寄与**を検証します。

- **Constrained Patching**：
  - ある特徴`a_i`を0にしたり、別の値に置き換え、その分だけCLT再構成を修正。
- **Multiplicative Steering**：
  - `a_i`に-1を掛けて抑制（`a_i' = -a_i`）し、効果を観察。

**PyTorchコード例（特徴抑制）**：

```python
def inhibit_feature(model, sae_list, input_ids, layer_idx, feature_idx, multiplier=-1.0):
    # 1. 通常のforwardで特徴量を取得
    outputs = model(input_ids, output_hidden_states=True)
    x_l = outputs.hidden_states[layer_idx]
    _, a_l, _ = sae_list[layer_idx](x_l)

    # 2. 特定特徴を抑制
    a_l_inhibited = a_l.clone()
    a_l_inhibited[:, :, feature_idx] *= multiplier

    # 3. CLT/local replacementで再構成し、モデルに流す
    # （ここでは概念例）
    # y_hat_l = CLT(a_l_inhibited)
    # outputs_inhibited = model.forward_with_replaced_mlp(y_hat_l)
    # return outputs_inhibited
```

**評価**：
- 抑制前後で**ターゲットlogitの変化**を比較。
- 変化が大きい → その特徴は**因果的に重要**。

---

## 3. BERTでの具体的な適用例（例：NERタスク）

1. **タスク設定**：
   - BERT＋分類ヘッドで**固有表現認識（NER）**。
   - 例：`[CLS] John lives in Paris [SEP]` → `PER`, `LOC`ラベル。
2. **Circuit Tracingの適用**：
   - ターゲットlogit：`Paris`の`LOC`ラベルlogit。
   - SAEでMLP活性を特徴化。
   - CLTで層間再構成。
   - Local Replacement Modelで線形化。
   - Attribution Graphを計算し、**「Paris」トークンと「LOC」logitを結ぶ特徴パス**を可視化。
   - Feature Inhibitionで、**地名関連特徴**を抑制し、LOC確率が下がるか検証。

---

## 4. まとめ

- **Circuit Tracing**は、LLM内部の**特徴量同士の接続（circuits）**を可視化し、**因果的寄与**を検証する手法です。
- BERTでも、以下の手順で再現可能です：
  1. **SAEでMLP活性をスパース特徴化**
  2. **CLTで層間再構成を線形近似**
  3. **Local Replacement ModelでAttention/LayerNormを固定し線形化**
  4. **Attribution Graphs（backwards Jacobian）で特徴→logitの寄与を計算**
  5. **Graph Pruningで重要パスのみ抽出**
  6. **Feature Inhibitionで因果検証**
- これにより、BERTが**どのトークン・どの層のどの特徴を使って判断しているか**を、**「AI顕微鏡」としてトレース**できます。

実装の詳細は、Anthropicの**Methods論文**と**GitHubフロントエンド**を参照してください[Circuit Tracing: Revealing computational graphs in language models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)[Anthropic: Tracing the thoughts of a large language model](https://www.anthropic.com/research/tracing-thoughts-language-model)。


