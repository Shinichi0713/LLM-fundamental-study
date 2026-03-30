Identity Preference Optimization（IPO）は、**人間の好み（preference）に基づいてLLMを調整する「選考学習法」の一種**で、特にDPO（Direct Preference Optimization）の弱点（過学習・報酬の発散）を改善する目的で提案された手法です[Emergent Mind](https://www.emergentmind.com/topics/identity-preference-optimization-ipo)[Hugging Face Blog](https://huggingface.co/blog/pref-tuning)。

以下では、IPOの位置づけ、目的、損失関数、DPOとの違いを順に説明します。

![1774912491780](image/method/1774912491780.png)

## 1. IPOの位置づけ：RLHF → DPO → IPO

### 1.1 RLHFとDPOの簡単な復習

- **RLHF（Reinforcement Learning from Human Feedback）**  
  人間のフィードバックから報酬モデルを学習し、その報酬モデルを使って強化学習でLLMを更新します。  
  → 報酬モデルの学習とRLの2段階が必要で、実装が重く不安定になりがちです。

- **DPO（Direct Preference Optimization）**  
  報酬モデルを明示的に学習せず、**「好ましい応答」と「好ましくない応答」のペア**から直接ポリシー（LLM）を更新します。  
  → RLHFよりシンプルで安定、近年広く使われています。

DPOは、Bradley–Terryモデルに基づき、「好ましい応答の対数尤度が、好ましくない応答より十分大きくなるように」学習します。  
しかし、**報酬の差（マージン）が際限なく大きくなり、過学習や報酬の発散を招く**という問題が指摘されています。

### 1.2 IPOの立ち位置

IPOは、DPOと同じく**ペア形式の好みデータ**（プロンプト $x$、好ましい応答 $y_w$、好ましくない応答 $y_l$）を使いますが、

- DPOの損失関数を**より強く正則化したバージョン**
- あるいは、**ΨPO（より一般的な好み最適化フレームワーク）の特殊ケース**

として位置づけられます[Argilla Blog](https://argilla.io/blog/mantisnlp-rlhf-part-6/)。


## 2. IPOの目的と直感的なアイデア

### 2.1 目的

IPOの主な目的は次の通りです[Emergent Mind](https://www.emergentmind.com/topics/identity-preference-optimization-ipo)。

- **報酬マージンの「適切な大きさ」に固定する**  
  → 好ましい応答と好ましくない応答の報酬差が、際限なく大きくなるのを防ぐ。
- **強い正則化を備えつつ、計算的にシンプルで安定な学習**  
  → 報酬モデルやRLソルバーを必要とせず、オフラインで学習可能。
- **DPOで見られる過学習・報酬発散の問題を軽減**  
  → 早期打ち切りなどのテクニックに頼らず、収束まで学習できる。

### 2.2 直感的なイメージ

- DPO：  
  「好ましい応答の確率を、好ましくない応答より**できるだけ大きく**する」  
  → マージンが際限なく大きくなり、過学習しやすい。

- IPO：  
  「好ましい応答と好ましくない応答の報酬差を、**ある固定値 $c$ に近づける**」  
  → マージンが一定に保たれ、過度な「自信過剰」を防ぐ。


## 3. IPOの損失関数（数式）

### 3.1 データ形式

- 各サンプルは $(x, y_w, y_l)$ の組：
  - $x$：プロンプト
  - $y_w$：好ましい応答（chosen）
  - $y_l$：好ましくない応答（rejected）

### 3.2 暗黙の報酬マージン

LLMのポリシーを $\pi_\theta$ とすると、その**暗黙的な報酬差（マージン）**を

$$
\delta_r(x, y_w, y_l) = \log \frac{\pi_\theta(y_w \mid x)}{\pi_\theta(y_l \mid x)}
$$

と定義します。これは、好ましい応答の対数尤度から、好ましくない応答の対数尤度を引いたものです。

### 3.3 IPO損失（標準形）

IPOの標準的な損失関数は、**二乗誤差（squared-error）**を用いて次のように定義されます[Emergent Mind](https://www.emergentmind.com/topics/identity-preference-optimization-ipo)：

$$
L_{\text{IPO}}(\theta) = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( \delta_r(x, y_w, y_l) - c \right)^2 \right]
$$

ここで、
- $\delta_r(x, y_w, y_l)$：上で定義した暗黙の報酬マージン
- $c = \frac{1}{2\beta}$：**固定のターゲットマージン**
- $\beta > 0$：正則化の強さを制御するハイパーパラメータ

**ポイント**：
- DPOは**クロスエントロピー型**の損失で、マージンを「できるだけ大きく」する。
- IPOは**二乗誤差型**の損失で、マージンを「$c$ に近づける」。
- これにより、マージンが際限なく大きくなるのを防ぎ、**強い正則化**がかかります。


## 4. IPOとDPOの主な違い

### 4.1 損失関数の形

- **DPO**：  
  クロスエントロピー損失（ロジスティック損失）を用い、  
  「好ましい応答が選ばれる確率 → 1」に近づける。

- **IPO**：  
  二乗誤差損失を用い、  
  「報酬マージン → 固定値 $c$」に近づける。

→ IPOは、**マージンが一定以上に大きくならないように制御**するため、過学習や報酬発散を抑制しやすいです[Emergent Mind](https://www.emergentmind.com/topics/identity-preference-optimization-ipo)。

### 4.2 正則化の強さ

- DPO：  
  ロジスティック関数の性質上、マージンが大きくなると勾配が飽和し、正則化が弱くなる。

- IPO：  
  二乗誤差はマージンが大きくなっても勾配が大きいままなので、**より強い正則化**がかかります。

### 4.3 実装上の注意点（Hugging Faceブログより）

Hugging Faceの実装では、IPOの損失を計算する際に、

- 各応答の対数尤度を**合計ではなく平均**する必要がある

という点が強調されています[Hugging Face Blog](https://huggingface.co/blog/pref-tuning)。  
これを守らないと性能が大きく低下するため、実装時には注意が必要です。


## 5. 実務的な評価と位置づけ

### 5.1 実験結果（Hugging Faceの比較）

Hugging Faceによる比較実験では[Hugging Face Blog](https://huggingface.co/blog/pref-tuning)：

- **IPOはDPOとほぼ同等の性能**を示す。
- ペア形式の好みデータがある設定では、**KTO（Kahneman–Tversky Optimization）より優れる**。
- ただし、DPOは依然として実務で非常に頑健で広く使われている。

また、IPOの性能が最も良くなる $\beta$ は**非常に小さい値（例：0.01）**であることが報告されています。

### 5.2 ΨPOフレームワークの中でのIPO

Argillaの解説では[Argilla Blog](https://argilla.io/blog/mantisnlp-rlhf-part-6/)：

- RLHFとDPOを統一する**ΨPO（Psi-PO）**という一般フレームワークが提案されている。
- IPOはその中で、**logit関数をidentity関数に置き換えた特殊ケース**として位置づけられる。
- これにより、**KL正則化が決定的な好み（deterministic preferences）の状況でも有効に働く**ようになり、DPOの過学習問題を改善できるとされています。


## 6. 実装手順

IPOの実装手順を、**データ準備 → 損失計算 → モデル更新**の流れで具体的に説明します。  
ここでは、Hugging Face Transformers + PyTorch を想定した疑似コード形式で示します。


### 1. 前提：データ形式とモデル

__1.1 データ形式__

IPOはDPOと同じく、**ペア形式の好みデータ**を使います。  
各サンプルは以下の情報を持ちます：

- `prompt`：入力テキスト（プロンプト）
- `chosen`：好ましい応答（win / preferred）
- `rejected`：好ましくない応答（lose / dispreferred）

例（JSONL形式）：

```json
{
  "prompt": "Explain the concept of IPO.",
  "chosen": "IPO stands for Identity Preference Optimization...",
  "rejected": "IPO is a financial term meaning Initial Public Offering..."
}
```

__1.2 モデル__

- ベースとなるLLM：`model`（例：`Mistral-7B`, `Llama-3-8B` など）
- トークナイザ：`tokenizer`

PyTorch + Transformers の例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 必要に応じて
```

### 2. IPO損失の計算手順（ステップバイステップ）

IPOの標準的な損失は、以下のように定義されます[Emergent Mind](https://www.emergentmind.com/topics/identity-preference-optimization-ipo)：

$$L_{\text{IPO}}(\theta) = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( \delta_r(x, y_w, y_l) - c \right)^2 \right]$$

ここで、

- $$\delta_r(x, y_w, y_l) = \log \frac{\pi_\theta(y_w \mid x)}{\pi_\theta(y_l \mid x)}$$
- $$c = \frac{1}{2\beta}$$（$\beta > 0$ はハイパーパラメータ）

__2.1 ステップ1：トークナイズとログ確率の取得__

1. プロンプト＋応答を結合し、トークナイズします。
2. 各トークンに対する**対数確率（log-likelihood）**を取得します。
3. 応答部分のログ確率を**平均**します（Hugging Faceブログで強調されているポイント）[Hugging Face Blog](https://huggingface.co/blog/pref-tuning)。

疑似コード：

```python
def get_logp(model, tokenizer, prompt, response):
    # プロンプト＋応答を結合
    text = prompt + response
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # ログ確率を計算
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)
        # 各トークンの対数確率
        log_probs = torch.log_softmax(logits, dim=-1)
    
    # 応答部分のみを抽出（プロンプト部分をマスク）
    prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
    response_log_probs = log_probs[0, prompt_len-1:-1, :]  # シフトに注意
    
    # 実際に生成されたトークンのログ確率を取得
    input_ids = inputs["input_ids"][0, prompt_len:]
    selected_log_probs = response_log_probs[torch.arange(len(input_ids)), input_ids]
    
    # 応答部分のログ確率の平均を返す
    return selected_log_probs.mean()
```

__2.2 ステップ2：暗黙の報酬マージン $\delta_r$ の計算__

各サンプル $(x, y_w, y_l)$ について：

- $$\log p_w = \text{get\_logp}(x, y_w)$$
- $$\log p_l = \text{get\_logp}(x, y_l)$$

とすると、

$$\delta_r = \log p_w - \log p_l$$

疑似コード：

```python
logp_chosen = get_logp(model, tokenizer, prompt, chosen)
logp_rejected = get_logp(model, tokenizer, prompt, rejected)
delta_r = logp_chosen - logp_rejected
```

__2.3 ステップ3：IPO損失の計算__

ターゲットマージン $$c = \frac{1}{2\beta}$$ を設定し、二乗誤差を計算します。

```python
beta = 0.01  # 例：Hugging Faceブログで推奨される小さな値
c = 1.0 / (2.0 * beta)

loss = (delta_r - c) ** 2
loss = loss.mean()  # バッチ平均
```

**ポイント**：
- DPOは `-log(sigmoid(delta_r))` のようなクロスエントロピー損失を使いますが、IPOは**二乗誤差**です。
- これにより、マージンが $c$ を超えて大きくなると、**ペナルティが増える**ため、過学習を抑制できます。

### 3. 学習ループの疑似コード（簡略版）

```python
import torch
from torch.utils.data import DataLoader
from transformers import AdamW

# データローダの準備（仮）
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-6)

model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        prompts = batch["prompt"]
        chosens = batch["chosen"]
        rejecteds = batch["rejected"]
        
        losses = []
        for prompt, chosen, rejected in zip(prompts, chosens, rejecteds):
            logp_w = get_logp(model, tokenizer, prompt, chosen)
            logp_l = get_logp(model, tokenizer, prompt, rejected)
            delta_r = logp_w - logp_l
            loss_i = (delta_r - c) ** 2
            losses.append(loss_i)
        
        loss = torch.stack(losses).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4. Hugging Face TRL を使った実装（推奨）

手動で実装するよりも、**Hugging Face TRL（Transformer Reinforcement Learning）ライブラリ**を使うのが現実的です。

__4.1 インストール__

```bash
pip install trl
```

__4.2 データセットの準備__

`datasets` ライブラリなどで、以下の形式のデータセットを用意します：

```python
from datasets import Dataset

dataset_dict = {
    "prompt": ["Explain IPO.", "What is RLHF?"],
    "chosen": ["IPO is...", "RLHF is..."],
    "rejected": ["IPO is a finance term...", "RLHF is not important..."],
}
dataset = Dataset.from_dict(dataset_dict)
```

### 4.3 `DPOTrainer` を `loss_type="ipo"` で使用

TRLの `DPOTrainer` は、`loss_type` に `"ipo"` を指定することでIPOをサポートしています[Hugging Face Blog](https://huggingface.co/blog/pref-tuning)。

```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 参照モデル（必要に応じて）
    args=training_args,  # TrainingArguments
    beta=beta,           # IPOのbeta
    train_dataset=dataset,
    tokenizer=tokenizer,
    loss_type="ipo",     # ここが重要
)

dpo_trainer.train()
```

**注意点**（Hugging Faceブログより）[Hugging Face Blog](https://huggingface.co/blog/pref-tuning)：

- IPOでは、**各応答のログ確率を合計ではなく平均する**必要があります。  
  TRLの実装ではこの点が考慮されているため、自前実装より安全です。
- `beta` は小さな値（例：0.01）から試すのが推奨されています。

### 5. 実装時のチェックリスト

1. **データ形式**：`(prompt, chosen, rejected)` のペアデータを用意。
2. **ログ確率の計算**：応答部分のみを対象にし、**平均**を取る。
3. **損失関数**：`(delta_r - c)^2` の形で実装し、`c = 1/(2*beta)` とする。
4. **ハイパーパラメータ**：`beta` は小さめ（0.01など）からチューニング。
5. **ライブラリ利用**：可能なら `trl.DPOTrainer(loss_type="ipo")` を使用。



## 7. まとめ

- **Identity Preference Optimization（IPO）**は、DPOと同じく**ペア形式の好みデータ**を用いてLLMをアラインメントする手法です。
- 損失関数は**二乗誤差**を用い、暗黙の報酬マージン $\delta_r$ を**固定のターゲット $c = 1/(2\beta)$ に近づける**ように学習します。
- これにより、DPOで問題となる**報酬マージンの発散や過学習**を抑制し、**より強い正則化**と**安定した学習**を実現します。
- 実験的には、DPOとほぼ同等の性能を示しつつ、過学習を抑えられることが報告されています[Hugging Face Blog](https://huggingface.co/blog/pref-tuning)。
- 理論的には、RLHFとDPOを統一する**ΨPOフレームワークの特殊ケース**として位置づけられ、より一般的な好み最適化の枠組みの中で理解できます[Argilla Blog](https://argilla.io/blog/mantisnlp-rlhf-part-6/)。



