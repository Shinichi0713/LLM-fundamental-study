
本日は以前説明したTransformerの軽量モデルであるMamba-2の実装、実験を行います。
実験目的は学習してランダム文字列以外を出せるようになるかです。
実験にあたり、利用する環境は例の如くGoogle Colabとします。

またMamba-2の詳細については以下の記事をご参考下さい。
実装にあたっては記事の内容を用いて行っていきます。

## Mamba-2の内部構成

Mamba-2の内部構造を、**実装のヒントになるように**分解して説明します。

### 1. Mamba-2の基本アイデア：SSD（State Space Duality）

Mamba-2の核は、**SSM（State Space Model）とAttentionの「双対性（duality）」** を利用することです。

- **SSM側**：  
  連続時間の線形システム  
  \(h'(t) = A h(t) + B x(t)\), \(y(t) = C h(t)\)  
  を離散化して、再帰的に状態を更新する（O(n)）。

- **Attention側**：  
  入力トークン同士のペアワイズな相互作用（O(n²)）。

Mamba-2は、**SSMを「行列積の形」に書き換えることで、Attentionと同じ構造（行列積）で計算できる**ことを示しました。  
これにより、**SSMの計算をGPUのテンソルコアで高速に実行**できるようになります。

### 2. Mamba-2のレイヤー構造（実装のヒント）

Mamba-2の1ブロック（`MambaBlock`）は、おおまかに以下の構造です。

__2.1 入力投影（`in_proj`）__

```python
self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
```

- 入力次元 `d_model` を `d_inner`（拡張次元）に投影し、**2倍のチャネル**にします。
- 半分は SSM 用（`x`）、半分はゲート用（`z`）です。

__2.2 1D因果畳み込み（`conv1d`）__

```python
self.conv1d = nn.Conv1d(
    in_channels=d_inner,
    out_channels=d_inner,
    kernel_size=d_conv,
    groups=d_inner,  # depthwise conv
    padding=d_conv - 1,
    bias=False,
)
```

- **局所的な依存関係**（n-gram的な情報）を捉えるための因果畳み込みです。
- `groups=d_inner` なので、**depthwise convolution** になっています。

__2.3 SSMパラメータ生成（`x_proj`, `dt_proj`, `A_log`, `D`）__

```python
self.x_proj = nn.Linear(d_inner, d_state + d_inner, bias=False)
self.dt_proj = nn.Linear(d_inner, d_time_rank, bias=False)
self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
self.D = nn.Parameter(torch.ones(d_inner))
```

- `x_proj`：入力 `x` から **B, C, Δ（dt）** を生成します。
- `dt_proj`：Δ（時間ステップ）を計算するための投影です。
- `A_log`：対数スケールの状態遷移行列（`A = -exp(A_log)`）。
- `D`：スキップ接続のスケール（`y = C h + D x`）。

__2.4 SSD（State Space Duality）による高速計算__

Mamba-2の肝は、**SSMの再帰計算を「行列積」に変換**することです。

- 通常のSSM：  
  \(h_t = \bar{A} h_{t-1} + \bar{B} x_t\)（逐次更新）
- Mamba-2（SSD）：  
  \(H = (I - A)^{-1} B X\) のような形で、**全シーケンスを一度に計算**します。

実装的には、以下のようなイメージです。

```python
# 疑似コード
# A_bar, B_bar, C を計算（dt に依存）
A_bar = exp(A * dt)  # (d_inner, d_state)
B_bar = (A_bar - I) * (A^{-1}) * B * dt  # 簡略化したイメージ

# 行列積で状態を計算
# H = (I - A_bar)^{-1} B_bar X のような形
H = torch.einsum("bli,id->bld", X, A_bar) + torch.einsum("bld,bli->bli", B_bar, X)
Y = torch.einsum("bli,bli->bli", C, H) + D * X
```

実際には、**連続時間のSSMを「台形則（trapezoidal rule）」で離散化**し、  
それを行列積の形に書き換えることで、**GPUのテンソルコアで高速に計算**します。

__2.5 ゲートと出力投影__

```python
y = y * F.silu(z)  # ゲート適用
y = self.out_proj(y)  # d_inner -> d_model
y = y + residual  # 残差接続
```

- `z` は入力投影のもう半分で、**選択的な情報の流れ**を制御します（Mambaの「選択的SSM」の核心）。
- `out_proj` で元の次元に戻し、残差接続を加えます。

### 3. Mamba-2の実装のポイント（ヒント）

__3.1 選択的SSM（Selective SSM）__

Mamba-2は、**入力に依存してSSMパラメータ（B, C, Δ）を変化させる**ことで、  
「どの情報を状態に保持するか」を動的に選択します。

- `x_proj` と `dt_proj` で、**入力ごとに異なるB, C, Δ**を生成。
- これにより、**長い依存関係を捉えつつ、不要な情報を忘れる**ことができます。

__3.2 SSDによる高速化__

- 再帰計算を**行列積に変換**することで、**シーケンス長に対して並列に計算**できます。
- GPUのテンソルコア（行列積ユニット）を最大限に活用できるため、**学習が非常に高速**になります。

__3.3 実装の簡略化（Colab用）__

Colabで自作する場合は、以下のように簡略化できます。

- `dt_proj` と複雑な離散化は省略し、**固定の時間ステップ**で近似。
- SSDの行列積も、**単純な `einsum` で近似**（前回提示したコードのような形）。
- まずは「動くMamba風モデル」を作り、その後で本家のSSDを参考に改良するのが現実的です。

### 4. まとめ

- Mamba-2は、**SSMとAttentionの双対性（SSD）**を利用して、SSMを**行列積の形で高速計算**します。
- 1ブロックは、**入力投影 → 1D因果畳み込み → SSMパラメータ生成 → SSD計算 → ゲート＆出力投影**という流れです。
- 実装のヒントとしては、
  - `in_proj` でチャネルを拡張し、`x` と `z` に分割。
  - `conv1d` で局所情報を抽出。
  - `x_proj`, `dt_proj`, `A_log`, `D` でSSMパラメータを生成。
  - SSD（行列積）で状態を一度に計算。
  - `z` でゲートをかけ、`out_proj` で次元を戻す。

## 実装

コードは以下に保管しています。

https://github.com/Shinichi0713/LLM-fundamental-study/tree/main/llm/mamba/src/mamba_v2


学習データは実験のため、以下の簡単な少数の文章で実行します。

```python
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

# 1. 学習データの再定義（少し情報を増やす）
rich_texts = [
    "Mamba is a linear-time sequence model for long context processing.",
    "Mamba is highly efficient and outperforms transformers in speed.",
    "Mamba is a breakthrough in state space model architectures.",
    "Mamba is used for both language and vision tasks nowadays."
] * 5 # 繰り返しを減らす（過学習防止）

# データセット更新
def collate_fn_ignore_padding(batch):
    # パディング値を -100 に設定すると、PyTorchのCrossEntropyLossは無視してくれます
    ignore_index = -100
    input_ids_list = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    max_len = 64 
    
    padded_inputs = []
    padded_labels = []
    
    for ids in input_ids_list:
        ln = len(ids)
        # 入力側は pad_token_id (なければ0) で埋める
        p_input = torch.cat([ids, torch.full((max_len - ln,), tokenizer.pad_token_id or 0)])
        # ラベル側は パディング部分を ignore_index で埋める
        p_label = torch.cat([ids, torch.full((max_len - ln,), ignore_index)])
        
        padded_inputs.append(p_input)
        padded_labels.append(p_label)
    
    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels)
    }
    
raw_dataset = Dataset.from_dict({"text": rich_texts})
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
train_loader = DataLoader(tokenized_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_ignore_padding)

# 2. モデルをリセット（または学習率を極小にする）
optimizer = AdamW(model.parameters(), lr=1e-6) # 学習率をさらに下げる
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Re-training with lower LR...")
model.train()
for epoch in range(10): # 少しじっくり回す
    print(epoch)
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(loss)
```

実装を行う上での注意点を挙げます。

1. tokenizerはgpt-2を使ってください。mamba-2でHugging faceより公開されているものは、語彙のサイズが想定と異なっているもののようです。
2. 学習する場合、勾配爆発を防止するため勾配クリッピングを行ってください。

```Python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
3. サイズは絞る。公式通りのサイズはGoogle Colabで動作しません。

以下でようやく動作しました。

```python
# 130mの構造（次元数）を保ちつつ、層の数だけを削ってVRAMを節約する設定
config = Mamba2Config(
    num_hidden_layers=4,      # 軽量化
    hidden_size=512,          # 128の倍数で安定
    num_heads=8,              # エラーが求めている 8 に合わせる
    head_dim=128,             
    expand=2,
    n_groups=8,               # 明示的に指定して安全策をとる
    vocab_size=tokenizer.vocab_size
)
```


__学習した結果__

未だ言語の分散表現が構築されてないでしょうから、文章の羅列マシーンです。
学習前はただのランダム文字列(動詞、名詞の区別がない)状態だったのですが、おぼろげながら文章の並びは正規に近づいているかもしれません。

当たり前ですが、まだまだです。
サイズ、言語の常識を学ぶ十分なデータ量、ともにまだありません。

__学習前__
```
Prompt: Mamba is
Generating: fare cigar CONTIN chunksikolb Youominium Pog Township describe times Bes%;Henryisbury Secure consortium gamble gen reasonable794 1949 Carnage Solitaireheavy extensivelyGrab Cyrus node
```

__学習後__
```
Prompt: Mamba is
Generating: armed Python EVENT discreditCRIP Andreasdeveloparial Leave overlooked power Uh gradually fearing Cant NON affluentGET Filmamenomez Riy unl scantENSE slimstyleachment shaft plastic
```

## 総括

一旦とりあえず学習して動作するMamba-2は実装出来ました。
Google Colab内で動かすには、サイズが思ったより軽量ではありませんでした。
モデルのサイズを絞れば、wikipediaなどのデータセットで学習はさせられそうですが、もう少し工夫してみたい結果となりました。



