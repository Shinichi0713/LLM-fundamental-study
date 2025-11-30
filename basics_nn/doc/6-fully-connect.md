# 簡単な計算例題
以下は、**初心者が「全結合層（Fully Connected Layer）」の計算ロジックを“0から本質的に理解できる”ように設計した NumPy 演習問題**です。
数式 → 図解 → 具体的な数字 → NumPyコード の順で「階段式に理解」できるように作ってあります。

---

# 🎓 **例題：NumPy で “全結合層の計算” をゼロから理解する演習**

---

# 📌 **ゴール**

* 全結合層の計算式が理解できる
* 行列積の意味が理解できる
* NumPyで実際に全結合層を実装できる

---

# 🧩 **STEP 1：全結合層が何をしているかを理解しよう**

全結合層（Linear layer / Dense layer）では次の計算をしています：

[
\textbf{y} = \textbf{x}W + \textbf{b}
]

---

### 🔍 **図解（イメージ）**

```
入力 x = [x1, x2]

重み W =
    [[w11, w12],
     [w21, w22]]

バイアス b = [b1, b2]

出力 y = [y1, y2]
```

計算：

```
y1 = x1*w11 + x2*w21 + b1
y2 = x1*w12 + x2*w22 + b2
```

**つまり、入力 × 重み + バイアス の単純計算の集合**です！

---

# 📘 **STEP 2：具体的な数字で理解しよう**

### ▼ 次の値を使います

* 入力（1データ）

  ```
  x = [1.0, 2.0]
  ```
* 重み

  ```
  W = [[3.0, 1.0],
       [2.0, 4.0]]
  ```
* バイアス

  ```
  b = [1.0, -1.0]
  ```

---

### 🔍 **計算してみよう**

### 出力 y1

[
y1 = 1\cdot 3 + 2\cdot 2 + 1 = 3 + 4 + 1 = 8
]

### 出力 y2

[
y2 = 1\cdot 1 + 2\cdot 4 - 1 = 1 + 8 -1 = 8
]

---

### 👉 **答え**

[
y = [8, ; 8]
]

シンプル！

---

# 🧮 **STEP 3：NumPy を使って計算してみよう**

初心者が理解しやすいように「行列積の中身を手で展開」と「行列積を使う場合」の両方を示します。

---

# ⭐ **例題コード（初級：手計算を NumPy でそのまま書く）**

```python
import numpy as np

# --- 入力 ---
x = np.array([1.0, 2.0])  # shape = (2,)

# --- 重み ---
W = np.array([
    [3.0, 1.0],
    [2.0, 4.0]
])  # shape = (2, 2)

# --- バイアス ---
b = np.array([1.0, -1.0])  # shape = (2,)

# 手計算風の実装
y1 = x[0] * W[0, 0] + x[1] * W[1, 0] + b[0]
y2 = x[0] * W[0, 1] + x[1] * W[1, 1] + b[1]

y = np.array([y1, y2])
print(y)
```

---

# ⭐ **例題コード（中級：行列積を使って書く）**

```python
import numpy as np

x = np.array([1.0, 2.0])           # shape = (2,)
W = np.array([[3.0, 1.0],
              [2.0, 4.0]])          # shape = (2,2)
b = np.array([1.0, -1.0])           # shape = (2,)

# 行列積 + バイアス
y = x @ W + b
print(y)
```

**出力：**

```
[8. 8.]
```

---

# 💡 **なぜ @（matmul）で同じ結果になるのか？**

行列積：

[
xW =
[1,2]
\begin{bmatrix}
3 & 1 \
2 & 4
\end{bmatrix}
=============

[1\cdot 3 + 2\cdot 2,;; 1\cdot 1 + 2\cdot 4]
]

その後、バイアスを足しているだけです。

---

# 🏋️‍♂️ **STEP 4：ミニバッチに拡張してみよう（複数データ）**

これも初心者にとって大切な理解です。

例：2つのデータを同時に入力

```python
X = np.array([
    [1.0, 2.0],   # データ1
    [0.5, -1.0]   # データ2
])  # shape = (2データ, 2次元)

y = X @ W + b
print(y)
```

---

# 🎉 **ここまでで理解できること**

* 全結合層は「入力 × 重み + バイアス」をしているだけ
* NumPy では `x @ W + b` で簡単に実装できる
* 重みの shape が重要（in × out）
* 複数データは「行列 × 行列」で処理できる（ミニバッチ）

---

# 📘 **さらに理解を深める追加例題（希望あれば解説付きで作成できます）**

* バイアスがなかったらどうなる？
* 重みの shape を入れ替えると何が起こる？
* 1データと複数データの shape の違いを確かめよう
* 3次元データ（画像の flatten）を全結合に通す例題
* ReLU をつけて1層MLPを作る mini課題

---

もし希望があれば、

* **ワークシート形式の練習問題集**
* **授業スライド風の詳しい解説**
* **計算グラフつきの発展問題**
  も作成できます。

追加しますか？




# 計算例題

はい、PyTorchを使わずに、NumPyだけを使って全結合層（線形変換）の計算ロジックを理解できる例題を作成します。

この例題では、前の質問で使った\*\*「ラーメン屋さんの満足度予測」\*\*のロジックをNumPyの行列計算で再現します。

-----

## 🍜 例題：NumPyで再現する「全結合層」

全結合層の核心は、入力ベクトル $x$ と重み行列 $W$ の**行列の積**（ドット積）に、バイアスベクトル $b$ を足すという計算です。

$$y = x \cdot W^T + b$$

### 1\. 定義（あなたの好み）

| 要素 | Python変数 | 値 | 役割 |
| :--- | :--- | :--- | :--- |
| **重み** | `W` (Weights) | `[+10, -5, -10]` | 味は重要（+）、価格と行列はマイナス（-）評価。 |
| **バイアス** | `b` (Bias) | `+50` | ラーメン愛（基礎点）。 |

### 2\. データ（お店の情報）

| お店 | 味 ($x_1$) | 価格 ($x_2$) | 行列 ($x_3$) |
| :--- | :--- | :--- | :--- |
| **A店** | 10.0 | 1.0 | 1.0 |
| **B店** | 3.0 | 0.5 | 0.0 |

-----

## 💻 計算ロジックの実装（NumPy）

```python
import numpy as np

# --- 1. 定義（学習されたパラメータ） ---

# 重み行列 W: 形状 (出力の次元数, 入力の次元数)
# 今回は出力が「満足度」の1次元なので、(1, 3)の行列として定義します。
W = np.array([[10.0, -5.0, -10.0]])

# バイアスベクトル b: 形状 (出力の次元数)
# 出力が1次元なので、要素が1つのベクトルです。
b = np.array([50.0])

print(f"重み W の形状: {W.shape}")
print(f"バイアス b の形状: {b.shape}")
print("-" * 30)


# --- 2. 入力データの準備 (複数バッチ) ---

# 入力行列 X: 形状 (バッチサイズ, 入力の次元数)
# A店とB店の2つのデータ（バッチサイズ=2）を一度に入力します。
X = np.array([
    [10.0, 1.0, 1.0],  # A店: (味, 価格, 行列)
    [3.0, 0.5, 0.0]    # B店: (味, 価格, 行列)
])

print(f"入力 X の形状: {X.shape} (2バッチ, 3特徴量)")
print("-" * 30)


# --- 3. 全結合層の計算 (線形変換) ---

# 処理 1: 行列の積 (X と W の転置 W.T)
# X (2, 3) と W.T (3, 1) のドット積の結果は (2, 1) になります。
# 転置 (W.T) を使うことで、Torchの nn.Linear と同じ計算順序になります。
Z = np.dot(X, W.T) 

# 処理 2: バイアス b の加算
# Z (2, 1) と b (1,) の間でブロードキャストが行われ、各行にバイアスが加算されます。
Y = Z + b

# --- 4. 結果の表示 ---

print("計算結果 Z (Wx):")
print(Z)
print("\n最終出力 Y (Wx + b):")
print(Y)

print("\n--- 結論 ---")
print(f"A店の満足度: {Y[0, 0]:.1f}")
print(f"B店の満足度: {Y[1, 0]:.1f}")
```

### 🧠 計算ロジックの解説

このコードが示しているのは、全結合層が**効率的な行列演算**を通じて複数の入力を処理していることです。

1.  **行列の積 (`np.dot(X, W.T)`)**:
      * これは、A店のデータ $\begin{pmatrix} 10.0 & 1.0 & 1.0 \end{pmatrix}$ と重み $\begin{pmatrix} 10.0 \\ -5.0 \\ -10.0 \end{pmatrix}$ の内積（ドット積）を一発で計算しています。
      * この計算により、**重み付けされた合計**（$\sum w_i x_i$）が一度に実行されます。
2.  **ブロードキャストによる加算 (`Y = Z + b`)**:
      * 計算結果 $Z$（重み付けされた合計）の各要素に、バイアス $b$ の値 **50.0** が自動的に足し算されます。

この**行列計算**こそが、ディープラーニングモデルが数百万、数十億ものパラメータ（重み）を持つにもかかわらず、高速に学習・推論できる根本的な理由です。



# 体感型例題

全結合層（Fully Connected Layer）の働きを直感的に理解するには、**「ラーメン屋さんの『満足度』スコア予測」**という例題が非常に分かりやすくておすすめです。

全結合層の役割は、**「複数の入力データに対して、それぞれの重要度（重み）を加味して、一つの結論（スコア）を出すこと」**です。

これをラーメンの評価に例えてみましょう。

---

## 🍜 例題：ラーメン屋さんの「満足度」を予測しよう

あなたは新しいラーメン屋さんに行こうとしています。その店に行くかどうかを決めるための**「満足度スコア」**を、全結合層の計算式を使って出してみましょう。

### 1. 入力データ (**$x$**)：お店の情報

お店には3つの特徴（入力）があります。

1. **味の濃さ (**$x_1$**)** : 10点満点
2. **価格 (**$x_2$**)** : 単位は千円（例: 1.0 = 1000円）
3. **行列の待ち時間 (**$x_3$**)** : 単位は時間（例: 0.5 = 30分）

### 2. 重み (**$W$**)：あなたの「こだわり」

全結合層における「重み」とは、**あなたが何を重視するか**という「性格」や「こだわり」のことです。

* **味 (**$w_1$**)** : とにかく味が大事！ **$\rightarrow$** **+10** （正の大きな値）
* **価格 (**$w_2$**)** : 高いのは嫌だ！ **$\rightarrow$** **-5** （負の値）
* **行列 (**$w_3$**)** : 待つのは大嫌いだ！ **$\rightarrow$** **-10** （負の大きな値）

### 3. バイアス (**$b$**)：あなたの「ラーメン愛」

入力がすべてゼロ（味も価格も待ち時間もない虚無の状態）だったとしても、そもそもあなたがどれくらいラーメンが好きかという**下駄（ゲタ）**を履かせます。

* **基礎点 (**$b$**)** : ラーメンというだけで嬉しい **$\rightarrow$** **+50**

---

### 🧮 全結合層の計算（線形変換）

全結合層の中で行われている計算は、これだけです。

$$
\text{満足度} = (\text{味} \times \text{重み}) + (\text{価格} \times \text{重み}) + (\text{行列} \times \text{重み}) + \text{バイアス}
$$

$$
y = (x_1 w_1) + (x_2 w_2) + (x_3 w_3) + b
$$

#### 具体的なお店「A店」の場合

* 味: **10点** (最高！)
* 価格: **1.0** (1000円)
* 行列: **1.0** (1時間待ち)

計算してみましょう：

$$
y = (10 \times 10) + (1.0 \times -5) + (1.0 \times -10) + 50
$$

$$
y = 100 - 5 - 10 + 50
$$

$$
y = \mathbf{135}
$$

**$\rightarrow$** **スコア 135点！** 並んででも食べる価値あり！

#### 具体的なお店「B店」の場合

* 味: **3点** (いまいち)
* 価格: **0.5** (500円・安い)
* 行列: **0** (待ちなし)

計算してみましょう：

$$
y = (3 \times 10) + (0.5 \times -5) + (0 \times -10) + 50
$$

$$
y = 30 - 2.5 - 0 + 50
$$

$$
y = \mathbf{77.5}
$$

**$\rightarrow$** **スコア 77.5点。** まあ、安くて早いなら行ってもいいかな。

---

## 💻 Pythonコードで確認

この「ラーメン評価脳」をPyTorchの全結合層（`nn.Linear`）で作ってみましょう。

**Python**

```
import torch
import torch.nn as nn

# 1. 全結合層を定義
# 入力が3つ（味、価格、行列）、出力が1つ（満足度）
fc_layer = nn.Linear(3, 1)

# 2. 「重み」と「バイアス」を手動で設定（＝あなたの好みを注入）
# 通常はここをAIが学習しますが、今回は手動でセットします
with torch.no_grad():
    fc_layer.weight.data = torch.tensor([[10.0, -5.0, -10.0]]) # [味, 価格, 行列]
    fc_layer.bias.data = torch.tensor([50.0])                  # ラーメン愛

# 3. お店のデータ（入力）を作成
# A店: 味10, 価格1.0(1000円), 行列1.0(1時間)
shop_a = torch.tensor([[10.0, 1.0, 1.0]])

# B店: 味3, 価格0.5(500円), 行列0
shop_b = torch.tensor([[3.0, 0.5, 0.0]])

# 4. 全結合層に通す（計算実行！）
score_a = fc_layer(shop_a).item()
score_b = fc_layer(shop_b).item()

print(f"A店の満足度: {score_a} 点")
print(f"B店の満足度: {score_b} 点")
```

### 🎓 この例題から学ぶべきこと

1. **全結合層の仕事** : 入力データに対して「重み付け（掛け算）」をして「合計」すること。
2. **重みの意味** : 「重み」は、その入力要素が結果に対して**どれくらい重要か（プラスかマイナスか）**を表していること。
3. **学習とは** : AIの学習とは、大量のデータ（食べたラーメンの記録と実際の満足度）を使って、この**最適な「重み（**$W$**）」と「バイアス（**$b$**）」を自動で見つけ出す作業**のことです。


# 全結合層の実装

以下は **「NumPyだけで全結合層（Fully Connected Layer / Linear Layer）を実装し、かつ計算グラフの仕組みも理解できる」**ように設計した演習課題です。
初心者でも取り組めるように段階構成で、最終的には自作の「自動微分のミニ計算グラフ」が完成します。

---

# 🧪 **演習：NumPyで計算グラフを持つ全結合層を実装しよう**

## 🎯 **ゴール**

* NumPyで最小限の計算グラフを実装できる
* 前向き計算（forward）、後ろ向き計算（backward）を自作できる
* 全結合層（Linear Layer）を構築し、誤差逆伝播できる
* 最終的に「1層のミニMLP」を学習できる

---

# 🔰 **ステップ0：準備**

NumPyがあればOK。

```python
import numpy as np
```

---

# 🧩 **ステップ1：計算ノード（Variable）を作ろう**

計算グラフの基本は「ノード（値 + 勾配 + どの演算で生まれたか）」。

### ▼ 演習①：次の機能を持つ Variable クラスを作れ

* `value`：実際の数値（np.ndarray）
* `grad`：勾配（同じ形）
* `op`：どの演算で生成されたか（足し算, 行列積など）
* `parents`：親ノード（a+b なら a と b）

### ✔︎ サンプル回答（モデル解の一例）

```python
class Variable:
    def __init__(self, value, parents=None, op=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.parents = parents
        self.op = op

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.value)

        self.grad += grad

        if self.op is not None:
            self.op.backward(self, grad)
```

---

# 🧮 **ステップ2：演算（Op）クラスを作ろう**

ここでは「加算」と「行列積」の計算グラフを作ろう。

---

### ▼ 演習②：加算(Add)演算を作ろう

* forward：a.value + b.value
* backward：

  * dL/da = grad
  * dL/db = grad

### ✔︎ モデル解

```python
class AddOp:
    def forward(self, a, b):
        out = Variable(a.value + b.value, parents=(a, b), op=self)
        return out

    def backward(self, out, grad):
        a, b = out.parents
        a.backward(grad)
        b.backward(grad)
```

---

### ▼ 演習③：行列積(MatMul)演算を作ろう

* forward：a.value @ b.value
* backward：

  * dL/da = grad @ bᵀ
  * dL/db = aᵀ @ grad

### ✔︎ モデル解

```python
class MatMulOp:
    def forward(self, a, b):
        out = Variable(a.value @ b.value, parents=(a, b), op=self)
        return out

    def backward(self, out, grad):
        a, b = out.parents
        a.backward(grad @ b.value.T)
        b.backward(a.value.T @ grad)
```

---

# 🧱 **ステップ3：全結合層（Linear Layer）を作ろう**

Linear Layer：
[
y = xW + b
]

ここでは

* `xW` → MatMul
* `+ b` → Add

なので上で作った演算を使えば作れる。

---

### ▼ 演習④：Linear Layer の forward を作ろう

### ✔︎ モデル解

```python
class Linear:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = Variable(np.random.uniform(-limit, limit, (in_features, out_features)))
        self.b = Variable(np.zeros(out_features))

        self.matmul = MatMulOp()
        self.add = AddOp()

    def forward(self, x):
        out = self.matmul.forward(x, self.W)
        out = self.add.forward(out, self.b)
        return out
```

---

# 📉 **ステップ4：損失関数（MSE）を作ろう**

[
L = \frac{1}{N} \sum (y - t)^2
]

---

### ▼ 演習⑤：MSE Loss の forward/backward を作れ

### ✔︎ モデル解

```python
class MSELoss:
    def forward(self, pred, target):
        diff = pred.value - target.value
        out = Variable(np.mean(diff ** 2), parents=(pred, target), op=self)
        return out

    def backward(self, out, grad):
        pred, target = out.parents
        diff = pred.value - target.value
        N = diff.size
        pred.backward(grad * (2 / N) * diff)
```

---

# 🤖 **ステップ5：実際に学習させてみる**

今回は

* 入力2次元
* 出力1次元
* ダミーデータで y = 2x₁ - 3x₂ を学習

---

### ▼ 演習⑥：以下を完成させて学習せよ

```python
np.random.seed(0)

# データ作成
X = Variable(np.random.randn(100, 2))
true_W = np.array([[2.0], [-3.0]])
y_true = Variable(X.value @ true_W)

# モデル作成
linear = Linear(2, 1)
loss_fn = MSELoss()

# 学習ループ
lr = 0.1
for epoch in range(200):

    pred = linear.forward(X)
    loss = loss_fn.forward(pred, y_true)

    # 勾配初期化
    linear.W.grad = np.zeros_like(linear.W.grad)
    linear.b.grad = np.zeros_like(linear.b.grad)

    loss.backward()

    # SGD
    linear.W.value -= lr * linear.W.grad
    linear.b.value -= lr * linear.b.grad

    if epoch % 20 == 0:
        print(epoch, loss.value)

print("Learned W:", linear.W.value)
print("Learned b:", linear.b.value)
```

---

# 🎉 **最終的に得られるもの**

* 自作の計算グラフ
* 自作の自動微分
* NumPyだけで構築した全結合層
* 学習可能なミニMLP

---

# 📌 必要なら追加演習も作れます！

例：

* ReLUの追加
* Sigmoidの計算グラフ
* 多層MLP化
* ミニバッチ対応
* 計算グラフの可視化（Graphviz風）


