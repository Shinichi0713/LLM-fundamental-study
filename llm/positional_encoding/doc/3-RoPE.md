# RoPE

RoPE（Rotary Positional Embedding）は、**相対位置関係を自然にAttention内で扱える位置エンコーディング**です。

Transformer 系モデル（GPT-NeoX, LLaMA, Mistral など）でよく使われています。

結論から言うと：

> **埋め込みベクトルを周波数ごとに回転させて位置情報を与える方式**

です。

> RoPEはAttention内で相対位置関係を捉える位置エンコーディング。
>
> LLaMaなどれよく使われる。


## ✅ RoPEの数式と直感

入力ベクトル $x$ を偶数次元ごとに2次元ペア $(x_{2i}, x_{2i+1})$ で扱い、角度 $\theta_{pos,i}$ だけ回転させます：

$$
\theta_{pos,i} = pos / 10000^{2i/d}
$$


```math
\begin{pmatrix}

x' *{2i} \

x'* {2i+1}

\end{pmatrix}

\begin{pmatrix}

\cos \theta_{pos,i} & -\sin \theta_{pos,i} \

\sin \theta_{pos,i} & \cos \theta_{pos,i}

\end{pmatrix}

\begin{pmatrix}

x_{2i} \

x_{2i+1}

\end{pmatrix}
```


つまり、**周波数の異なる回転行列を掛ける**

> 各次元ペアが「回転する小さなコンパス」になる

---

## ✅ なぜAttentionと相性が良い？

内積計算 $QK^T$ の中で、

$$
\langle R(p)x, R(q)y \rangle
$$

が

$$
= \langle x, R(q-p)y \rangle
$$

になる性質があります。

つまり、

> **距離差 ( q - p ) が Attention 内で自動的に生まれる**

これが **RoPE = 相対位置が自然に出る** 理由。


## ✅ Sinusoidal PE と何が違う？

| 特徴               | Sinusoidal PE | RoPE           |
| ------------------ | ------------- | -------------- |
| 位置ベクトルを加算 | ✅            | ❌             |
| ベクトルを回転     | ❌            | ✅             |
| 相対位置が直接出る | 半分（近似）  | ✅ 厳密        |
| 長文に強い         | 普通          | ✅ LLaMAで証明 |
| Attention内で作用  | ❌            | ✅             |

Sinusoidal は「位置ベクトルを足す」

RoPE は「ベクトルそのものを**回転**する」


## ✅ RoPE は Fourier 変換の応用

PE を波で表す理論は同じですが、

* Sinusoidal：**位相を足す**
* RoPE：**ベクトルを回転 = 複素数の掛け算**

複素平面で書くと非常にスッキリ：

$$
x' = x \cdot e^{i\theta}
$$

> Transformer が「角度で距離を感じる」ようになる


## ✅ RoPE のメリット

| メリット               | 内容                     |
| ---------------------- | ------------------------ |
| 相対位置を直接モデル化 | 計算が自然に含まれる     |
| 長距離依存に強い       | LLaMAが実証              |
| 外挿性良い             | 長い文で性能維持         |
| 計算が軽い             | 足し算 → 回転の行列だけ |
| メモリ効率高い         | テーブル不要             |



## ✅ RoPE の弱点

| 弱点                       | 内容                     |
| -------------------------- | ------------------------ |
| 極端に長い文では精度落ちる | → YaRN, NTKなど改良技術 |
| 実装が少し複雑             | sin/cos対を回転          |



## 🧠 直感まとめ

* PE を「音（波）」とするなら
* RoPE は「回転するレーダーアンテナ」

信号の位相差で距離を感じるレーダー原理に近い。


## 📌 一言で説明すると？

> **RoPE は、埋め込みを回転させることで相対位置を表す PE。
>
> 距離情報がAttention内に自然に入る。**


## 計算法
RoPE（Rotary Positional Embedding）は、**トークン埋め込み（Q / K）を位置に応じて回転させる処理**です。

結論：

> **単純に、(2次元ずつ)ベクトルを回転させて位置情報を与える**
> (x \to R(\theta)x) を全次元ペアに適用

---

## ✅ RoPE の内部処理を直感で

埋め込みベクトルを

$$
(x_0, x_1), (x_2, x_3), (x_4, x_5), ...
$$

のように2次元ずつ分けて、
各ペアを「その位置に応じた角度」で回転させます。

2次元回転行列：

$$
\begin{pmatrix}
\cos\theta & -\sin\theta \
\sin\theta & \cos\theta
\end{pmatrix}
$$

これで各ペアを回す。

---

## ✅ 処理式（実際の計算）

位置 `p` の埋め込みベクトル (x) を RoPE すると：

$$
x'*{2i} = x*{2i}\cos\theta_{p,i} - x_{2i+1}\sin\theta_{p,i}
$$
$$
x'*{2i+1} = x*{2i}\sin\theta_{p,i} + x_{2i+1}\cos\theta_{p,i}
$$

角度は位置 `p` と次元 `i` で決まる：

$$
\theta_{p,i} = p / 10000^{2i/d}
$$

---

## ✅ なぜ相対位置が分かる？

Attention で QK^T を計算するとき、

$$
\langle R(p)Q, R(q)K \rangle
============================

\langle Q, R(q-p)K \rangle
$$

つまり：

> **Q と K に位置回転を与えると、内積が距離差 (q - p) を表す**

これが RoPE が **相対位置を自然に学習**できる理由。

---

## ✅ 処理フロー（擬似コード）

```python
# Q, K shape: $$batch, seq, dim$$
# split dim into 2 channels and rotate
Q_rot = rotate_every_two(Q) * cos(theta$$p$$) + rotate_every_two(Q) * sin(theta$$p$$)
K_rot = rotate_every_two(K) * cos(theta$$p$$) + rotate_every_two(K) * sin(theta$$p$$)

score = dot(Q_rot, K_rot.T) / sqrt(d)
```

---

## ✅ なぜ足し算じゃダメなの？

Sinusoidal PE は

> **埋め込みにベクトルを足す**

RoPE は

> **埋め込みを回転(掛け算)**

足し算 → 絶対位置しか使えない
回転 → **位置差（相対位置）が自然に出る**

---

## ✅ イメージ

| Sinusoidal    | RoPE                |
| ------------- | ------------------- |
| 🔢 "数字タグを付ける" | 🔁 "方向と角度を持つ羅針盤にする" |
| ベクトルを動かす      | ベクトルを回す             |
| 絶対位置中心        | 相対位置を自然に表現          |

---

## ✅ まとめ

RoPE は

* Q/Kベクトルを位置ごとに2次元回転
* 回転角は周波数スケールごとに変える
* Attention内で距離差が直接出る → **相対位置が自然に利用**

結果：

> LLaMA, Mistral, GPT-NeoX が RoPE を採用
> 長文性能が劇的向上


# Good point

とても良い質問です。
RoPE（Rotary Position Embedding）は、従来の**絶対位置エンコーディング（absolute positional encoding）**の限界を超えるために導入された手法です。
以下で、数式と直感を交えて「なぜRoPEが優れているか」を整理します👇

---

## 🧩 従来：絶対位置エンコーディングの仕組み

Transformerの初期モデル（BERT、Vaswani et al., 2017）は次のように位置情報を付加していました。

[
x_i' = x_i + p_i
]

ここで

* (x_i)：単語の埋め込みベクトル
* (p_i)：その位置 (i) の位置ベクトル（sin/cos波など）

Attention計算は：
[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
]

この方式では、**位置情報は入力に足し込むだけ**です。
→ モデルは「どこにあるか」は分かりますが、
「**単語間の距離や順序差**」を直接扱えません。

---

## 🚀 RoPE（Rotary Position Embedding）の発想

RoPEは、各トークンのベクトルを位置に応じて**回転**させます。

数式で書くと：

[
\text{RoPE}(x_i) = R_{\theta_i} , x_i
]

ただし、(R_{\theta_i}) は回転行列で、
位置 (i) に応じた角度 (\theta_i) でベクトルを回転させます。

---

## ⚙️ Attentionにおける効果

Attentionスコアは：

[
q_i^\top k_j
]

RoPEを適用すると：

[
(R_{\theta_i} q_i)^\top (R_{\theta_j} k_j)
= q_i^\top R_{\theta_j - \theta_i} k_j
]

🧠 **注目ポイント：**

➡ 内積が **位置差 ((j - i))** のみに依存！

つまり、**「相対的な距離」情報**を自然に持つようになる。

---

## 📈 RoPEの主なメリットまとめ

| 観点           | 絶対位置埋め込み           | RoPE（回転型）                            |
| ------------ | ------------------ | ------------------------------------ |
| 埋め込み方法       | (x_i + p_i) （単純加算） | 回転行列 (R_{\theta_i}) を適用              |
| 表現できる位置情報    | 絶対位置のみ             | 相対位置（距離・順序）                          |
| Attentionの性質 | 内積が位置に依存           | 内積が位置差（距離）に依存                        |
| 長文への汎化       | 弱い（訓練長を超えると劣化）     | 強い（距離の差は変わらない）                       |
| 実装コスト        | 低                  | 少し高（cos/sin計算あり）                     |
| 使われている代表モデル  | BERT, GPT-2        | Llama, GPT-NeoX, ChatGLM, Mistral など |

---

## 🧠 直感的に言うと

* 絶対埋め込みは「この単語は文の5番目にある」としか分からない。
* RoPEは「この単語は**他の単語から何語離れているか**」がわかる。
* だから、**文脈の相対的構造（主語→述語など）**をモデルが理解しやすい。

---

## 💡 一言でまとめると

> **RoPEは「位置そのもの」ではなく「位置の差」を学習可能にする。**
>
> これにより、Transformerは文中の相対的な関係（距離や順序）を直接捉えられるようになり、
> 長文・未知長への汎化性能が大幅に向上します。


