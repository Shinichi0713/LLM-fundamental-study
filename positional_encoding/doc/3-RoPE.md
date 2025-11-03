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



