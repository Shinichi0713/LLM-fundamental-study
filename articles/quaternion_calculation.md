---
title: "Calculation Quaternion"
emoji: "👩‍👩‍👦"
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["sensing", "imu", "self-positioning"]
published: true
---
クォータニオンの掛け算をする計算で $\otimes$ という計算を行うことがあります。
この計算がどんな演算を行っているかということについて説明していきます。

## どんな場面で使うか

クォータニオンの積 $\otimes$ は、**「回転を合成する」** 場面で必ず現れます。以下、IMUの自己位置推定を中心に、具体的な場面を挙げて説明します。

### 1. 姿勢の更新（IMUの角速度から姿勢を求める）

例えばIMUから得られた角速度 $\boldsymbol{\omega}$ をもとに、微小時間 $\Delta t$ 後の姿勢を求める場面です。

- 現在の姿勢：$\mathbf{q}(t)$
- 微小回転：$\Delta\mathbf{q}$（角速度 $\boldsymbol{\omega}$ から作られるクォータニオン）

次の時刻の姿勢は、**現在の姿勢に微小回転を合成**することで得られます。

$$
\mathbf{q}(t+\Delta t) = \mathbf{q}(t) \otimes \Delta\mathbf{q}
$$

ここで $\otimes$ は「現在の向きに、さらに微小回転を加える」という意味です。

**直感的な例**: 東を向いている体が、さらに右に10度回る → 現在の姿勢クォータニオンに、10度右回転のクォータニオンを積する。


### 2. 座標系の変換（センサ座標系 → 世界座標系）

IMUの加速度センサーは、**センサが付いている物体の座標系（ボディ座標系）** で加速度を測ります。しかし、自己位置を推定するには、**世界座標系（慣性座標系）** での加速度が必要です。

この変換にクォータニオンの積が使われます。

$$
\begin{bmatrix} 0 \\ \mathbf{a}_{\text{world}} \end{bmatrix} = \mathbf{q} \otimes \begin{bmatrix} 0 \\ \mathbf{a}_{\text{body}} \end{bmatrix} \otimes \mathbf{q}^*
$$

ここでは、**クォータニオンを左から掛け、共役クォータニオンを右から掛ける**という形で、ベクトルを回転させています。

**直感的な例**: 傾いたスマートフォンの加速度計が「上方向に9.8m/s²」と読んでいても、世界から見ると「真下に重力がある」だけかもしれない。クォータニオンで回転変換することで、世界座標系での意味がわかる。


### 3. 複数の回転の合成

ロボットやドローンの運動では、複数の回転が連続して起こります。

- 例：ロボットの「関節の回転」＋「胴体全体の回転」＋「地形に対する傾き」

これらを組み合わせて、最終的な姿勢を求める際にクォータニオンの積を使います。

$$
\mathbf{q}_{\text{total}} = \mathbf{q}_{\text{地形}} \otimes \mathbf{q}_{\text{胴体}} \otimes \mathbf{q}_{\text{関節}}
$$

**重要な点**: 積の順番が結果を変える（非可換性）。回転を適用する順序を間違えると、全く異なる姿勢になります。

### 4. 姿勢の補間（3DCGやアニメーション）

ゲームやCGアニメーションでもクォータニオンの積は使われます。

- キャラクターの姿勢を「姿勢Aから姿勢Bへスムーズに移動させる」場合、クォータニオンの球面線形補間（SLERP）を使います。
- SLERPの内部では、2つのクォータニオンの「差分」を求めるために積やその逆演算が使われます。


### 5. 姿勢推定のセンサーフュージョン（補正）

IMUだけでは誤差が蓄積するため、カルマンフィルタなどで他のセンサー情報と融合します。

- 予測された姿勢：$\hat{\mathbf{q}}$
- 観測された姿勢（例：カメラやGPSから得られる）：$\mathbf{q}_{\text{obs}}$

これらの「差分」を求める際、クォータニオンの積と逆元（共役）が使われます。

$$
\Delta\mathbf{q} = \mathbf{q}_{\text{obs}} \otimes \hat{\mathbf{q}}^{-1}
$$

この $\Delta\mathbf{q}$ が予測と観測のズレを表します。


## クォータニオン積

クォータニオン（四元数）の積 $\otimes$ は、**ハミルトン積（Hamilton product）** と呼ばれる計算です。

以下、2つのクォータニオンを使って、具体的にどう計算するかを示します。


### 1. クォータニオンの積の定義

2つのクォータニオンを次のように定義します。

$$
\mathbf{q} = q_0 + q_1 i + q_2 j + q_3 k = \begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \end{bmatrix}, \quad
\mathbf{p} = p_0 + p_1 i + p_2 j + p_3 k = \begin{bmatrix} p_0 \\ p_1 \\ p_2 \\ p_3 \end{bmatrix}
$$

ここで、$i, j, k$ は虚数単位で、以下の**ハミルトンの関係式**を満たします。

$$
i^2 = j^2 = k^2 = ijk = -1
$$

これから導かれる積のルールは：

| 積 | 結果 |
|---|---|
| $ij = k$ | $ji = -k$ |
| $jk = i$ | $kj = -i$ |
| $ki = j$ | $ik = -j$ |

（順番を入れ替えると符号が反転します。これが**非可換性**です。）


### 2. 積の展開（代数形式）

$\mathbf{q} \otimes \mathbf{p}$ を分配法則で展開すると、次のようになります。

$$
\mathbf{q} \otimes \mathbf{p} = (q_0 + q_1 i + q_2 j + q_3 k)(p_0 + p_1 i + p_2 j + p_3 k)
$$

実部（スカラ部分）と虚部（ベクトル部分）に分けると：

$$
\mathbf{q} \otimes \mathbf{p} = \underbrace{(q_0 p_0 - q_1 p_1 - q_2 p_2 - q_3 p_3)}_{\text{実部}} + \underbrace{(q_0 \mathbf{p}_v + p_0 \mathbf{q}_v + \mathbf{q}_v \times \mathbf{p}_v)}_{\text{虚部（ベクトル）}}
$$

ここで、$\mathbf{q}_v = [q_1, q_2, q_3]^T$、$\mathbf{p}_v = [p_1, p_2, p_3]^T$ で、$\times$ は**3次元ベクトルの外積（クロス積）** です。

### 3. 行列形式（実用的な計算式）

プログラムで計算する際は、以下の行列・ベクトル積の形が便利です。

$$
\mathbf{q} \otimes \mathbf{p} = \begin{bmatrix}
q_0 p_0 - q_1 p_1 - q_2 p_2 - q_3 p_3 \\
q_0 p_1 + q_1 p_0 + q_2 p_3 - q_3 p_2 \\
q_0 p_2 - q_1 p_3 + q_2 p_0 + q_3 p_1 \\
q_0 p_3 + q_1 p_2 - q_2 p_1 + q_3 p_0
\end{bmatrix}
$$

これは、$\mathbf{q}$ を $4 \times 4$ 行列にしたものと $\mathbf{p}$ の積としても表せます。

$$
\mathbf{q} \otimes \mathbf{p} = \mathbf{Q}(\mathbf{q}) \mathbf{p} = \begin{bmatrix}
q_0 & -q_1 & -q_2 & -q_3 \\
q_1 & q_0 & -q_3 & q_2 \\
q_2 & q_3 & q_0 & -q_1 \\
q_3 & -q_2 & q_1 & q_0
\end{bmatrix} \begin{bmatrix} p_0 \\ p_1 \\ p_2 \\ p_3 \end{bmatrix}
$$


### 4. 具体的な数値例

例として、2つのクォータニオンを掛けてみます。

$$
\mathbf{q} = \begin{bmatrix} 1 \\ 2 \\ 0 \\ 0 \end{bmatrix}, \quad
\mathbf{p} = \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix}
$$

行列形式で計算すると：

$$
\mathbf{q} \otimes \mathbf{p} = \begin{bmatrix}
1 & -2 & 0 & 0 \\
2 & 1 & 0 & 0 \\
0 & 0 & 1 & -2 \\
0 & 0 & 2 & 1
\end{bmatrix} \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} -2 \\ 1 \\ 1 \\ 2 \end{bmatrix}
$$

順番を逆にすると 
$$\mathbf{p} \otimes \mathbf{q}$$
が出てきます。

$$
\mathbf{p} \otimes \mathbf{q} = \begin{bmatrix}
0 & -1 & -1 & 0 \\
1 & 0 & 0 & -1 \\
1 & 0 & 0 & 0 \\
0 & 1 & -1 & 0
\end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} -2 \\ 1 \\ 1 \\ -2 \end{bmatrix}
$$

結果が異なります。これがクォータニオンの積の**非可換性**です。


### 5. 自己位置推定で使う場面

先ほどの導出で出てきた式：

$$
\mathbf{v}' = \mathbf{q} \otimes \begin{bmatrix} 0 \\ \mathbf{v} \end{bmatrix} \otimes \mathbf{q}^*
$$

これは、3次元ベクトル $\mathbf{v}$ をクォータニオン $\mathbf{q}$ で回転させる操作です。

- まず $\mathbf{q} \otimes \begin{bmatrix} 0 \\ \mathbf{v} \end{bmatrix}$ を計算（純虚クォータニオンを左から掛ける）
- 次にその結果に $\mathbf{q}^*$（共役クォータニオン）を右から掛ける

この「左から掛けて、右から共役を掛ける」という形が、3次元ベクトルを回転させる標準的な方法です。


## 計算例

クォータニオンの積を、**実際の数値を使って1つずつ展開**してみます。

### 例1: 基本的なハミルトン積

2つのクォータニオンを次のように定義します。

$$
\mathbf{q} = \begin{bmatrix} 2 \\ 3 \\ 0 \\ 1 \end{bmatrix} = 2 + 3i + 0j + 1k, \quad
\mathbf{p} = \begin{bmatrix} 1 \\ 2 \\ 1 \\ 0 \end{bmatrix} = 1 + 2i + 1j + 0k
$$

積 $\mathbf{q} \otimes \mathbf{p}$ は、**分配法則**で展開します。

$$
\mathbf{q} \otimes \mathbf{p} = (2 + 3i + 0j + 1k)(1 + 2i + 1j + 0k)
$$

各項を掛け合わせます（$i^2 = j^2 = k^2 = -1$、$ij = k$、$jk = i$、$ki = j$ を使います）。

| 項 | 計算 | 結果 |
|---|---|---|
| $2 \times 1$ | $2$ | $2$ |
| $2 \times 2i$ | $4i$ | $4i$ |
| $2 \times 1j$ | $2j$ | $2j$ |
| $2 \times 0k$ | $0$ | $0$ |
| $3i \times 1$ | $3i$ | $3i$ |
| $3i \times 2i$ | $6i^2$ | $-6$ |
| $3i \times 1j$ | $3ij$ | $3k$ |
| $3i \times 0k$ | $0$ | $0$ |
| $0j \times 1$ | $0$ | $0$ |
| $0j \times 2i$ | $0$ | $0$ |
| $0j \times 1j$ | $0$ | $0$ |
| $0j \times 0k$ | $0$ | $0$ |
| $1k \times 1$ | $k$ | $k$ |
| $1k \times 2i$ | $2ki$ | $2j$ |
| $1k \times 1j$ | $kj$ | $-i$ |
| $1k \times 0k$ | $0$ | $0$ |

これらを**実部・$i$・$j$・$k$ ごとにまとめる**と：

- **実部**: $2 - 6 = -4$
- **$i$**: $4 + 3 - 1 = 6$
- **$j$**: $2 + 2 = 4$
- **$k$**: $3 + 1 = 4$

したがって：

$$
\mathbf{q} \otimes \mathbf{p} = \begin{bmatrix} -4 \\ 6 \\ 4 \\ 4 \end{bmatrix}
$$

### 例2: 行列形式での計算（プログラムと同じ方法）

先ほどの例を、C++コードと同じ**行列形式**で計算してみます。

$$
\mathbf{q} \otimes \mathbf{p} = \mathbf{Q}(\mathbf{q}) \mathbf{p}
$$

ここで、$\mathbf{Q}(\mathbf{q})$ は：

$$
\mathbf{Q}(\mathbf{q}) = \begin{bmatrix}
2 & -3 & 0 & -1 \\
3 & 2 & -1 & 0 \\
0 & 1 & 2 & -3 \\
1 & 0 & 3 & 2
\end{bmatrix}
$$

これに $\mathbf{p} = [1, 2, 1, 0]^T$ を掛けます。

$$
\begin{bmatrix}
2 & -3 & 0 & -1 \\
3 & 2 & -1 & 0 \\
0 & 1 & 2 & -3 \\
1 & 0 & 3 & 2
\end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix}
2 \times 1 + (-3) \times 2 + 0 \times 1 + (-1) \times 0 \\
3 \times 1 + 2 \times 2 + (-1) \times 1 + 0 \times 0 \\
0 \times 1 + 1 \times 2 + 2 \times 1 + (-3) \times 0 \\
1 \times 1 + 0 \times 2 + 3 \times 1 + 2 \times 0
\end{bmatrix} = \begin{bmatrix} -4 \\ 6 \\ 4 \\ 4 \end{bmatrix}
$$

展開した結果と一致します。

### 例3: 非可換性の確認（順番を変えると結果が変わる）

同じ $\mathbf{q}$、$\mathbf{p}$ で、順番を逆にして $\mathbf{p} \otimes \mathbf{q}$ を計算します。

行列形式で：

$$
\mathbf{P}(\mathbf{p}) = \begin{bmatrix}
1 & -2 & -1 & 0 \\
2 & 1 & 0 & -1 \\
1 & 0 & 1 & -2 \\
0 & 1 & -1 & 1
\end{bmatrix}
$$

$$
\mathbf{P}(\mathbf{p}) \mathbf{q} = \begin{bmatrix}
1 & -2 & -1 & 0 \\
2 & 1 & 0 & -1 \\
1 & 0 & 1 & -2 \\
0 & 1 & -1 & 1
\end{bmatrix} \begin{bmatrix} 2 \\ 3 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix}
1 \times 2 + (-2) \times 3 + (-1) \times 0 + 0 \times 1 \\
2 \times 2 + 1 \times 3 + 0 \times 0 + (-1) \times 1 \\
1 \times 2 + 0 \times 3 + 1 \times 0 + (-2) \times 1 \\
0 \times 2 + 1 \times 3 + (-1) \times 0 + 1 \times 1
\end{bmatrix} = \begin{bmatrix} -4 \\ 6 \\ 0 \\ 4 \end{bmatrix}
$$

比較すると：

$$
\mathbf{q} \otimes \mathbf{p} = \begin{bmatrix} -4 \\ 6 \\ 4 \\ 4 \end{bmatrix}, \quad
\mathbf{p} \otimes \mathbf{q} = \begin{bmatrix} -4 \\ 6 \\ 0 \\ 4 \end{bmatrix}
$$

**3番目の要素（$j$成分）が異なります。** これが非可換性です。

### 例4: 回転の合成（IMUで実際に使われる場面）

IMUの姿勢更新で使われる形を数値で確認します。

現在の姿勢を「Z軸周りに30度回転」とします。

$$
\mathbf{q} = \begin{bmatrix} \cos 15^\circ \\ 0 \\ 0 \\ \sin 15^\circ \end{bmatrix} \approx \begin{bmatrix} 0.9659 \\ 0 \\ 0 \\ 0.2588 \end{bmatrix}
$$

微小回転を「Z軸周りに10度（$\approx$ 0.1745 rad）」とします。微小時間 $\Delta t$ での角速度に対応するクォータニオンは：

$$
\Delta\mathbf{q} \approx \begin{bmatrix} 1 \\ 0 \\ 0 \\ \frac{0.1745}{2} \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0.0873 \end{bmatrix}
$$

姿勢の更新は：

$$
\mathbf{q}_{\text{new}} = \mathbf{q} \otimes \Delta\mathbf{q}
$$

行列形式で計算します。

$$
\mathbf{q}_{\text{new}} = \begin{bmatrix}
0.9659 & 0 & 0 & -0.2588 \\
0 & 0.9659 & -0.2588 & 0 \\
0 & 0.2588 & 0.9659 & 0 \\
0.2588 & 0 & 0 & 0.9659
\end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0.0873 \end{bmatrix} = \begin{bmatrix}
0.9659 - 0.0226 \\ 0 \\ 0 \\ 0.2588 + 0.0843
\end{bmatrix} = \begin{bmatrix} 0.9433 \\ 0 \\ 0 \\ 0.3431 \end{bmatrix}
$$

正規化後の回転角を確認すると：

$$
\theta_{\text{new}} = 2 \arctan\left(\frac{0.3431}{0.9433}\right) \approx 2 \times 20^\circ = 40^\circ
$$

**30度 + 10度 = 40度** となり、回転が正しく合成されています。

### 計算順序

クォータニオンの積は、**「2つの回転を1つにまとめる」**操作です。数値を実際に追うと：

1. **分配法則で展開**し、$i^2 = j^2 = k^2 = -1$ と $ij = k$ などのルールで整理する
2. **行列形式**にするとプログラムへの実装が容易になる
3. **順番を変えると結果が変わる**（非可換性）ため、回転の適用順序に注意が必要

もっと複雑な例（例えば3軸同時回転の合成など）が必要でしたら、お知らせください。


ということで本日はクォータニオン積について扱いました。

