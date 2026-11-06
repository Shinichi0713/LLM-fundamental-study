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

IMUから得られた角速度 $\boldsymbol{\omega}$ をもとに、微小時間 $\Delta t$ 後の姿勢を求める場面です。

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

ということで本日はクォータニオン積について扱いました。

