

## 自己位置推定
IMU（慣性計測装置）から自己位置を推定する際のクォータニオン（四元数）の使い方について、導出を含めて説明いたします。

まず、核心となるアイデアを述べます。**IMUから得られる角速度をもとに、姿勢を表すクォータニオンの時間微分方程式を立て、数値積分することで姿勢を推定します。さらに、その姿勢で重力分を除去した加速度を2回積分することで位置を求めます。**

以下、ステップバイステップで導出していきます。

### 1. なぜクォータニオンなのか

3次元空間での回転を表す方法として、オイラー角（ロール・ピッチ・ヨー）を幾何関係から算出する方法が基本的ですが、この算出法には以下の問題があります。

- **ジンバルロック（Gimbal Lock）**: 特定の姿勢で自由度が1つ失われ、回転が表現できなくなる
- **補間の困難さ**: 2つの姿勢の中間を計算する際に、直感的な補間ができない
- **数値的な問題**: 角速度から姿勢を積分する際、三角関数の計算が多く、計算コストや特異点が生じる

クォータニオンはこれらを回避でき、4次元の単位ベクトルとして回転を表すため、計算が安定します。

### 2. クォータニオンの定義

クォータニオン $\mathbf{q}$ は以下のように定義されます。

$$
\mathbf{q} = q_0 + q_1 i + q_2 j + q_3 k = \begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \end{bmatrix}
$$

ここで、$q_0$ は実部（スカラ部分）、$(q_1, q_2, q_3)$ は虚部（ベクトル部分）です。

単位クォータニオン（回転を表す場合）は、回転軸を $\mathbf{n} = [n_x, n_y, n_z]^T$（ただし $\|\mathbf{n}\| = 1$）、回転角を $\theta$ とすると：

$$
\mathbf{q} = \begin{bmatrix} \cos\frac{\theta}{2} \\ n_x \sin\frac{\theta}{2} \\ n_y \sin\frac{\theta}{2} \\ n_z \sin\frac{\theta}{2} \end{bmatrix}
$$

これにより、3次元ベクトル $\mathbf{v}$ の回転は、クォータニオンを用いて以下のように表されます。

$$
\mathbf{v}' = \mathbf{q} \otimes \begin{bmatrix} 0 \\ \mathbf{v} \end{bmatrix} \otimes \mathbf{q}^*
$$

ここで、$\otimes$ はクォータニオンの積、$\mathbf{q}^*$ は共役クォータニオン（$\mathbf{q}^* = [q_0, -q_1, -q_2, -q_3]^T$）です。


### 3. 角速度とクォータニオンの時間微分

IMUから得られる角速度を、センサ座標系（ボディ座標系）におけるベクトル $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$ とします。

微小時間 $\Delta t$ の間に生じる微小回転角は $\Delta\boldsymbol{\theta} = \boldsymbol{\omega} \Delta t$ です。この微小回転に対応するクォータニオンは：

$$
\Delta\mathbf{q} = \begin{bmatrix} \cos\frac{\|\Delta\boldsymbol{\theta}\|}{2} \\ \frac{\Delta\boldsymbol{\theta}}{\|\Delta\boldsymbol{\theta}\|} \sin\frac{\|\Delta\boldsymbol{\theta}\|}{2} \end{bmatrix}
$$

$\Delta t \to 0$ の極限で、$\|\Delta\boldsymbol{\theta}\| \approx 0$ より $\cos\frac{\|\Delta\boldsymbol{\theta}\|}{2} \approx 1$、$\sin\frac{\|\Delta\boldsymbol{\theta}\|}{2} \approx \frac{\|\Delta\boldsymbol{\theta}\|}{2}$ となるため：

$$
\Delta\mathbf{q} \approx \begin{bmatrix} 1 \\ \frac{\omega_x \Delta t}{2} \\ \frac{\omega_y \Delta t}{2} \\ \frac{\omega_z \Delta t}{2} \end{bmatrix}
$$

時刻 $t$ の姿勢を $\mathbf{q}(t)$ とすると、時刻 $t+\Delta t$ の姿勢はクォータニオンの積によって：

$$
\mathbf{q}(t+\Delta t) = \mathbf{q}(t) \otimes \Delta\mathbf{q}
$$

クォータニオンの積を行列形式で表すと、$\mathbf{q}(t) = [q_0, q_1, q_2, q_3]^T$ に対して：

$$
\mathbf{q}(t+\Delta t) = \mathbf{q}(t) + \frac{\Delta t}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}) \mathbf{q}(t)
$$

ここで、$\boldsymbol{\Omega}(\boldsymbol{\omega})$ は角速度の斜対称行列を拡張した $4 \times 4$ 行列です。

$$
\boldsymbol{\Omega}(\boldsymbol{\omega}) = \begin{bmatrix} 
0 & -\omega_x & -\omega_y & -\omega_z \\
\omega_x & 0 & \omega_z & -\omega_y \\
\omega_y & -\omega_z & 0 & \omega_x \\
\omega_z & \omega_y & -\omega_x & 0
\end{bmatrix}
$$

$\Delta t \to 0$ とすると、**クォータニオンの時間微分方程式**が得られます。

$$
\boxed{\frac{d\mathbf{q}}{dt} = \frac{1}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}) \mathbf{q}}
$$

これが、IMUの角速度から姿勢を更新するための基本方程式です。


### 4. 数値積分による姿勢推定

上記の微分方程式を離散化して数値積分します。最も基本的な方法は**1次のオイラー法**です。

$$
\mathbf{q}_{k+1} = \mathbf{q}_k + \frac{\Delta t}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}_k) \mathbf{q}_k
$$

ただし、このままではクォータニオンのノルムが1からずれていくため、**毎ステップで正規化**する必要があります。

$$
\mathbf{q}_{k+1} \leftarrow \frac{\mathbf{q}_{k+1}}{\|\mathbf{q}_{k+1}\|}
$$

より精度が高い方法として、**指数マップ（Exponential Map）**を用いる方法もあります。

$$
\mathbf{q}_{k+1} = \mathbf{q}_k \otimes \exp\left(\frac{1}{2}\begin{bmatrix} 0 \\ \boldsymbol{\omega}_k \Delta t \end{bmatrix}\right)
$$

ここで、クォータニオンの指数関数は：

$$
\exp(\mathbf{q}) = e^{q_0}\begin{bmatrix} \cos\|\mathbf{v}\| \\ \frac{\mathbf{v}}{\|\mathbf{v}\|}\sin\|\mathbf{v}\| \end{bmatrix}
$$

（$\mathbf{v} = [q_1, q_2, q_3]^T$）

### 5. 位置推定への応用（自己位置推定）

姿勢が求まったら、位置を推定します。IMUから得られる加速度 $\mathbf{a}_{\text{imu}}$（センサ座標系）を、世界座標系（慣性座標系）に変換する必要があります。

世界座標系での加速度 $\mathbf{a}_{\text{world}}$ は、クォータニオンによる回転変換で得られます。

$$
\begin{bmatrix} 0 \\ \mathbf{a}_{\text{world}} \end{bmatrix} = \mathbf{q} \otimes \begin{bmatrix} 0 \\ \mathbf{a}_{\text{imu}} \end{bmatrix} \otimes \mathbf{q}^*
$$

これを行列形式で書くと：

$$
\mathbf{a}_{\text{world}} = \mathbf{R}(\mathbf{q}) \mathbf{a}_{\text{imu}}
$$

ここで、$\mathbf{R}(\mathbf{q})$ はクォータニオンから作られる回転行列です。

$$
\mathbf{R}(\mathbf{q}) = \begin{bmatrix}
q_0^2+q_1^2-q_2^2-q_3^2 & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\
2(q_1 q_2 + q_0 q_3) & q_0^2-q_1^2+q_2^2-q_3^2 & 2(q_2 q_3 - q_0 q_1) \\
2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & q_0^2-q_1^2-q_2^2+q_3^2
\end{bmatrix}
$$

IMUの加速度には重力加速度 $\mathbf{g} = [0, 0, g]^T$ が含まれているため、これを除去します。

$$
\mathbf{a}_{\text{corrected}} = \mathbf{a}_{\text{world}} - \mathbf{g}
$$

あとは、この加速度を2回積分すれば位置が得られます。

$$
\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{a}_{\text{corrected},k} \Delta t
$$

$$
\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2} \mathbf{a}_{\text{corrected},k} \Delta t^2
$$

### 6. まとめ：アルゴリズム全体の流れ

1. **初期化**: $\mathbf{q}_0 = [1, 0, 0, 0]^T$（単位クォータニオン）、$\mathbf{v}_0 = \mathbf{0}$、$\mathbf{p}_0 = \mathbf{0}$
2. **角速度読み取り**: $\boldsymbol{\omega}_k$（ジャイロスコープ）
3. **姿勢更新**: $\mathbf{q}_{k+1} = \mathbf{q}_k + \frac{\Delta t}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}_k)\mathbf{q}_k$、その後正規化
4. **加速度読み取り**: $\mathbf{a}_{\text{imu},k}$（加速度センサー）
5. **座標変換**: $\mathbf{a}_{\text{world}} = \mathbf{R}(\mathbf{q}_{k+1})\mathbf{a}_{\text{imu},k}$
6. **重力補正**: $\mathbf{a}_{\text{corrected}} = \mathbf{a}_{\text{world}} - \mathbf{g}$
7. **速度・位置更新**: 上記の積分式で $\mathbf{v}_{k+1}$、$\mathbf{p}_{k+1}$ を計算
8. **k ← k+1** で2に戻る


### 7. 限界と注意点

- **数値誤差の累積**: IMUだけでの自己位置推定は、バイアスやノイズの影響で誤差が時間とともに二次的に発散します。実用的には、カルマンフィルタやマーダンターフィルタを用いて、GPSや视觉センサーと融合（センサーフュージョン）する必要があります。
- **クォータニオンの正規化**: 数値誤差によりノルムが1からずれるため、正規化は必須です。
- **初期姿勢の推定**: 静止時に重力方向からロール・ピッチを求めることはできますが、ヨー（水平面内の向き）は加速度計だけでは決まらず、磁気コンパスなどが必要です。



