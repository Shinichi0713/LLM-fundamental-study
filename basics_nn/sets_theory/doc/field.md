
[昨日の環]()に引き続いて四則演算を定義できる集合とはどのようなものか、ということを抽象化した定義である体。
体の定義が固まるとその後に続く線形代数やベクトル演算を厳密に定義できるようになります。

本日テーマ：
>集合論における体の定義をしてみる

## 体

集合論における**体（field）** は、**「集合と2つの演算（足し算・掛け算）の組」** として、集合論の枠組みの中で定義されるものです。

### 集合論的な定義

集合 $K$ と、その上の2つの演算
- $+ : K \times K \to K$（加法）
- $\cdot : K \times K \to K$（乗法）

が与えられ、次の条件を満たすとき、$(K,+,\cdot)$ を**体**といいます。

1. $(K,+)$ は**可換群**（単位元 0 と逆元 $-a$ を持つ）
2. $(K\setminus\{0\},\cdot)$ も**可換群**（単位元 1 と逆元 $a^{-1}$ を持つ）
3. **分配法則**が成り立つ：
   $$
   a \cdot (b + c) = a \cdot b + a \cdot c
   $$

### 集合論との関係

- 体は**集合と写像（演算）の組**として定義されるので、集合論の言葉で厳密に記述できます。
- 体の要素は集合の元であり、部分体・拡大体・同型写像なども集合論的に扱えます。
- 現代数学の多くの分野（線形代数、代数幾何、数論など）は、集合論の上に体の理論を構築しています。

### 体を一言で言うと

> 集合論でいう体＝  
> 「**足し算と掛け算が定義された集合**で、足し算は可換群、0 以外の元は掛け算でも可換群になり、分配法則を満たすもの」

です。

### 体が必要だった理由

体が必要となった理由を、**歴史的・理論的・応用的**な観点から簡潔にまとめます。

__1. 「四則演算が自由にできる世界」を厳密に定義するため__

- 有理数・実数・複素数など、日常的に使う数は、  
  足し算・引き算・掛け算・割り算（0 以外）が自由にできます。
- これを数学的に厳密に扱うために、  
  「**足し算と掛け算の両方が可換群で、分配法則で結びつく集合**」として体を定義した。

__2. 線形代数・ベクトル空間の土台として__

- ベクトル空間は「体上の加群」として定義されます。
- スカラー倍（λv）や基底・次元・線形写像などの概念は、**スカラーが体であること**を前提に成り立ちます。
- 体がないと、線形代数の理論（行列・行列式・固有値など）がきちんと構築できない。

__3. 方程式の解法・代数拡大を扱うため__

- 多項式方程式の解を求めるには、係数が体であることが必要です。
- 代数拡大（例：ℚ(√2)）やガロア理論は、**体の拡大**として記述されます。
- 体の理論により、「どの方程式がべき根で解けるか」などの問題が厳密に扱える。

__4. 幾何（代数幾何）との対応のため__

- 代数幾何では、多項式環のイデアルと代数多様体が対応しますが、  
  その係数は体（ℝ, ℂ, 有限体など）です。
- 体の選択（ℝ か ℂ か有限体か）によって、幾何的な性質が大きく変わる。

__5. 応用（符号・暗号・物理）の基礎として__

- 符号理論：線形符号は**有限体上のベクトル空間**として定義される。
- 暗号：有限体や楕円曲線上の体が公開鍵暗号の基礎。
- 物理：複素数体 ℂ は量子力学の状態空間（ヒルベルト空間）の係数体として不可欠。

__つまり__

> 体が必要だったのは、  
> 「四則演算が自由にできる数の世界」を厳密に定義し、線形代数・方程式論・幾何・符号・暗号・物理など、広い分野で共通の土台として使うため。

です。

### 具体例

**体（field）** の具体例を、**集合と演算**の形で詳しく挙げます。

__1. 有理数体 ℚ__

- **集合**：有理数全体 ℚ = $\{p/q \mid p,q \in \mathbb{Z},\ q \ne 0\}$
- **演算**：
  - 加法：通常の足し算 $+$
  - 乗法：通常の掛け算 $\times$
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 $-a$）
  - 0 以外の有理数は乗法について可換群（単位元 1、逆元 $1/a$）
  - 分配法則が成り立つ
- **特徴**：
  - 最も基本的な**無限体**の一つ
  - 数論・代数の基礎となる体

__2. 実数体 ℝ__

- **集合**：実数全体 ℝ（有理数と無理数を含む）
- **演算**：通常の足し算 $+$ と掛け算 $\times$
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 $-a$）
  - 0 以外の実数は乗法について可換群（単位元 1、逆元 $1/a$）
  - 分配法則が成り立つ
- **特徴**：
  - **完備な順序体**（順序数は9章で扱う概念実数の連続性・極限が定義できる）
  - 解析学（微積分）の舞台

__3. 複素数体 ℂ__

- **集合**：複素数全体 ℂ = $\{a + bi \mid a,b \in \mathbb{R},\ i^2 = -1\}$
- **演算**：複素数の足し算・掛け算
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 $-a-bi$）
  - 0 以外の複素数は乗法について可換群（単位元 1、逆元 $1/(a+bi)$）
  - 分配法則が成り立つ
- **特徴**：
  - **代数的閉体**（すべての多項式が根を持つ）
  - 線形代数・量子力学などで重要

__4. 有限体（ガロア体）𝔽ₚ__

- **集合**：整数を素数 $p$ で割った余りの集合 $\{0,1,2,\dots,p-1\}$
- **演算**：mod $p$ での足し算・掛け算
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 $-a \bmod p$）
  - 0 以外の元は乗法について可換群（単位元 1、逆元は mod $p$ での逆数）
  - 分配法則が成り立つ
- **例**：
  - 𝔽₂ = $\{0,1\}$：足し算は XOR、掛け算は AND
  - 𝔽₃ = $\{0,1,2\}$：mod 3 の演算
- **特徴**：
  - **有限個の元からなる体**
  - 符号理論・暗号・組合せ論で使われる

__5. 有理関数体 ℚ(x)__

- **集合**：有理数係数の多項式の比全体  
  ℚ(x) = $\{f(x)/g(x) \mid f,g \in \mathbb{Q}[x],\ g \ne 0\}$
- **演算**：有理関数の足し算・掛け算（通分・約分）
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 $-f/g$）
  - 0 以外の有理関数は乗法について可換群（単位元 1、逆元 $g/f$）
  - 分配法則が成り立つ
- **特徴**：
  - **無限次元の体拡大**（ℚ の超越拡大）
  - 代数幾何・関数体の理論で重要

__6. 代数体（例：ℚ(√2)）__

- **集合**：ℚ(√2) = $\{a + b\sqrt{2} \mid a,b \in \mathbb{Q}\}$
- **演算**：通常の足し算・掛け算（√2 の性質 √2²=2 を使う）
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 $-a - b\sqrt{2}$）
  - 0 以外の元は乗法について可換群（逆元は共役を用いて計算）
  - 分配法則が成り立つ
- **特徴**：
  - ℚ の**有限次代数拡大体**
  - 数論（代数整数論）で重要

__7. p-進数体 ℚₚ__

- **集合**：有理数体 ℚ を素数 $p$ に関する「p-進距離」で完備化したもの
- **演算**：p-進数の足し算・掛け算（p-進展開を用いる）
- **体の条件**：
  - 加法は可換群
  - 0 以外の元は乗法について可換群
  - 分配法則が成り立つ
- **特徴**：
  - **非アルキメデス的体**（通常の絶対値とは異なる距離）
  - 数論（p-進解析）で重要

__8. その他の例__

- **ℝ(x)**：実係数の有理関数体
- **ℂ(x)**：複素数係数の有理関数体
- **有限体の拡大体** 𝔽_{p^n}：素数べき個の元を持つ体（ガロア体）

### Pythonでイメージ

__1. 有限体 𝔽ₚと複素数体 ℂ__

有限体 𝔽ₚと複素数体 ℂを例として、加法・乗法の演算表の可視化を行います。
体としての性質（零因子の有無、逆元の存在）を可視化してイメージできるようにしてみます。

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_finite_field(p=5):
    """
    有限体 𝔽ₚ の加法・乗法表と逆元の分布を可視化
    """
    if not (isinstance(p, int) and p > 1):
        raise ValueError("p は 2 以上の整数で指定してください")

    values = list(range(p))
    size = p

    # 加法表 (mod p)
    add_table = np.zeros((size, size), dtype=int)
    for i, a in enumerate(values):
        for j, b in enumerate(values):
            add_table[i, j] = (a + b) % p

    # 乗法表 (mod p)
    mul_table = np.zeros((size, size), dtype=int)
    for i, a in enumerate(values):
        for j, b in enumerate(values):
            mul_table[i, j] = (a * b) % p

    # 逆元のチェック（0 以外）
    inverses = {}
    for a in range(1, p):
        for b in range(1, p):
            if (a * b) % p == 1:
                inverses[a] = b
                break

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 加法表
    im0 = axes[0].imshow(add_table, cmap='viridis', interpolation='nearest')
    axes[0].set_title(f'加法表 𝔽_{p} (mod {p})', fontsize=14)
    axes[0].set_xticks(range(size))
    axes[0].set_yticks(range(size))
    axes[0].set_xticklabels(values)
    axes[0].set_yticklabels(values)
    axes[0].set_xlabel('b')
    axes[0].set_ylabel('a')
    plt.colorbar(im0, ax=axes[0])

    # 乗法表（逆元を強調）
    im1 = axes[1].imshow(mul_table, cmap='viridis', interpolation='nearest')
    axes[1].set_title(f'乗法表 𝔽_{p} (逆元を強調)', fontsize=14)
    axes[1].set_xticks(range(size))
    axes[1].set_yticks(range(size))
    axes[1].set_xticklabels(values)
    axes[1].set_yticklabels(values)
    axes[1].set_xlabel('b')
    axes[1].set_ylabel('a')

    # 逆元の位置をマーキング（a の逆元が b なら (a,b) に印）
    for a, inv in inverses.items():
        axes[1].plot(inv, a, 'ro', markersize=8, markeredgecolor='white')

    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.show()

    # 逆元の情報をテキストで表示
    print(f"𝔽_{p} の乗法の逆元:")
    for a in range(1, p):
        print(f"  {a} の逆元: {inverses[a]} ({a} × {inverses[a]} ≡ 1 mod {p})")

# 実行例
plot_finite_field(p=5)  # 素数の例（体）
plot_finite_field(p=7)  # 別の素数の例
```

__実行結果__

コードで表示する左側は加法表で右側は乗法表です。

- 加法表は対称で、0 の列・行が単位元。
- 乗法表で、0 以外の行には必ず 1 が現れる（逆元の存在）。

赤丸は「a の逆元が b」であることを示し、0 以外のすべての元に逆元があることが視覚的にわかると思います。

<img src="image/1_set/1783234409052.png" alt="代替テキスト" width="500" style="display: block; margin: 0 auto;">

```
𝔽_5 の乗法の逆元:
  1 の逆元: 1 (1 × 1 ≡ 1 mod 5)
  2 の逆元: 3 (2 × 3 ≡ 1 mod 5)
  3 の逆元: 2 (3 × 2 ≡ 1 mod 5)
  4 の逆元: 4 (4 × 4 ≡ 1 mod 5)
```

<img src="image/1_set/1783234420812.png" alt="代替テキスト" width="500" style="display: block; margin: 0 auto;">

```
𝔽_7 の乗法の逆元:
  1 の逆元: 1 (1 × 1 ≡ 1 mod 7)
  2 の逆元: 4 (2 × 4 ≡ 1 mod 7)
  3 の逆元: 5 (3 × 5 ≡ 1 mod 7)
  4 の逆元: 2 (4 × 2 ≡ 1 mod 7)
  5 の逆元: 3 (5 × 3 ≡ 1 mod 7)
  6 の逆元: 6 (6 × 6 ≡ 1 mod 7)
```

__複素数体 ℂ の可視化__

次は複素隊 ℂ が体であるということを確認してみようと思います。

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_complex_field():
    """
    複素数体 ℂ の加法・乗法を幾何的に可視化
    """
    # 例として z = 1+2i, w = 2+1i を選ぶ
    z = complex(1, 2)
    w = complex(2, 1)

    # 加法: z + w
    add_result = z + w

    # 乗法: z * w
    mul_result = z * w

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 加法の可視化（ベクトルの和）
    axes[0].quiver(0, 0, z.real, z.imag, angles='xy', scale_units='xy', scale=1, color='blue', label=f'z = {z}')
    axes[0].quiver(z.real, z.imag, w.real, w.imag, angles='xy', scale_units='xy', scale=1, color='green', label=f'w = {w}')
    axes[0].quiver(0, 0, add_result.real, add_result.imag, angles='xy', scale_units='xy', scale=1, color='red', label=f'z+w = {add_result}')
    axes[0].set_xlim(-1, 5)
    axes[0].set_ylim(-1, 5)
    axes[0].set_aspect('equal')
    axes[0].grid(True)
    axes[0].set_title('複素数の加法 (ベクトルの和)', fontsize=14)
    axes[0].legend()

    # 乗法の可視化（極形式：絶対値と偏角）
    r_z, theta_z = np.abs(z), np.angle(z)
    r_w, theta_w = np.abs(w), np.angle(w)
    r_mul, theta_mul = np.abs(mul_result), np.angle(mul_result)

    # 極座標プロット
    angles = [theta_z, theta_w, theta_mul]
    radii = [r_z, r_w, r_mul]
    labels = [f'z (r={r_z:.2f}, θ={theta_z:.2f})',
              f'w (r={r_w:.2f}, θ={theta_w:.2f})',
              f'z×w (r={r_mul:.2f}, θ={theta_mul:.2f})']
    colors = ['blue', 'green', 'red']

    for i, (theta, r, label, color) in enumerate(zip(angles, radii, labels, colors)):
        axes[1].plot([0, r*np.cos(theta)], [0, r*np.sin(theta)], color=color, linewidth=2, label=label)

    axes[1].set_xlim(-1, 6)
    axes[1].set_ylim(-1, 6)
    axes[1].set_aspect('equal')
    axes[1].grid(True)
    axes[1].set_title('複素数の乗法 (絶対値と偏角)', fontsize=14)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # テキストでの説明
    print("複素数体 ℂ の性質:")
    print(f"  加法: {z} + {w} = {add_result} (ベクトルの和)")
    print(f"  乗法: {z} × {w} = {mul_result}")
    print(f"    絶対値: |z|×|w| = {r_z:.2f}×{r_w:.2f} = {r_mul:.2f}")
    print(f"    偏角: arg(z)+arg(w) = {theta_z:.2f} + {theta_w:.2f} = {theta_mul:.2f} rad")
    print("  0 以外の複素数は逆元を持つ（例: 1/z など）")

# 実行例
plot_complex_field()
```

__実行結果__

先程のと同様で左が加法、右が乗法の結果を可視化したものです。

- 加法：複素数を平面ベクトルとして足し算（平行四辺形の法則）。
- 乗法： 絶対値は掛け算、偏角は足し算になる（極形式）。

これにより、ℂ が「足し算・掛け算・引き算・割り算（0 以外）が自由にできる体」であることが幾何的にイメージできます。


<img src="image/1_set/1783235551360.png" alt="代替テキスト" width="500" style="display: block; margin: 0 auto;">

```
複素数体 ℂ の性質:
  加法: (1+2j) + (2+1j) = (3+3j) (ベクトルの和)
  乗法: (1+2j) × (2+1j) = 5j
    絶対値: |z|×|w| = 2.24×2.24 = 5.00
    偏角: arg(z)+arg(w) = 1.11 + 0.46 = 1.57 rad
  0 以外の複素数は逆元を持つ（例: 1/z など）
```

### まとめ

- 体は「足し算・掛け算・引き算・割り算（0 以外）が自由にできる集合」です。
- 代表例として、
  - ℚ, ℝ, ℂ（標準的な無限体）
  - 𝔽ₚ（有限体）
  - ℚ(x), ℝ(x), ℂ(x）（有理関数体）
  - ℚ(√2) などの代数体
  - ℚₚ（p-進数体）
  などがあります。
- これらはすべて、集合論の枠組みの中で「集合と2つの演算の組」として厳密に定義できます。

