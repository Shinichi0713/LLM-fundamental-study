# 集合

## 集合はなぜ必要か

集合は、数学の「ものの集まり」をきちんと扱うための土台として必要とされています。  
主な理由は次の3つです。

1. **数学の対象をはっきりさせるため**  
   数学では「数」「図形」「関数」など、いろいろな対象を扱いますが、それらを「集合」としてまとめておくと、  
   - どの範囲の対象を考えているのか  
   - ある対象が含まれるかどうか  
   を厳密に決められます。  
   たとえば「自然数の集合」「実数の集合」と言えば、扱う数が何かが明確になります。

2. **数学の議論を厳密にするため**  
   19世紀ごろまで、数学は直感的な説明が多く、あいまいさもありました。  
   そこで「集合」と「集合の間の関係（包含・共通部分・和集合など）」を基礎に置くことで、  
   - 関数  
   - 無限  
   - 連続性  
   などを厳密に定義できるようになりました。  
   これが「集合論」という分野で、現代数学のほとんどは集合論の言葉で書かれています。

3. **抽象的な構造を扱うため**  
   集合を使うと、  
   - 群・環・体（代数学の構造）  
   - 位相空間（連続性や近さを扱う構造）  
   - 確率空間（確率を厳密に扱うための枠組み）  
   など、さまざまな数学的構造を **「集合＋追加のルール」** として統一的に扱えます。  
   これにより、異なる分野の結果を横断的に使ったり、一般化したりしやすくなります。

まとめると、集合は  
- 数学の対象を明確にし  
- 議論を厳密にし  
- 抽象的な構造を統一的に扱う  
ために必要とされています。  
その意味で、集合は「現代数学の共通言語」のような役割を果たしています。

## 集合の用語

集合論における基本的な用語の定義を、順に説明します。

### 1. 元（要素）
- **定義**  
  ある集合に属している個々の「もの」のことを、その集合の**元（げん）**または**要素（ようそ）** といいます。
- **記号**  
  - 「a は集合 A の元である」ことを  
    $$
    a \in A
    $$
    と書きます（「a は A に属する」と読みます）。
  - 「a は集合 A の元ではない」ことを  
    $$
    a \notin A
    $$
    と書きます。

- **例**  
  - 集合 $ A = \{1, 2, 3\} $ に対して、  
    $ 1 \in A $, $ 2 \in A $, $ 3 \in A $ ですが、  
    $ 4 \notin A $ です。


### 2. 部分集合
- **定義**  
  集合 A の**すべての元**が、別の集合 B にも属しているとき、  
  「A は B の**部分集合**である」といいます。
- **記号**  
  - 「A は B の部分集合である」ことを  
    $$
    A \subset B
    $$
    と書きます。
  - 定義を式で書くと：
    $$
    A \subset B \quad \text{とは} \quad (\forall x)(x \in A \Rightarrow x \in B)
    $$
    「任意の x について、x が A の元ならば x は B の元でもある」という意味です。

- **例**  
  - $ A = \{1, 2\}, \quad B = \{1, 2, 3\} $ のとき、  
    A の元 1, 2 はどちらも B に属しているので、  
    $ A \subset B $ です。
  - どんな集合 A に対しても、  
    $ A \subset A $（自分自身は自分の部分集合）とみなします。

### 3. 空集合
- **定義**  
  **一つも元を持たない集合**のことを**空集合（くうしゅうごう）** といいます。
- **記号**  
  - 空集合は  
    $$
    \emptyset \quad \text{または} \quad \{\}
    $$
    と書きます。

- **性質**  
  - 空集合は、**任意の集合の部分集合**とみなします。  
    つまり、どんな集合 A に対しても  
    $$
    \emptyset \subset A
    $$
    が成り立ちます。
  - 理由（直感的な説明）：  
    「$\emptyset$ のすべての元が A に属している」という条件を考えますが、  
    $\emptyset$ には元が一つもないので、この条件は自動的に満たされます（偽の前提は何でも導く、という論理的な扱いになります）。

- **例**  
  - $ A = \{1, 2, 3\} $ のとき、  
    $ \emptyset \subset A $ です。
  - 空集合自身も集合なので、  
    $ \emptyset \subset \emptyset $ も成り立ちます。

## 集合の代表例

集合の代表的な例を、いくつかのタイプに分けて紹介します。


### 1. 数の集合（数学でよく使うもの）

- **自然数の集合**  
  $$
  \mathbb{N} = \{1, 2, 3, 4, \dots\}
  $$
  （0 を含める流儀もありますが、多くの高校数学では 1 から始めます）

- **整数の集合**  
  $$
  \mathbb{Z} = \{\dots, -3, -2, -1, 0, 1, 2, 3, \dots\}
  $$

- **有理数の集合**  
  $$
  \mathbb{Q} = \left\{ \frac{p}{q} \mid p, q \in \mathbb{Z},\ q \neq 0 \right\}
  $$
  （分数で書ける数の集合）

- **実数の集合**  
  $$
  \mathbb{R}
  $$
  （数直線上のすべての点に対応する数：有理数＋無理数）

- **複素数の集合**  
  $$
  \mathbb{C} = \{ a + bi \mid a, b \in \mathbb{R} \}
  $$
  （実数に虚数単位 $i$ を加えた数の集合）

### 2. 有限集合の例

- **1桁の自然数の集合**  
  $$
  A = \{1, 2, 3, 4, 5, 6, 7, 8, 9\}
  $$

- **アルファベットの集合**  
  $$
  B = \{a, b, c, \dots, z\}
  $$

- **あるクラスの生徒の集合**  
  $$
  C = \{\text{山田}, \text{佐藤}, \text{鈴木}, \dots\}
  $$

### 3. 図形や点の集合（幾何学的な例）

- **平面上の点の集合**  
  $$
  D = \{(x, y) \mid x, y \in \mathbb{R} \}
  $$
  （座標平面全体）

- **単位円（中心が原点、半径1の円）**  
  $$
  E = \{(x, y) \mid x^2 + y^2 = 1 \}
  $$

- **x 軸より上にある点の集合**  
  $$
  F = \{(x, y) \mid y > 0 \}
  $$


### 4. 条件で決まる集合（内包的記法の例）

- **偶数の集合**  
  $$
  G = \{ x \in \mathbb{Z} \mid x \text{ は偶数} \}
  $$

- **3 で割って 1 余る自然数の集合**  
  $$
  H = \{ n \in \mathbb{N} \mid n \equiv 1 \pmod{3} \}
  $$

- **100 以下の素数の集合**  
  $$
  I = \{ p \in \mathbb{N} \mid p \text{ は素数}, p \leq 100 \}
  $$

### 5. 特殊な集合

- **空集合**  
  $$
  \emptyset = \{\}
  $$
  （元を一つも持たない集合）

- **一点集合（シングルトン）**  
  $$
  J = \{0\}, \quad K = \{\text{東京}\}
  $$
  （元が1つだけの集合）

- **集合の集合（集合族）**  
  $$
  L = \{ \{1,2\}, \{3,4\}, \{5\} \}
  $$
  （集合を元として持つ集合）

### 6. 日常的なものの集合

- **日本の都道府県の集合**  
  $$
  M = \{\text{北海道}, \text{青森県}, \dots, \text{沖縄県}\}
  $$

- **ある駅から乗れる電車の路線の集合**  
  $$
  N = \{\text{山手線}, \text{中央線}, \text{京浜東北線}, \dots\}
  $$

- **ある本に登場する人物の集合**  
  $$
  O = \{\text{ハリー・ポッター}, \text{ロン}, \text{ハーマイオニー}, \dots\}
  $$

## 和集合・共通集合・差集合

集合の基本的な演算である「和集合」「共通部分（共通集合）」「差集合」について、順に説明します。

### 1. 和集合（Union）

__定義__
2つの集合 $A$, $B$ に対して、  
「**A か B の少なくとも一方に属する元全体の集合**」を、  
$A$ と $B$ の**和集合**といいます。

- 記号：$A \cup B$
- 定義式：
  $$
  A \cup B = \{ x \mid x \in A \text{ または } x \in B \}
  $$

__例__
- $A = \{1, 2, 3\},\quad B = \{3, 4, 5\}$ のとき、
  $$
  A \cup B = \{1, 2, 3, 4, 5\}
  $$
  （重複している 3 は1回だけ書きます）

- 図形的なイメージ（ベン図）  
  - 2つの円（A と B）を合わせた領域全体が $A \cup B$ です。

### 2. 共通部分（Intersection、共通集合）

__定義__
2つの集合 $A$, $B$ に対して、  
「**A にも B にも属する元全体の集合**」を、  
$A$ と $B$ の**共通部分**（または**共通集合**）といいます。

- 記号：$A \cap B$
- 定義式：
  $$
  A \cap B = \{ x \mid x \in A \text{ かつ } x \in B \}
  $$

__例__
- $A = \{1, 2, 3\},\quad B = \{3, 4, 5\}$ のとき、
  $$
  A \cap B = \{3\}
  $$

- $A = \{1, 2\},\quad B = \{3, 4\}$ のとき、
  $$
  A \cap B = \emptyset
  $$
  （共通する元がないので空集合）

- 図形的なイメージ（ベン図）  
  - 2つの円（A と B）が重なっている部分が $A \cap B$ です。

### 3. 差集合（Set Difference）

__定義__
2つの集合 $A$, $B$ に対して、  
「**A には属するが、B には属さない元全体の集合**」を、  
$A$ と $B$ の**差集合**といいます。

- 記号：$A \setminus B$（または $A - B$）
- 定義式：
  $$
  A \setminus B = \{ x \mid x \in A \text{ かつ } x \notin B \}
  $$

__例__
- $A = \{1, 2, 3\},\quad B = \{3, 4, 5\}$ のとき、
  $$
  A \setminus B = \{1, 2\}
  $$
  （A のうち、B にも属する 3 を除いたもの）

- $A = \{1, 2\},\quad B = \{3, 4\}$ のとき、
  $$
  A \setminus B = \{1, 2\} = A
  $$
  （A の元はどれも B に属さないので、そのまま A になる）


### 4. 補集合との関係（参考）

全体集合 $U$（考えている範囲のすべての元の集合）を決めたとき、  
ある集合 $A$ の**補集合** $A^c$ は、
$$
A^c = U \setminus A
$$
と定義されます。  
つまり、差集合の特別な場合が補集合です。

## 演習

集合の演算に関する証明問題をいくつか出題します。  
必要に応じて、全体集合 $U$ や補集合 $A^c$ も使って構いません。

### 問題

__問題1（基本）__

集合 $A, B$ について、次を示せ。
$$
A \subset B \quad \Rightarrow \quad A \cup B = B
$$



__問題2（分配法則）__

集合 $A, B, C$ について、次を示せ。
$$
A \cap (B \cup C) = (A \cap B) \cup (A \cap C)
$$



__問題3（ド・モルガンの法則）__

全体集合 $U$ とその部分集合 $A, B$ について、次を示せ。
$$
(A \cup B)^c = A^c \cap B^c
$$


__問題4（差集合の性質）__

集合 $A, B$ について、次を示せ。
$$
A \setminus B = A \cap B^c
$$
ただし、全体集合 $U$ を考え、$B^c$ は $B$ の補集合とする。

__問題5（少し応用）__

集合 $A, B, C$ について、次を示せ。
$$
A \setminus (B \cup C) = (A \setminus B) \cap (A \setminus C)
$$

__問題6（包含関係の証明）__

集合 $A, B, C$ について、次を示せ。
$$
A \subset B \quad \Rightarrow \quad A \setminus C \subset B \setminus C
$$

__問題7（対称差の性質）__

集合 $A, B$ の**対称差**を
$$
A \triangle B = (A \setminus B) \cup (B \setminus A)
$$
と定義する。このとき、次を示せ。
$$
A \triangle B = (A \cup B) \setminus (A \cap B)
$$

### 解答

先ほどの各問題について、順に証明します。

__問題1：$A \subset B \Rightarrow A \cup B = B$__

**証明**  
$A \subset B$ を仮定する。  
任意の元 $x$ について、

- $x \in A \cup B$  
  $\Leftrightarrow x \in A$ または $x \in B$  
  $\Rightarrow x \in B$（∵ $A \subset B$ より $x \in A \Rightarrow x \in B$）  
  よって $A \cup B \subset B$。

- 一方、$x \in B \Rightarrow x \in A$ または $x \in B$  
  $\Rightarrow x \in A \cup B$  
  よって $B \subset A \cup B$。

以上より $A \cup B = B$。□

__問題2：分配法則 $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$__

**証明**  
任意の元 $x$ について、

$$
\begin{aligned}
x \in A \cap (B \cup C)
&\Leftrightarrow x \in A \text{ かつ } (x \in B \text{ または } x \in C) \\
&\Leftrightarrow (x \in A \text{ かつ } x \in B) \text{ または } (x \in A \text{ かつ } x \in C) \\
&\Leftrightarrow x \in A \cap B \text{ または } x \in A \cap C \\
&\Leftrightarrow x \in (A \cap B) \cup (A \cap C)
\end{aligned}
$$

よって両辺は等しい。□

__問題3：ド・モルガンの法則 $(A \cup B)^c = A^c \cap B^c$__

**証明**  
全体集合を $U$ とし、任意の元 $x \in U$ について、

$$
\begin{aligned}
x \in (A \cup B)^c
&\Leftrightarrow x \notin A \cup B \\
&\Leftrightarrow \text{「}x \in A \text{ または } x \in B\text{」が偽} \\
&\Leftrightarrow x \notin A \text{ かつ } x \notin B \\
&\Leftrightarrow x \in A^c \text{ かつ } x \in B^c \\
&\Leftrightarrow x \in A^c \cap B^c
\end{aligned}
$$

よって $(A \cup B)^c = A^c \cap B^c$。□

__問題4：差集合 $A \setminus B = A \cap B^c$__

**証明**  
全体集合 $U$ を考え、任意の元 $x \in U$ について、

$$
\begin{aligned}
x \in A \setminus B
&\Leftrightarrow x \in A \text{ かつ } x \notin B \\
&\Leftrightarrow x \in A \text{ かつ } x \in B^c \\
&\Leftrightarrow x \in A \cap B^c
\end{aligned}
$$

よって $A \setminus B = A \cap B^c$。□

__問題5：$A \setminus (B \cup C) = (A \setminus B) \cap (A \setminus C)$__

**証明**  
任意の元 $x$ について、

$$
\begin{aligned}
x \in A \setminus (B \cup C)
&\Leftrightarrow x \in A \text{ かつ } x \notin B \cup C \\
&\Leftrightarrow x \in A \text{ かつ } (x \notin B \text{ かつ } x \notin C) \\
&\Leftrightarrow (x \in A \text{ かつ } x \notin B) \text{ かつ } (x \in A \text{ かつ } x \notin C) \\
&\Leftrightarrow x \in A \setminus B \text{ かつ } x \in A \setminus C \\
&\Leftrightarrow x \in (A \setminus B) \cap (A \setminus C)
\end{aligned}
$$

よって両辺は等しい。□

__問題6：$A \subset B \Rightarrow A \setminus C \subset B \setminus C$__

**証明**  
$A \subset B$ を仮定する。  
任意の元 $x$ について、

$$
\begin{aligned}
x \in A \setminus C
&\Rightarrow x \in A \text{ かつ } x \notin C \\
&\Rightarrow x \in B \text{ かつ } x \notin C \quad (\because A \subset B) \\
&\Rightarrow x \in B \setminus C
\end{aligned}
$$

よって $A \setminus C \subset B \setminus C$。□

__問題7：対称差 $A \triangle B = (A \cup B) \setminus (A \cap B)$__

**証明**  
定義より
$$
A \triangle B = (A \setminus B) \cup (B \setminus A)
$$
である。  
任意の元 $x$ について、

$$
\begin{aligned}
x \in A \triangle B
&\Leftrightarrow x \in A \setminus B \text{ または } x \in B \setminus A \\
&\Leftrightarrow (x \in A \text{ かつ } x \notin B) \text{ または } (x \in B \text{ かつ } x \notin A) \\
&\Leftrightarrow x \in A \cup B \text{ かつ } x \notin A \cap B \\
&\Leftrightarrow x \in (A \cup B) \setminus (A \cap B)
\end{aligned}
$$

よって $A \triangle B = (A \cup B) \setminus (A \cap B)$。□


<div style="page-break-before:always"></div>

