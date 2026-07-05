---
title: "わかりやすい集合論"
author: "3 sons lover"
language: "ja"
---

# 前書き

著者は企業に所属して機械設計から、ソフトエンジニアとなり、AI技術を開発、もしくはAIを使ったシステムを開発しています。

それほど勉強に熱心だった訳でない大学・大学院生活を経て企業に就職した後、AI技術開発を行うことになりました。
すると、論文の重要性が身に染みてきました。

AI技術では、実験結果を確認するサイクルは比較的短く済む一方で、新しい理論や手法は次々と提案されています。そのため、すべてを一から試していると、理論の進歩のスピードに追いつけません。
こういう技術をつくるとどうなるだろう？
こういう手法とりいれるとどうなるだろう？
を一から実験しているとスピードにのっていけません。
そんな時に同じようなことをやっている研究ないか論文で探してイメージをするようにします。
これはAI技術開発をやっている方でなくても、研究や先行研究をされている方であれば皆さん経験することだと思います。

論文を理解するために必要なことは何か？
- 知識のとっかかりがある(論文の研究分野に関する背景を持っていること)
- 基礎理論を理解すること
前者は研究に取り組んでいるうちに、関連する方と交流するうちに自然と身に着けていきます。
ネックになるのは後者ではないでしょうか。

基礎理論は基本的に数学や物理です。
そして、基礎理論を記載する際に、**理論の前提条件を定義**する必要があります。
この理論の前提条件に使われているのが集合論です。

集合論は数、集合、関数、群などの数や数の構造化で必ず使う必要があります。
なぜなら、集合論が出来た理由がこの必要性だったからです。集合論が整備される以前は、数学の各分野が独自の言葉で理論を記述しており、全体を統一して厳密に定義することが難しかったという背景があります。そのため、19世紀末〜20世紀にかけて、数学の基礎を集合論の上に構築する流れが生まれました。

集合論は、数や関数、群・環・体といった代数的構造、位相空間、確率空間など、現代数学のあらゆる対象を「集合と写像」という単一の枠組みで定義するための言語です。そのため、新しい理論を厳密に記述する際には、その前提条件や対象を集合論の言葉で明確に定義する必要があります。今後、数学や物理学でどのような新しい学問体系が現れたとしても、それを厳密に定義するには、集合論の考え方が不可欠になるでしょう。

集合論は大学の数学課程で扱われるものの、多くの学生にとってイメージがつかみにくい分野です。理由の一つとして、集合論そのものを専門とする教員は少なく、他の分野の講義の一部として、定義や記号を簡潔に紹介するにとどまることが多いからです。その結果、集合論が「数学全体の共通言語」としてどのように機能するのか、実感を持って学ぶ機会が少ないという課題を抱えています。

のような必要性と課題を踏まえ、AI技術開発や研究開発に携わる方々が、集合論を「論文を読み解くための実用的な道具」として身につけられるように、本書を作成しました。

__本書の狙い__

これから集合論を学ぶ社会人・学生の方々に押さえて頂きたい以下の内容を、扱っていきます。
これらを押さえると、その後の数学・情報系の学習で「集合論は数学の共通言語」という実感を基に基礎理論のより深い理解につながると考えています。

## 1. 集合の基本と記法
- 集合・要素（∈, ∉）、部分集合（⊆, ⊂）
- 空集合 ∅、よく使う集合 ℕ, ℤ, ℚ, ℝ, ℂ
- 内包的記法・外延的記法

## 2. 集合演算
- 和集合 ∪、共通部分 ∩、差集合 \、補集合 A^c
- べき集合 P(X)、直積 A × B
- 分配法則・ド・モルガンの法則など基本性質

## 3. 写像（関数）
- 写像の定義（定義域・値域・像・逆像）
- 単射・全射・全単射
- 合成写像、逆写像（全単射のとき）

## 4. 関係と同値関係
- 二項関係（A × A の部分集合）
- 反射・対称・推移 → 同値関係
- 同値類、商集合 A / ∼

## 5. 濃度（集合の「大きさ」）
- 有限集合・無限集合
- 可算集合・非可算集合
- ベルンシュタインの定理
- カントールの対角線論法（ℝ の非可算性）

## 6. 順序と整列
- 半順序・全順序
- 最大元・最小元・極大元・極小元
- 上界・下界・上限・下限
- 整列集合（概要）


**ポイント**：  
- 定義を正確に読み、具体例で確かめる  
- 集合・写像・関係・順序という「構造」を、集合論の言葉で扱えるようにする  
- 濃度と選択公理で「無限」と「公理」の感覚をつかむ  



__本書の対象者__

- 大学で理工系学科で学ぶ学生
- 就職の後に理論を応用・研究するため学び直しをされる社会人

__本書が意識している点__

ずばり分かりやすい言葉で説明することです。
著者自身、集合の教科書を地道に読んでいるときに、説明の難しさに驚いた記憶が沢山あります。
数学を厳密に定義するから、こんな難しい言い方をしているのか、と言い聞かせてきましたが、そんなはずはありません。
分かりやすく説明して理解できる形にすることが意識した点です。

__本書の構成__

以下のように構成しました。

__1章 集合__

一番入口となる集合の定義を行います。

__2章 べき集合、直積集合、写像__

数学では、「集合そのもの」を要素とする集合を扱うことがよくあります。
集合の集合を定義するためのべき集合について説明します。

__3章 添数づけられた族と選択公理__

数学では、有限個の集合 A₁, A₂, …, Aₙ だけでなく、無限個（可算無限や非可算無限）の集合を扱うことがよくあります。
「添数づけられた族」と「選択公理」は、無限個の集合から要素を選びたいときに本質的に必要になる概念です。

__4章 同値関係__

数学では、「本質的に同じとみなせるもの」を一つの塊として扱いたいことがよくあります。
「同じもの」をまとめて、構造を簡約するために必要な考えである同値関係の概念を説明します。

__5章 集合の対等__

集合の「大きさ」を厳密に比較するために、全単射の存在で「同じ大きさ」を定義する概念が必要になります。これが集合の対等です。有限集合では「要素数が同じ」と同じ意味ですが、無限集合では「可算・非可算」といった重要な区別を与えます。

__6章 順序集合__

現実世界でも数学でも、「順序」や「大小関係」は基本的な概念です。
これらを一つの抽象的な枠組みで扱いたい → それが順序集合です。

__7章 整列集合__

整列集合は、「任意の空でない部分集合に最小元がある」という強い性質を持つ順序集合です。自然数のように「どこから始めても最初の要素が取れる」順序を抽象化したもので、この上で帰納法を無限に拡張した超限帰納法が成り立ちます。

__8章 Zornの補題と整列可能定理__

Zornの補題は、「鎖（全順序部分集合）が上界を持つ順序集合には極大元が存在する」という存在定理で、ベクトル空間の基底の存在や極大イデアルの存在など、多くの重要な定理の証明に使われます。整列可能定理は「任意の集合は適切な順序で整列集合にできる」という主張で、選択公理と同値です。これらにより、無限構造から極大元や整列順序を取り出し、超限帰納法・超限再帰を一般の集合に適用できるようになります。

__9章 順序数__

順序数は、整列順序の「型」を抽象的な数として表す概念です。自然数 0,1,2,… に加えて ω, ω+1, ω+ω などの超限順序数を導入することで、無限に続く整列順序を統一的に扱えます。これにより、超限帰納法・超限再帰が順序数全体を舞台として厳密に定義され、集合論の多くの議論（基数の定義など）の土台となります。

__10章 基数と濃度__

集合の「大きさ」を厳密に比較するために、濃度（cardinality）という概念を導入します。有限集合では要素数に対応し、無限集合では「可算・非可算」といった区別を与えます。これにより、確率論や測度論で必要となる「事象の大きさ」や、計算理論での「言語の複雑さ」などを厳密に議論できるようになります。

<div style="page-break-before:always"></div>


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

>__群（group）__  
>集合論でいう**群** は、**「一つの演算（多くは掛け算や足し算）が定義され、その演算について逆元と単位元を持つ集合」** のことです。
>もう少し正確に言うと、集合 <img src="tmp/b15d92e08fe483b65746431228085e71.png" class="math-inline" /> とその上の演算 <img src="tmp/356513ded34229fab8ea72b65824828d.png" class="math-inline" />（掛け算の記号で書くことが多い）が次の条件を満たすとき、<img src="tmp/f16de060cce53c647d47daeb7667bc7c.png" class="math-inline" /> は群です。
>1. **結合則**：<img src="tmp/9c2d4b1ce866bf7919c3822b73978c06.png" class="math-inline" />
>2. **単位元の存在**：ある元 <img src="tmp/8ac7967375b961b8ef9657c37855da8a.png" class="math-inline" /> が存在して、すべての <img src="tmp/3fff980604c814e2b9b235523d0573fe.png" class="math-inline" /> について <img src="tmp/c11c71fa6c5153921cd608c6cf238c80.png" class="math-inline" />
>3. **逆元の存在**：各 <img src="tmp/3fff980604c814e2b9b235523d0573fe.png" class="math-inline" /> に対して、ある <img src="tmp/c2f1aee58f84318be9878a799bdb403c.png" class="math-inline" /> が存在して <img src="tmp/63893eb1d5dcaddebf23c17d2df3232c.png" class="math-inline" />
>さらに、演算が**可換**（<img src="tmp/b2b1e7702098506cbf51c4ed2c0bdce4.png" class="math-inline" />）なら、**可換群（アーベル群）** と呼びます。
>__具体例__  
>- 整数全体 ℤ と足し算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />：可換群（単位元 0、逆元 <img src="tmp/688916247f0baa4a6682864234b8d5b1.png" class="math-inline" />）
>- 0 でない実数全体 ℝ∖{0} と掛け算 <img src="tmp/0169f361e6387b5186b557fdcc3af30b.png" class="math-inline" />：可換群（単位元 1、逆元 <img src="tmp/0112b5aef8e9acf74a820260ace9949f.png" class="math-inline" />）
>- 正三角形の回転対称の集合：群（回転の合成が演算）
>__集合論との関係__  
>- 群は**集合と演算の組**として定義されるので、集合論の枠組みの中で扱えます。
>- 群の要素は集合の元であり、群の構造（部分群・剰余群など）も集合論的に記述できます。
>要するに、**群＝「逆元と単位元を持つ演算が定義された集合」** です。

>__環（ring）__  
>**環**とは、**「足し算と掛け算の2つの演算が定義され、足し算については可換群、掛け算については結合的で、分配法則が成り立つ集合」** のことです。
>もう少し正確に言うと、集合 <img src="tmp/a0f7a58538051aad071147a20c796777.png" class="math-inline" /> とその上の2つの演算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />（加法）と <img src="tmp/356513ded34229fab8ea72b65824828d.png" class="math-inline" />（乗法）が次の条件を満たすとき、<img src="tmp/ac09066d037a4fc2518d68cf9691a2cd.png" class="math-inline" /> は環です。
>1. <img src="tmp/e23d22c3ed1394e2db2fb5143a2c30ba.png" class="math-inline" /> は可換群（足し算について逆元と単位元 0 を持つ）
>2. 乗法は結合的：<img src="tmp/9c2d4b1ce866bf7919c3822b73978c06.png" class="math-inline" />
>3. 分配法則が成り立つ：
>   - <img src="tmp/9df7c1a8791128bdb1bf257e3c5d703d.png" class="math-inline" />
>   - <img src="tmp/6c098c26c035650ca33efab9a86f4bb5.png" class="math-inline" />
>多くの場合、乗法の単位元 1 も持つ環（単位的環）を考えます。
>__具体例__  
>- 整数全体 ℤ（通常の足し算・掛け算）
>- 実数係数の多項式環 ℝ[x]
>- 行列環 <img src="tmp/afebcf945ec290c8fb8af6e4562217f8.png" class="math-inline" />
>これらはすべて環です。
__一言で言うと__  
>環＝「足し算と掛け算がきちんと定義された集合」  
>（足し算は可換群、掛け算は結合的で分配法則を満たす）です。


>__体（field）__  
>**体**とは、**「足し算・掛け算・引き算・割り算（0 以外で）が自由にできる集合」** のことです。
>もう少し正確に言うと、集合 <img src="tmp/2869f80330e3b768b3feec4cdf4c7583.png" class="math-inline" /> とその上の演算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />（加法）と <img src="tmp/356513ded34229fab8ea72b65824828d.png" class="math-inline" />（乗法）が次の条件を満たすとき、<img src="tmp/afee8c2f8af443e2dae18efdb623912c.png" class="math-inline" /> は体です。
>1. <img src="tmp/17f29554a4169fb8d50bc2f1692d3b25.png" class="math-inline" /> は可換群（足し算について逆元と単位元 0 を持つ）
>2. <img src="tmp/736c8a596d6fe07c4108caace61a7883.png" class="math-inline" /> も可換群（0 以外の元は掛け算について逆元と単位元 1 を持つ）
>3. 分配法則が成り立つ：<img src="tmp/9df7c1a8791128bdb1bf257e3c5d703d.png" class="math-inline" />
>つまり、
>- 足し算と掛け算の両方が「群」としてきちんと動く
>- 掛け算は 0 以外の元について可換群
>- 足し算と掛け算が分配法則で結びつく
>という構造です。
__具体例__  
>- 有理数全体 ℚ
>- 実数全体 ℝ
>- 複素数全体 ℂ
>- 有限体（例：mod 5 の世界 ℤ/5ℤ）
>これらはすべて体です。
>__一言で言うと__  
>体＝「足し算・掛け算・引き算・割り算（0 以外）が自由にできる集合」  
>（足し算と掛け算の両方が可換群で、分配法則を満たす）です。


## 集合の用語

集合論における基本的な用語の定義を、順に説明します。

### 1. 元（要素）
- **定義**  
  ある集合に属している個々の「もの」のことを、その集合の**元（げん）**または**要素（ようそ）** といいます。
- **記号**  
  - 「a は集合 A の元である」ことを  
    

<div class="math-display-container"><img src="tmp/96b46300df44c39eeb765cdee1a0e363.png" class="math-display" /></div>


    と書きます（「a は A に属する」と読みます）。
  - 「a は集合 A の元ではない」ことを  
    

<div class="math-display-container"><img src="tmp/ed7fb12bd329b3847ce248d42c177a29.png" class="math-display" /></div>


    と書きます。

- **例**  
  - 集合 <img src="tmp/a10641f0f0013c8447969faa7b6a21d0.png" class="math-inline" /> に対して、  
    <img src="tmp/991f3200efc494a996cad2bea6bf8d7f.png" class="math-inline" />, <img src="tmp/fbc5af78db33b9bc659f7fe1db68cf39.png" class="math-inline" />, <img src="tmp/d0ce2ed0b8029c86dbb4769dae08acac.png" class="math-inline" /> ですが、  
    <img src="tmp/0137318b78e96983be11ceccdd6127d2.png" class="math-inline" /> です。


### 2. 部分集合
- **定義**  
  集合 A の**すべての元**が、別の集合 B にも属しているとき、  
  「A は B の**部分集合**である」といいます。
- **記号**  
  - 「A は B の部分集合である」ことを  
    

<div class="math-display-container"><img src="tmp/8052d87fc278eafe38993e75401f837f.png" class="math-display" /></div>


    と書きます。
  - 定義を式で書くと：
    

<div class="math-display-container"><img src="tmp/ea490c940e9ba8d1fcde16689be2edbc.png" class="math-display" /></div>


    「任意の x について、x が A の元ならば x は B の元でもある」という意味です。

- **例**  
  - <img src="tmp/3ed4890a537591d465cd64c02186c8e4.png" class="math-inline" /> のとき、  
    A の元 1, 2 はどちらも B に属しているので、  
    <img src="tmp/d3cd613dbef8bc0f1be0fb2d340867e0.png" class="math-inline" /> です。
  - どんな集合 A に対しても、  
    <img src="tmp/ef1b6af8981071c83720cc230f850df6.png" class="math-inline" />（自分自身は自分の部分集合）とみなします。

### 3. 空集合
- **定義**  
  **一つも元を持たない集合**のことを**空集合（くうしゅうごう）** といいます。
- **記号**  
  - 空集合は  
    

<div class="math-display-container"><img src="tmp/6355a49fdc8ea3d84d6990d313d065b7.png" class="math-display" /></div>


    と書きます。

- **性質**  
  - 空集合は、**任意の集合の部分集合**とみなします。  
    つまり、どんな集合 A に対しても  
    

<div class="math-display-container"><img src="tmp/8aa897ca67b70a52e9c9fbc48b790491.png" class="math-display" /></div>


    が成り立ちます。
  - 理由（直感的な説明）：  
    「<img src="tmp/36682ee60666b219d2be3577c1bfda62.png" class="math-inline" /> のすべての元が A に属している」という条件を考えますが、  
    <img src="tmp/36682ee60666b219d2be3577c1bfda62.png" class="math-inline" /> には元が一つもないので、この条件は自動的に満たされます（偽の前提は何でも導く、という論理的な扱いになります）。

- **例**  
  - <img src="tmp/a10641f0f0013c8447969faa7b6a21d0.png" class="math-inline" /> のとき、  
    <img src="tmp/00572662d601d255c66ba6cbf0be6432.png" class="math-inline" /> です。
  - 空集合自身も集合なので、  
    <img src="tmp/b0b1fd40a53890cf09e7b949ec1b1a12.png" class="math-inline" /> も成り立ちます。

## 集合の代表例

集合の代表的な例を、いくつかのタイプに分けて紹介します。


### 1. 数の集合（数学でよく使うもの）

- **自然数の集合**  
  

<div class="math-display-container"><img src="tmp/2abe9b702e5a35deb23f1f6f74d4b71b.png" class="math-display" /></div>


  （0 を含める流儀もありますが、多くの高校数学では 1 から始めます）

- **整数の集合**  
  

<div class="math-display-container"><img src="tmp/578731240e7e216586a12ad0214c512f.png" class="math-display" /></div>



- **有理数の集合**  
  

<div class="math-display-container"><img src="tmp/0c5f30667a93a2fc224963f4cbadd98d.png" class="math-display" /></div>


  （分数で書ける数の集合）

- **実数の集合**  
  

<div class="math-display-container"><img src="tmp/a06653ca1d35fe996fdd5945d491318c.png" class="math-display" /></div>


  （数直線上のすべての点に対応する数：有理数＋無理数）

- **複素数の集合**  
  

<div class="math-display-container"><img src="tmp/6246ba29de9f35abfdc88f908eed4309.png" class="math-display" /></div>


  （実数に虚数単位 <img src="tmp/62e13e7c1114087e103f0e72f9db52d7.png" class="math-inline" /> を加えた数の集合）

### 2. 有限集合の例

- **1桁の自然数の集合**  
  

<div class="math-display-container"><img src="tmp/12670461eff773568a2562955a8546e3.png" class="math-display" /></div>



- **アルファベットの集合**  
  

<div class="math-display-container"><img src="tmp/65febe4adc65e8ff6be306d24f234e8f.png" class="math-display" /></div>



- **あるクラスの生徒の集合**  
  

<div class="math-display-container"><img src="tmp/8e8f31c9cae79fc40ae1a7524f5facfe.png" class="math-display" /></div>



### 3. 図形や点の集合（幾何学的な例）

- **平面上の点の集合**  
  

<div class="math-display-container"><img src="tmp/95794dae6347c4386d9f3edf67dbdf53.png" class="math-display" /></div>


  （座標平面全体）

- **単位円（中心が原点、半径1の円）**  
  

<div class="math-display-container"><img src="tmp/4ca0b234c911ae54db9a9b74ceae9164.png" class="math-display" /></div>



- **x 軸より上にある点の集合**  
  

<div class="math-display-container"><img src="tmp/19d9602ebe12226c992e652ee838ab58.png" class="math-display" /></div>




### 4. 条件で決まる集合（内包的記法の例）

- **偶数の集合**  
  

<div class="math-display-container"><img src="tmp/12b3190856f5feeefa2f3fbbe9888d03.png" class="math-display" /></div>



- **3 で割って 1 余る自然数の集合**  
  

<div class="math-display-container"><img src="tmp/4564e47f4510cc9ac15e2cacf009db2c.png" class="math-display" /></div>



- **100 以下の素数の集合**  
  

<div class="math-display-container"><img src="tmp/f28986cdafa97dc3ac75d19a27cc8efe.png" class="math-display" /></div>



### 5. 特殊な集合

- **空集合**  
  

<div class="math-display-container"><img src="tmp/acba7aeccf9f40702ad6cbd60f2d88d1.png" class="math-display" /></div>


  （元を一つも持たない集合）

- **一点集合（シングルトン）**  
  

<div class="math-display-container"><img src="tmp/fb9172dae20b3580e9317ddab29ed311.png" class="math-display" /></div>


  （元が1つだけの集合）

- **集合の集合（集合族）**  
  

<div class="math-display-container"><img src="tmp/768cafefec98c337d0b26906cedd7efb.png" class="math-display" /></div>


  （集合を元として持つ集合）

### 6. 日常的なものの集合

- **日本の都道府県の集合**  
  

<div class="math-display-container"><img src="tmp/465f720638cd09a3b45604198301b4b2.png" class="math-display" /></div>



- **ある駅から乗れる電車の路線の集合**  
  

<div class="math-display-container"><img src="tmp/72418d5c5bf89e463e9664724a5f8f29.png" class="math-display" /></div>



- **ある本に登場する人物の集合**  
  

<div class="math-display-container"><img src="tmp/4bd97653ef8394e2d6066fc95e4c126f.png" class="math-display" /></div>



## 和集合・共通集合・差集合

集合の基本的な演算である「和集合」「共通部分（共通集合）」「差集合」について、順に説明します。

### 1. 和集合（Union）

__定義__
2つの集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" />, <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> に対して、  
「**A か B の少なくとも一方に属する元全体の集合**」を、  
<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の**和集合**といいます。

- 記号：<img src="tmp/2a2d668fa15d8d000993fbeec5fc1f82.png" class="math-inline" />
- 定義式：
  

<div class="math-display-container"><img src="tmp/55330e2898d1ac397e862691b1e9b03e.png" class="math-display" /></div>



__例__
- <img src="tmp/52d14abeb9299a74d17662f903468668.png" class="math-inline" /> のとき、
  

<div class="math-display-container"><img src="tmp/427101ef7b64e055c080582dc5b8b322.png" class="math-display" /></div>


  （重複している 3 は1回だけ書きます）

- 図形的なイメージ（ベン図）  
  - 2つの円（A と B）を合わせた領域全体が <img src="tmp/2a2d668fa15d8d000993fbeec5fc1f82.png" class="math-inline" /> です。

### 2. 共通部分（Intersection、共通集合）

__定義__
2つの集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" />, <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> に対して、  
「**A にも B にも属する元全体の集合**」を、  
<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の**共通部分**（または**共通集合**）といいます。

- 記号：<img src="tmp/93840aa29d13dd6fd5056013d05f866b.png" class="math-inline" />
- 定義式：
  

<div class="math-display-container"><img src="tmp/a3b74c84d04d5237523e36ce77b678fc.png" class="math-display" /></div>



__例__
- <img src="tmp/52d14abeb9299a74d17662f903468668.png" class="math-inline" /> のとき、
  

<div class="math-display-container"><img src="tmp/724a66771d74fef59e6bbc094721cf83.png" class="math-display" /></div>



- <img src="tmp/c3657888b822b9c120280a1158aaf907.png" class="math-inline" /> のとき、
  

<div class="math-display-container"><img src="tmp/1961879ac4a09b4afdd21e56b68672de.png" class="math-display" /></div>


  （共通する元がないので空集合）

- 図形的なイメージ（ベン図）  
  - 2つの円（A と B）が重なっている部分が <img src="tmp/93840aa29d13dd6fd5056013d05f866b.png" class="math-inline" /> です。

### 3. 差集合（Set Difference）

__定義__
2つの集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" />, <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> に対して、  
「**A には属するが、B には属さない元全体の集合**」を、  
<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の**差集合**といいます。

- 記号：<img src="tmp/3cf4232b855ceecbd5d6471c5ae6fdba.png" class="math-inline" />（または <img src="tmp/6e03f97cc1dede590ce1d4e99ec1ba9b.png" class="math-inline" />）
- 定義式：
  

<div class="math-display-container"><img src="tmp/5b7e5a90e9ff7446ef193ecdc09d5dff.png" class="math-display" /></div>



__例__
- <img src="tmp/52d14abeb9299a74d17662f903468668.png" class="math-inline" /> のとき、
  

<div class="math-display-container"><img src="tmp/1387dbb087ea24f2c051a236d73319fd.png" class="math-display" /></div>


  （A のうち、B にも属する 3 を除いたもの）

- <img src="tmp/c3657888b822b9c120280a1158aaf907.png" class="math-inline" /> のとき、
  

<div class="math-display-container"><img src="tmp/cf90687077f6a85385998acf142d7ece.png" class="math-display" /></div>


  （A の元はどれも B に属さないので、そのまま A になる）


### 4. 補集合との関係（参考）

全体集合 <img src="tmp/2321eadfdfcad762b559d2c1971bf109.png" class="math-inline" />（考えている範囲のすべての元の集合）を決めたとき、  
ある集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**補集合** <img src="tmp/785a6da13086feb5b6bf91dedbb16481.png" class="math-inline" /> は、


<div class="math-display-container"><img src="tmp/90f1b7054f51d0abb63fe49f49ae82fa.png" class="math-display" /></div>


と定義されます。  
つまり、差集合の特別な場合が補集合です。

## 群

**群（group）** とは、**「一つの演算が定義され、その演算について逆元と単位元を持つ集合」** のことです。

もう少し正確に言うと、集合 <img src="tmp/b15d92e08fe483b65746431228085e71.png" class="math-inline" /> とその上の演算 <img src="tmp/356513ded34229fab8ea72b65824828d.png" class="math-inline" />（掛け算の記号で書くことが多い）が次の条件を満たすとき、<img src="tmp/f16de060cce53c647d47daeb7667bc7c.png" class="math-inline" /> は**群**です。

1. **結合則**：<img src="tmp/9c2d4b1ce866bf7919c3822b73978c06.png" class="math-inline" />
2. **単位元の存在**：ある元 <img src="tmp/8ac7967375b961b8ef9657c37855da8a.png" class="math-inline" /> が存在して、すべての <img src="tmp/3fff980604c814e2b9b235523d0573fe.png" class="math-inline" /> について <img src="tmp/c11c71fa6c5153921cd608c6cf238c80.png" class="math-inline" />
3. **逆元の存在**：各 <img src="tmp/3fff980604c814e2b9b235523d0573fe.png" class="math-inline" /> に対して、ある <img src="tmp/c2f1aee58f84318be9878a799bdb403c.png" class="math-inline" /> が存在して <img src="tmp/63893eb1d5dcaddebf23c17d2df3232c.png" class="math-inline" />

さらに、演算が**可換**（<img src="tmp/b2b1e7702098506cbf51c4ed2c0bdce4.png" class="math-inline" />）なら、次節で説明する **可換群（アーベル群）** と呼びます。

### 具体例

- 整数全体 ℤ と足し算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />：可換群（単位元 0、逆元 <img src="tmp/688916247f0baa4a6682864234b8d5b1.png" class="math-inline" />）
- 0 でない実数全体 ℝ∖{0} と掛け算 <img src="tmp/0169f361e6387b5186b557fdcc3af30b.png" class="math-inline" />：可換群（単位元 1、逆元 <img src="tmp/0112b5aef8e9acf74a820260ace9949f.png" class="math-inline" />）
- 正三角形の回転対称の集合：群（回転の合成が演算）

### 集合論との関係

- 群は**集合と演算の組**として定義されるので、集合論の枠組みの中で扱えます。
- 群の要素は集合の元であり、群の構造（部分群・剰余群など）も集合論的に記述できます。

要するに、**群＝「逆元と単位元を持つ演算が定義された集合」** です。

### なぜ群が必要だった？

群の定義が必要だった理由を、**歴史的・理論的・応用的**な観点から簡潔にまとめます。

__1. 共通の「構造」を抽象化するため__

- 整数の足し算、0 でない実数の掛け算、図形の回転、ベクトルの和など、一見バラバラに見える対象が、実は**同じルール（結合則・単位元・逆元）** で動いていることに気づいた。
- そこで、「個々の具体例に依存せず、**共通の性質だけを抜き出して定義**しよう」という発想から群の定義が生まれた。

__2. 対称性・変換を統一的に扱うため__

- 図形の対称性（回転・鏡映・並進）や、物理法則の対称性（並進・回転・ゲージ変換）は、  
  いずれも「操作の合成」という演算を持ち、逆操作と恒等操作が存在する。
- 群の定義により、**対称性や変換を抽象的な“群”として扱える**ようになり、  
  幾何・物理・結晶学などで統一的に議論できるようになった。

__3. 数学の基礎を厳密化するため__

- 19世紀後半から20世紀にかけて、数学の厳密化が進み、  
  「集合と演算」という立場から数学を再構築する動きが強まった。
- 群の定義は、**集合論の枠組みの中で“演算の構造”を厳密に記述するための最小限の条件**として整理された。
- これにより、環・体・ベクトル空間など、より複雑な構造も群の上に乗せて定義できるようになった。

__4. 応用（暗号・符号・物理）の土台として__

- 群の理論が整備されたことで、
  - 暗号（楕円曲線暗号など）
  - 符号理論（線形符号）
  - 物理の対称性と保存則（ネーターの定理）
  など、実用的な分野でも「群」という共通言語で議論できるようになった。

__一言で言うと__

> 群の定義は、  
> 「バラバラに見える対象（数・変換・対称性）に共通する“演算の構造”を抽象化し、  
> 数学・物理・情報科学で統一的に扱うため」  
> に必要だった。

です。

## 可換群

**可換群（commutative group）** とは、**「演算が可換な群」** のことです。（可換群は、**アーベル群（abelian group）** とも呼ばれます。）

### 定義

集合 <img src="tmp/b15d92e08fe483b65746431228085e71.png" class="math-inline" /> とその上の演算 <img src="tmp/356513ded34229fab8ea72b65824828d.png" class="math-inline" />（掛け算の記号で書くことが多い）が次の条件を満たすとき、<img src="tmp/f16de060cce53c647d47daeb7667bc7c.png" class="math-inline" /> は**群**です。

1. **結合則**：<img src="tmp/9c2d4b1ce866bf7919c3822b73978c06.png" class="math-inline" />
2. **単位元の存在**：ある元 <img src="tmp/8ac7967375b961b8ef9657c37855da8a.png" class="math-inline" /> が存在して、すべての <img src="tmp/3fff980604c814e2b9b235523d0573fe.png" class="math-inline" /> について <img src="tmp/c11c71fa6c5153921cd608c6cf238c80.png" class="math-inline" />
3. **逆元の存在**：各 <img src="tmp/3fff980604c814e2b9b235523d0573fe.png" class="math-inline" /> に対して、ある <img src="tmp/c2f1aee58f84318be9878a799bdb403c.png" class="math-inline" /> が存在して <img src="tmp/63893eb1d5dcaddebf23c17d2df3232c.png" class="math-inline" />

さらに、演算が**可換**であるとき、つまり

4. **可換性**：<img src="tmp/b2b1e7702098506cbf51c4ed2c0bdce4.png" class="math-inline" />（すべての <img src="tmp/e87a4032aee85615c7b05c732d884a43.png" class="math-inline" /> について）

が成り立つとき、<img src="tmp/f16de060cce53c647d47daeb7667bc7c.png" class="math-inline" /> を**可換群（アーベル群）** といいます。

### 具体例

可換群（アーベル群）の具体例を、**集合と演算**の形で詳しく説明します。

__1. 整数全体 ℤ と足し算__

- **集合**：整数全体 ℤ = {…, −2, −1, 0, 1, 2, …}
- **演算**：通常の足し算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />
- **群の条件**：
  - 結合則：<img src="tmp/3c493847292d5ecd0ed50207eecf9e15.png" class="math-inline" />
  - 単位元：0（<img src="tmp/ffca29ae12a0b263eea32e3216bc608e.png" class="math-inline" />）
  - 逆元：各 <img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> に対して <img src="tmp/688916247f0baa4a6682864234b8d5b1.png" class="math-inline" />（<img src="tmp/e877542bbfec79a31a52ef7346b7677b.png" class="math-inline" />）
  - 可換性：<img src="tmp/17379e8bed96d9effdedfefa83539a0a.png" class="math-inline" />
- **特徴**：
  - 最も基本的な**無限可換群**の例
  - 環・体の「加法群」としても現れる

__2. 実数全体 ℝ と足し算__

- **集合**：実数全体 ℝ（有理数と無理数を含む）
- **演算**：通常の足し算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />
- **群の条件**：
  - 結合則・単位元 0・逆元 <img src="tmp/688916247f0baa4a6682864234b8d5b1.png" class="math-inline" />・可換性はいずれも成り立つ
- **特徴**：
  - 連続な無限可換群
  - 解析学（微積分）の舞台となる加法群

__3. 0 でない実数 ℝ∖{0} と掛け算__

- **集合**：0 を除いた実数全体 ℝ∖{0}
- **演算**：通常の掛け算 <img src="tmp/0169f361e6387b5186b557fdcc3af30b.png" class="math-inline" />
- **群の条件**：
  - 結合則：<img src="tmp/b10f04fb64de17be5bab47d88e1af13d.png" class="math-inline" />
  - 単位元：1（<img src="tmp/996609c20e01e79c7d38dec0eb5e6646.png" class="math-inline" />）
  - 逆元：各 <img src="tmp/d681d29d6af5e2ee73516eb6d63d2efe.png" class="math-inline" /> に対して <img src="tmp/0112b5aef8e9acf74a820260ace9949f.png" class="math-inline" />（<img src="tmp/092c1e787dc45f603a755926ed98a76e.png" class="math-inline" />）
  - 可換性：<img src="tmp/564da58d908677ca72cc53928342ae50.png" class="math-inline" />
- **特徴**：
  - 体 ℝ の**乗法群**（multiplicative group）の例
  - 0 を除くことで逆元が存在する

__4. 有理数全体 ℚ と足し算__

- **集合**：有理数全体 ℚ（整数の比 <img src="tmp/f4f4ff9a31e576a624cc206bae56eadb.png" class="math-inline" />）
- **演算**：足し算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />
- **群の条件**：
  - 結合則・単位元 0・逆元 <img src="tmp/688916247f0baa4a6682864234b8d5b1.png" class="math-inline" />・可換性が成り立つ
- **特徴**：
  - 可算無限の可換群
  - 体 ℚ の加法群としても重要

__5. 複素数全体 ℂ と足し算__

- **集合**：複素数全体 ℂ（<img src="tmp/27d66e7d0c359124224de4c66c932520.png" class="math-inline" />）
- **演算**：複素数の足し算
- **群の条件**：
  - 結合則・単位元 0・逆元 <img src="tmp/688916247f0baa4a6682864234b8d5b1.png" class="math-inline" />・可換性が成り立つ
- **特徴**：
  - 2次元のベクトル空間としての加法群
  - 幾何的には平面の平行移動に対応

__6. ベクトル空間 ℝⁿ とベクトルの足し算__

- **集合**：n 次元実ベクトル全体 ℝⁿ
- **演算**：ベクトルの足し算（成分ごとの和）
- **群の条件**：
  - 結合則・単位元 0（零ベクトル）・逆元 <img src="tmp/a06dfed74235df3bac1178c94c64b6c1.png" class="math-inline" />・可換性が成り立つ
- **特徴**：
  - 線形代数の基本構造
  - 幾何的には「平行移動の群」として解釈できる

__7. 有限可換群の例：ℤ/nℤ（整数 mod n）__

- **集合**：整数を n で割った余りの集合 <img src="tmp/a48383ebb7a2d7fcd2ef62f9376326d7.png" class="math-inline" />
- **演算**：mod n での足し算 <img src="tmp/47ae095c98ed13f7bd3c9b9d5e57dfa0.png" class="math-inline" />
- **群の条件**：
  - 結合則・単位元 0・逆元 <img src="tmp/4f75b9b6f7242fce3d39dc76577b7e73.png" class="math-inline" />・可換性が成り立つ
- **特徴**：
  - 有限個の元からなる可換群（位数 n）
  - 巡回群の典型例

__8. 円周群（circle group）S¹__

- **集合**：単位円周上の点（複素数で <img src="tmp/27bcc42ce780a78491a7f7c88dd6401b.png" class="math-inline" />）
- **演算**：複素数の掛け算（角度の足し算）
- **群の条件**：
  - 結合則・単位元 1・逆元 <img src="tmp/5fd91eb44ebee9ad8566595d3e864b75.png" class="math-inline" />・可換性が成り立つ
- **特徴**：
  - 連続な可換群（1次元トーラス）
  - フーリエ解析・表現論で重要

### 数学的な重要性

可換群の概念はこの後に続く環や体の概念を理解する上で前提となります。

__(1) 環・体の「土台」として__

環や体は、加法群が可換群であることが前提です。
例えば、整数環 ℤ、実数体 ℝ、複素数体 ℂ などは、いずれも足し算について可換群です。
可換群の理論が整っていないと、環や体の構造もきちんと扱えません。

__(2) 線形代数の基礎__

ベクトル空間は、ベクトルの足し算について可換群です。
線形写像・基底・次元などの概念は、この可換群構造の上に乗っています。
行列の演算や連立一次方程式の解法も、可換群の性質に支えられています。

__(3) ホモロジー・コホモロジー__

代数的位相幾何学では、ホモロジー群・コホモロジー群が可換群として現れます。
これらは位相空間の「穴」の数を数えたり、幾何的な不変量を与えたりします。
可換群の構造（自由部分・ねじれ部分など）が、空間の性質を反映します。

### 一言で言うと

> 可換群＝「逆元と単位元を持ち、演算が可換な集合」

です。

## 環

集合論における**環（ring）** の数学的な定義を、集合と写像の言葉で厳密に述べます。

### 環の定義

集合 <img src="tmp/a0f7a58538051aad071147a20c796777.png" class="math-inline" /> と、その上の2つの演算
- <img src="tmp/d12495036177e385eefe478957b368bb.png" class="math-inline" />（加法）
- <img src="tmp/cbd78280317c98b57d01b7e2ca9b2132.png" class="math-inline" />（乗法）

が与えられ、次の条件を満たすとき、<img src="tmp/ac09066d037a4fc2518d68cf9691a2cd.png" class="math-inline" /> を**環**といいます。

__1. 加法についての条件（可換群）__

<img src="tmp/e23d22c3ed1394e2db2fb5143a2c30ba.png" class="math-inline" /> は**可換群**である。すなわち：

- **結合則**：<img src="tmp/3c493847292d5ecd0ed50207eecf9e15.png" class="math-inline" />
- **単位元の存在**：ある元 <img src="tmp/e06bac59f1a862a6790bc0e2f8fc58a2.png" class="math-inline" /> が存在して、すべての <img src="tmp/e1eb396dc5c2d67d0a47d5f315f2dccb.png" class="math-inline" /> について  
  <img src="tmp/ffca29ae12a0b263eea32e3216bc608e.png" class="math-inline" />
- **逆元の存在**：各 <img src="tmp/e1eb396dc5c2d67d0a47d5f315f2dccb.png" class="math-inline" /> に対して、ある <img src="tmp/c0aec520a3d605bca410c3e207d72c15.png" class="math-inline" /> が存在して  
  <img src="tmp/2a602bf3561ab6054eab3831d570d813.png" class="math-inline" />
- **可換性**：<img src="tmp/17379e8bed96d9effdedfefa83539a0a.png" class="math-inline" />

__2. 乗法についての条件（結合的）__

<img src="tmp/d007d3819e711b1120d14dc8366ca7ca.png" class="math-inline" /> は**結合的**である。すなわち：

- **結合則**：<img src="tmp/9c2d4b1ce866bf7919c3822b73978c06.png" class="math-inline" />

（※乗法の可換性や単位元の存在は要求しません。  
　乗法の単位元 1 を持つ環を**単位的環（unital ring）** と呼びます。）

__3. 分配法則__

加法と乗法は**分配法則**で結びつく：

- <img src="tmp/9df7c1a8791128bdb1bf257e3c5d703d.png" class="math-inline" />
- <img src="tmp/6c098c26c035650ca33efab9a86f4bb5.png" class="math-inline" />

### 一言でまとめると

> 環＝  
> 「足し算については可換群、掛け算については結合的で、  
> 足し算と掛け算が分配法則で結びついている集合」

です。

### 具体例（集合論的に）

環の具体例を、**集合と演算**の形で挙げていきます。

__1. 整数環 ℤ__

- **集合**：整数全体 ℤ = {…, −2, −1, 0, 1, 2, …}
- **演算**：
  - 加法：通常の足し算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />
  - 乗法：通常の掛け算 <img src="tmp/0169f361e6387b5186b557fdcc3af30b.png" class="math-inline" />
- **環の条件**：
  - 加法について可換群（単位元 0、逆元 <img src="tmp/688916247f0baa4a6682864234b8d5b1.png" class="math-inline" />）
  - 乗法は結合的：<img src="tmp/b10f04fb64de17be5bab47d88e1af13d.png" class="math-inline" />
  - 分配法則：<img src="tmp/51fd3b8922d48d0db5322e3b9c6a0a7d.png" class="math-inline" /> など
- **特徴**：
  - 乗法の単位元 1 を持つので**単位的環**
  - 乗法は可換（<img src="tmp/564da58d908677ca72cc53928342ae50.png" class="math-inline" />）なので**可換環**
  - 0 以外の元が逆元を持つとは限らない（例：2 の逆元は ℤ にない）ので、**体ではない**

__2. 実数係数の多項式環 ℝ[x]__

- **集合**：実数係数の多項式全体  
  ℝ[x] = <img src="tmp/5bb04619b9a7f083dc15882e378c8f2c.png" class="math-inline" />
- **演算**：
  - 加法：多項式の足し算（同次の係数を足す）
  - 乗法：多項式の掛け算（分配して展開）
- **環の条件**：
  - 加法は可換群（零多項式が単位元 0）
  - 乗法は結合的（多項式の積の結合則）
  - 分配法則が成り立つ
- **特徴**：
  - 乗法の単位元は定数多項式 1
  - 乗法は可換なので**可換環**
  - 0 でない定数多項式以外は逆元を持たないので、**体ではない**

__3. 行列環 <img src="tmp/afebcf945ec290c8fb8af6e4562217f8.png" class="math-inline" />__

- **集合**：n 次正方行列全体  
  <img src="tmp/0cbe3e2961e04c8436c4a4af44bcd7b1.png" class="math-inline" />
- **演算**：
  - 加法：行列の成分ごとの和
  - 乗法：行列の積（行×列の内積）
- **環の条件**：
  - 加法は可換群（零行列が単位元 0）
  - 乗法は結合的（行列積の結合則）
  - 分配法則：<img src="tmp/c0fb3e45eb2d1facc1c4674bd2645a32.png" class="math-inline" /> など
- **特徴**：
  - 乗法の単位元は単位行列 <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" />
  - 乗法は**非可換**（一般に <img src="tmp/d7411775ba8964236f8bb7af6883ec30.png" class="math-inline" />）なので、**非可換環**
  - 正則行列以外は逆元を持たないので、**体ではない**

__4. 整数 mod n の環 ℤ/nℤ__

- **集合**：整数を n で割った余りの集合 <img src="tmp/a48383ebb7a2d7fcd2ef62f9376326d7.png" class="math-inline" />
- **演算**：
  - 加法：mod n での足し算
  - 乗法：mod n での掛け算
- **環の条件**：
  - 加法は可換群（単位元 0）
  - 乗法は結合的
  - 分配法則が成り立つ
- **特徴**：
  - 乗法の単位元は 1
  - 乗法は可換なので**可換環**
  - n が素数のときは**体**（0 以外の元が逆元を持つ）
  - n が合成数のときは体ではない（例：ℤ/4ℤ で 2 は逆元を持たない）

__5. 零環（自明な環）__

- **集合**：1 点集合 <img src="tmp/4774377971f2d39bdd07f44c6eeb2e2b.png" class="math-inline" />
- **演算**：
  - 加法：<img src="tmp/0c63e12ab54ed93952b3a833753627bd.png" class="math-inline" />
  - 乗法：<img src="tmp/a6503427f3d2b7677c6120d52a509d03.png" class="math-inline" />
- **環の条件**：
  - 加法は自明な可換群（単位元 0、逆元も 0）
  - 乗法は結合的
  - 分配法則は自明に成り立つ
- **特徴**：
  - 最も単純な環の例
  - 乗法の単位元も 0 とみなせるが、通常は「単位的環」とは区別する

__6. 函数環 <img src="tmp/5368d802ef03bb15d9060e0b2abaf4b3.png" class="math-inline" />__

- **集合**：閉区間 <img src="tmp/171d58b8063a24dc65a33bacb46a5fb7.png" class="math-inline" /> 上の実数値連続関数全体
- **演算**：
  - 加法：関数の点ごとの和 <img src="tmp/f207c10b51004b3b11e8be62cfefdc6b.png" class="math-inline" />
  - 乗法：関数の点ごとの積 <img src="tmp/a27dd0baef2068d1bc1d72fd0aa9aa9e.png" class="math-inline" />
- **環の条件**：
  - 加法は可換群（零関数が単位元）
  - 乗法は結合的
  - 分配法則が成り立つ
- **特徴**：
  - 乗法の単位元は定数関数 1
  - 乗法は可換なので**可換環**
  - 0 でない関数でも逆元（1/f）が常に存在するとは限らないので、**体ではない**

### 環が必要だった理由

環の定義が必要だった理由を、**歴史的・理論的・応用的**な観点から簡潔にまとめます。

__1. 共通の「2演算構造」を抽象化するため__

- 整数の足し算と掛け算、多項式の和と積、行列の和と積など、  
  一見バラバラに見える対象が、実は**同じルール**で動いていることに気づいた。
- そこで、「個々の具体例に依存せず、**“足し算と掛け算の組み合わせ”という共通構造だけを抜き出して定義**しよう」という発想から環の定義が生まれた。

__2. 体・代数・加群などの「土台」として__

- 体は「0 以外の元が逆元を持つ環」と見なせる。
- 線形代数のベクトル空間は、「体上の加群（環上の加群の特別な場合）」として定式化される。
- 環の理論が整っていないと、体・代数・加群・ホモロジーなど、より複雑な構造もきちんと扱えない。

__3. 数論・代数幾何での「数の拡張」を扱うため__

- 整数環 ℤ、多項式環 ℝ[x]、代数整数環などは、  
  「数の世界」を抽象化・拡張したものとして現れる。
- 環の定義により、**イデアル・剰余環・素イデアル**などの概念が導入され、  
  数論や代数幾何で「数の性質」を幾何的に研究できるようになった。

__4. 非可換構造（行列・作用素）を統一的に扱うため__

- 行列環 <img src="tmp/afebcf945ec290c8fb8af6e4562217f8.png" class="math-inline" /> や作用素環は、掛け算が非可換な環の典型例。
- 環の定義は可換性を要求しないので、**非可換な演算を持つ構造も同じ枠組みで扱える**。
- これにより、表現論・量子力学・作用素論などで「非可換な世界」を数学的に記述できる。

__5. 応用（符号・暗号・幾何）の基礎として__

- 符号理論では、線形符号を**有限体上のベクトル空間（＝体上の加群）** として定義するが、その土台は環の理論。
- 代数幾何では、多項式環のイデアルと代数多様体が対応する（ヒルベルトの零点定理）。
- 環の理論が整備されたことで、これらの応用分野でも「環」という共通言語で議論できるようになった。

__一言で言うと__

> 環の定義は、  
> 「足し算と掛け算が組み合わさった構造（数・多項式・行列など）に共通する性質を抽象化し、  
> 体・代数・加群・数論・幾何・物理などで統一的に扱うため」  
> に必要だった。

です。

### Pythonで確認

折角なのでPython使って環のイメージしてみたいと思います。
お題は、そこまでメチャうまいわけではありませんが。

__環の性質を確認する補助関数__


```python
def check_ring_properties(n=6):
    """
    ℤ/nℤ が環として満たす性質を簡単にチェック
    """
    print(f"=== ℤ/{n}ℤ の環としての性質チェック ===")

    # 加法の可換群か
    print("1. 加法は可換群か:")
    # 単位元 0 の存在
    print(f"   単位元: 0 (0 + a ≡ a mod {n})")
    # 各元に逆元が存在（-a mod n）
    for a in range(n):
        inv = (-a) % n
        print(f"   {a} の逆元: {inv} ({a} + {inv} ≡ 0 mod {n})")
    print("   → 可換群である")

    # 乗法の結合性（mod n では自明）
    print("2. 乗法は結合的か:")
    print("   (a*b)*c ≡ a*(b*c) mod n は常に成り立つ")
    print("   → 結合的である")

    # 分配法則（mod n では自明）
    print("3. 分配法則は成り立つか:")
    print("   a*(b+c) ≡ a*b + a*c mod n")
    print("   (a+b)*c ≡ a*c + b*c mod n")
    print("   → 分配法則が成り立つ")

    # 零因子の有無
    zero_divisors = []
    for a in range(1, n):
        for b in range(1, n):
            if (a * b) % n == 0:
                zero_divisors.append((a, b))
    if zero_divisors:
        print("4. 零因子の有無:")
        print(f"   零因子あり（例: {zero_divisors[0][0]} × {zero_divisors[0][1]} ≡ 0 mod {n}）")
    else:
        print("4. 零因子の有無:")
        print("   零因子なし → ℤ/{n}ℤ は整域（かつ体）")

# 実行例
check_ring_properties(n=6)
check_ring_properties(n=5)
```

```
=== ℤ/6ℤ の環としての性質チェック ===
1. 加法は可換群か:
   単位元: 0 (0 + a ≡ a mod 6)
   0 の逆元: 0 (0 + 0 ≡ 0 mod 6)
   1 の逆元: 5 (1 + 5 ≡ 0 mod 6)
   2 の逆元: 4 (2 + 4 ≡ 0 mod 6)
   3 の逆元: 3 (3 + 3 ≡ 0 mod 6)
   4 の逆元: 2 (4 + 2 ≡ 0 mod 6)
   5 の逆元: 1 (5 + 1 ≡ 0 mod 6)
   → 可換群である
2. 乗法は結合的か:
   (a*b)*c ≡ a*(b*c) mod n は常に成り立つ
   → 結合的である
3. 分配法則は成り立つか:
   a*(b+c) ≡ a*b + a*c mod n
   (a+b)*c ≡ a*c + b*c mod n
   → 分配法則が成り立つ
4. 零因子の有無:
   零因子あり（例: 2 × 3 ≡ 0 mod 6）
=== ℤ/5ℤ の環としての性質チェック ===
1. 加法は可換群か:
   単位元: 0 (0 + a ≡ a mod 5)
   0 の逆元: 0 (0 + 0 ≡ 0 mod 5)
   1 の逆元: 4 (1 + 4 ≡ 0 mod 5)
   2 の逆元: 3 (2 + 3 ≡ 0 mod 5)
   3 の逆元: 2 (3 + 2 ≡ 0 mod 5)
   4 の逆元: 1 (4 + 1 ≡ 0 mod 5)
   → 可換群である
2. 乗法は結合的か:
   (a*b)*c ≡ a*(b*c) mod n は常に成り立つ
   → 結合的である
3. 分配法則は成り立つか:
   a*(b+c) ≡ a*b + a*c mod n
   (a+b)*c ≡ a*c + b*c mod n
   → 分配法則が成り立つ
4. 零因子の有無:
   零因子なし → ℤ/{n}ℤ は整域（かつ体）
```

__出力の意味__

この結果は、**「環の定義と、環が体になる条件」** を具体的に示しています。

どちらの結果も、

- 加法は可換群（単位元 0、各元に逆元がある）
- 乗法は結合的
- 分配法則が成り立つ

という**環の定義**を満たしています。  
したがって、

> ℤ/6ℤ も ℤ/5ℤ も、**環**である

ということが確認できます。

## 体

集合論における**体（field）** は、**「集合と2つの演算（足し算・掛け算）の組」** として、集合論の枠組みの中で定義されるものです。

### 集合論的な定義

集合 <img src="tmp/2869f80330e3b768b3feec4cdf4c7583.png" class="math-inline" /> と、その上の2つの演算
- <img src="tmp/61bbd29a51c401a6f183514386099c73.png" class="math-inline" />（加法）
- <img src="tmp/b6ae7082b416c1d387bd2ccfcc704072.png" class="math-inline" />（乗法）

が与えられ、次の条件を満たすとき、<img src="tmp/afee8c2f8af443e2dae18efdb623912c.png" class="math-inline" /> を**体**といいます。

1. <img src="tmp/17f29554a4169fb8d50bc2f1692d3b25.png" class="math-inline" /> は**可換群**（単位元 0 と逆元 <img src="tmp/688916247f0baa4a6682864234b8d5b1.png" class="math-inline" /> を持つ）
2. <img src="tmp/736c8a596d6fe07c4108caace61a7883.png" class="math-inline" /> も**可換群**（単位元 1 と逆元 <img src="tmp/eb295687d57cd03ff5da483189c389d5.png" class="math-inline" /> を持つ）
3. **分配法則**が成り立つ：
   

<div class="math-display-container"><img src="tmp/b3a5fc123c435242e42c1806f11005db.png" class="math-display" /></div>



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

- **集合**：有理数全体 ℚ = <img src="tmp/162ac94cd19a9f0637f1381b9932fb0b.png" class="math-inline" />
- **演算**：
  - 加法：通常の足し算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />
  - 乗法：通常の掛け算 <img src="tmp/0169f361e6387b5186b557fdcc3af30b.png" class="math-inline" />
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 <img src="tmp/688916247f0baa4a6682864234b8d5b1.png" class="math-inline" />）
  - 0 以外の有理数は乗法について可換群（単位元 1、逆元 <img src="tmp/0112b5aef8e9acf74a820260ace9949f.png" class="math-inline" />）
  - 分配法則が成り立つ
- **特徴**：
  - 最も基本的な**無限体**の一つ
  - 数論・代数の基礎となる体

__2. 実数体 ℝ__

- **集合**：実数全体 ℝ（有理数と無理数を含む）
- **演算**：通常の足し算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" /> と掛け算 <img src="tmp/0169f361e6387b5186b557fdcc3af30b.png" class="math-inline" />
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 <img src="tmp/688916247f0baa4a6682864234b8d5b1.png" class="math-inline" />）
  - 0 以外の実数は乗法について可換群（単位元 1、逆元 <img src="tmp/0112b5aef8e9acf74a820260ace9949f.png" class="math-inline" />）
  - 分配法則が成り立つ
- **特徴**：
  - **完備な順序体**（順序数は9章で扱う概念実数の連続性・極限が定義できる）
  - 解析学（微積分）の舞台

__3. 複素数体 ℂ__

- **集合**：複素数全体 ℂ = <img src="tmp/15cf04ec05a89e654cf83034bf5bbcf4.png" class="math-inline" />
- **演算**：複素数の足し算・掛け算
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 <img src="tmp/b5d50e013d1a55fa8fce1b87643e986e.png" class="math-inline" />）
  - 0 以外の複素数は乗法について可換群（単位元 1、逆元 <img src="tmp/e0180ce3291d0354345fe8b0ecf4c4da.png" class="math-inline" />）
  - 分配法則が成り立つ
- **特徴**：
  - **代数的閉体**（すべての多項式が根を持つ）
  - 線形代数・量子力学などで重要

__4. 有限体（ガロア体）𝔽ₚ__

- **集合**：整数を素数 <img src="tmp/eb83e8cbc0f1e2e809a7c7e8a5cef02f.png" class="math-inline" /> で割った余りの集合 <img src="tmp/f8f39ce72b45d7161e3c2e09442aac68.png" class="math-inline" />
- **演算**：mod <img src="tmp/eb83e8cbc0f1e2e809a7c7e8a5cef02f.png" class="math-inline" /> での足し算・掛け算
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 <img src="tmp/8ff74770061ad8054f52dd4a2b7c130d.png" class="math-inline" />）
  - 0 以外の元は乗法について可換群（単位元 1、逆元は mod <img src="tmp/eb83e8cbc0f1e2e809a7c7e8a5cef02f.png" class="math-inline" /> での逆数）
  - 分配法則が成り立つ
- **例**：
  - 𝔽₂ = <img src="tmp/d8429da08828f207b99be93378783e65.png" class="math-inline" />：足し算は XOR、掛け算は AND
  - 𝔽₃ = <img src="tmp/fadd4bf13a996816a6459b1fda74aa5c.png" class="math-inline" />：mod 3 の演算
- **特徴**：
  - **有限個の元からなる体**
  - 符号理論・暗号・組合せ論で使われる

__5. 有理関数体 ℚ(x)__

- **集合**：有理数係数の多項式の比全体  
  ℚ(x) = <img src="tmp/68e4b5fda385363ba371d6812c1ebfd8.png" class="math-inline" />
- **演算**：有理関数の足し算・掛け算（通分・約分）
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 <img src="tmp/db561846b975d3983795eed80f395ba9.png" class="math-inline" />）
  - 0 以外の有理関数は乗法について可換群（単位元 1、逆元 <img src="tmp/80df1f9e0526317eaf16f830ba93c3a3.png" class="math-inline" />）
  - 分配法則が成り立つ
- **特徴**：
  - **無限次元の体拡大**（ℚ の超越拡大）
  - 代数幾何・関数体の理論で重要

__6. 代数体（例：ℚ(√2)）__

- **集合**：ℚ(√2) = <img src="tmp/5948922de440478079705d8f37d7e5f8.png" class="math-inline" />
- **演算**：通常の足し算・掛け算（√2 の性質 √2²=2 を使う）
- **体の条件**：
  - 加法は可換群（単位元 0、逆元 <img src="tmp/629f74c9eec88f2cf1dcf99734b21287.png" class="math-inline" />）
  - 0 以外の元は乗法について可換群（逆元は共役を用いて計算）
  - 分配法則が成り立つ
- **特徴**：
  - ℚ の**有限次代数拡大体**
  - 数論（代数整数論）で重要

__7. p-進数体 ℚₚ__

- **集合**：有理数体 ℚ を素数 <img src="tmp/eb83e8cbc0f1e2e809a7c7e8a5cef02f.png" class="math-inline" /> に関する「p-進距離」で完備化したもの
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



## 演習

集合の演算に関する証明問題をいくつか出題します。  
必要に応じて、全体集合 <img src="tmp/2321eadfdfcad762b559d2c1971bf109.png" class="math-inline" /> や補集合 <img src="tmp/785a6da13086feb5b6bf91dedbb16481.png" class="math-inline" /> も使って構いません。

### 問題

__問題1（基本）__

集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/d757ccb832383318d74ced1f828bf1f3.png" class="math-display" /></div>





__問題2（分配法則）__

集合 <img src="tmp/b9e3ac1dae0ae9cdc4c311c262881ff8.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/e21b4150c9db75709623e505e5cd1abf.png" class="math-display" /></div>





__問題3（ド・モルガンの法則）__

全体集合 <img src="tmp/2321eadfdfcad762b559d2c1971bf109.png" class="math-inline" /> とその部分集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/26be2990254c8124c5d13ab99c882ba3.png" class="math-display" /></div>




__問題4（差集合の性質）__

集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/e28b240b57f06232a875cfdcf4f15e21.png" class="math-display" /></div>


ただし、全体集合 <img src="tmp/2321eadfdfcad762b559d2c1971bf109.png" class="math-inline" /> を考え、<img src="tmp/cba5493b4c86dfd2457efa0fb0c8497d.png" class="math-inline" /> は <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の補集合とする。

__問題5（少し応用）__

集合 <img src="tmp/b9e3ac1dae0ae9cdc4c311c262881ff8.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/7a40fbe1ee179d09ff9b7dadc9a3ce3f.png" class="math-display" /></div>



__問題6（包含関係の証明）__

集合 <img src="tmp/b9e3ac1dae0ae9cdc4c311c262881ff8.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/0223b0cfe1c8eb9859d6ab2b0dd39917.png" class="math-display" /></div>



__問題7（対称差の性質）__

集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> の**対称差**を


<div class="math-display-container"><img src="tmp/be2f5a196ec392e0d5244415a039b9ca.png" class="math-display" /></div>


と定義する。このとき、次を示せ。


<div class="math-display-container"><img src="tmp/f541c4c761d222b3f61b2c6c457c7758.png" class="math-display" /></div>



### 解答

先ほどの各問題について、順に証明します。

__問題1：<img src="tmp/1762f7b22e12f7a6daa32efec4a9afc3.png" class="math-inline" />__

**証明**  
<img src="tmp/d3cd613dbef8bc0f1be0fb2d340867e0.png" class="math-inline" /> を仮定する。  
任意の元 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> について、

- <img src="tmp/b1ba2e7260eb6aa8369fbf40a2bb6dc5.png" class="math-inline" />  
  <img src="tmp/6d4869a1dd5b67e543f5da65e4d24f9f.png" class="math-inline" /> または <img src="tmp/740cd2ca02706f788216698c8d77470f.png" class="math-inline" />  
  <img src="tmp/b7ed6413025b8a13248b72426f378bd8.png" class="math-inline" />（∵ <img src="tmp/d3cd613dbef8bc0f1be0fb2d340867e0.png" class="math-inline" /> より <img src="tmp/76ba24283104f90f0316b1ce5de3e717.png" class="math-inline" />）  
  よって <img src="tmp/6f1c1e7352b1ab77bdb8462a671db384.png" class="math-inline" />。

- 一方、<img src="tmp/245e635d71d3c18cef49be6061dba08c.png" class="math-inline" /> または <img src="tmp/740cd2ca02706f788216698c8d77470f.png" class="math-inline" />  
  <img src="tmp/565ac3978d5ec8305836cc09b802f2af.png" class="math-inline" />  
  よって <img src="tmp/c242954dc23fd0a48d9dc34145489b5c.png" class="math-inline" />。

以上より <img src="tmp/bba4a1b5e26a99977185abd603c01530.png" class="math-inline" />。□

__問題2：分配法則 <img src="tmp/e93c71e00edc72e38f8e1cc140001036.png" class="math-inline" />__

**証明**  
任意の元 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> について、



<div class="math-display-container"><img src="tmp/5178714c5bc8d79462817ac389775bec.png" class="math-display" /></div>



よって両辺は等しい。□

__問題3：ド・モルガンの法則 <img src="tmp/64563eeeac73f42a18ab1bfb40a3f132.png" class="math-inline" />__

**証明**  
全体集合を <img src="tmp/2321eadfdfcad762b559d2c1971bf109.png" class="math-inline" /> とし、任意の元 <img src="tmp/557ea14c25d45503d350fc0cfecb6e35.png" class="math-inline" /> について、



<div class="math-display-container"><img src="tmp/30e91c7fa983f2351bb7c7e440800940.png" class="math-display" /></div>



よって <img src="tmp/64563eeeac73f42a18ab1bfb40a3f132.png" class="math-inline" />。□

__問題4：差集合 <img src="tmp/2bb46fe2ada3dbc7868215a8c3befca8.png" class="math-inline" />__

**証明**  
全体集合 <img src="tmp/2321eadfdfcad762b559d2c1971bf109.png" class="math-inline" /> を考え、任意の元 <img src="tmp/557ea14c25d45503d350fc0cfecb6e35.png" class="math-inline" /> について、



<div class="math-display-container"><img src="tmp/d587503a3fb3fc703eac88f066b94012.png" class="math-display" /></div>



よって <img src="tmp/2bb46fe2ada3dbc7868215a8c3befca8.png" class="math-inline" />。□

__問題5：<img src="tmp/e4e80cccebb1290ff499a65ffe5eeda5.png" class="math-inline" />__

**証明**  
任意の元 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> について、



<div class="math-display-container"><img src="tmp/fdc8c2bfd1bc946e2b62780605339d72.png" class="math-display" /></div>



よって両辺は等しい。□

__問題6：<img src="tmp/0ed15ea66722d6cc025e34e9b973eec1.png" class="math-inline" />__

**証明**  
<img src="tmp/d3cd613dbef8bc0f1be0fb2d340867e0.png" class="math-inline" /> を仮定する。  
任意の元 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> について、



<div class="math-display-container"><img src="tmp/fb725bf2415ba747501306be1ac06c29.png" class="math-display" /></div>



よって <img src="tmp/4cf05b99d2506a1ed928577b13de1d85.png" class="math-inline" />。□

__問題7：対称差 <img src="tmp/b562b3edcd14a07da9adee43c6ae2cc4.png" class="math-inline" />__

**証明**  
定義より


<div class="math-display-container"><img src="tmp/be2f5a196ec392e0d5244415a039b9ca.png" class="math-display" /></div>


である。  
任意の元 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> について、



<div class="math-display-container"><img src="tmp/576bf6686e3ca62792f9b235e11b0fd9.png" class="math-display" /></div>



よって <img src="tmp/b562b3edcd14a07da9adee43c6ae2cc4.png" class="math-inline" />。□


<div style="page-break-before:always"></div>



# べき集合、直積集合、写像

## べき集合

### べき集合の定義

べき集合（冪集合）の数学的な定義は次の通りです。

集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に対して、「**A の部分集合全体の集合**」を、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**べき集合**（power set）といいます。

- 記号：<img src="tmp/e1fa47c9075271d02cf8ead81516d053.png" class="math-inline" /> または <img src="tmp/4edf2a90895c2e5712fabf97dcbecf90.png" class="math-inline" />
- 定義式：
  

<div class="math-display-container"><img src="tmp/d7a6339b207f65bde3419c6fae210c48.png" class="math-display" /></div>


  「<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の部分集合である」という条件を満たす <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 全体の集合です。

__例__

__例1：有限集合の場合__

<img src="tmp/9abb08ba245573d07135446115b59007.png" class="math-inline" /> のとき、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の部分集合は：

- 0個の元：<img src="tmp/36682ee60666b219d2be3577c1bfda62.png" class="math-inline" />
- 1個の元：<img src="tmp/c7756f286789c539f1f7f137deda808c.png" class="math-inline" />
- 2個の元：<img src="tmp/eeaf6518a634a5dc912a5093dc67de6b.png" class="math-inline" />

したがって、


<div class="math-display-container"><img src="tmp/ee85efe7e041bab0b226854de542ffe3.png" class="math-display" /></div>


となります。

__例2：空集合のべき集合__

<img src="tmp/c7cb2a0954714e4427d98c5fb233f846.png" class="math-inline" /> のとき、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の部分集合は <img src="tmp/36682ee60666b219d2be3577c1bfda62.png" class="math-inline" /> だけです（空集合は任意の集合の部分集合）。  
よって、


<div class="math-display-container"><img src="tmp/123f29b83f605fc32226ad99eed30cc7.png" class="math-display" /></div>


となります。

### 元の個数（有限集合の場合）

<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が有限集合で、元の個数が <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> 個のとき、べき集合 <img src="tmp/e1fa47c9075271d02cf8ead81516d053.png" class="math-inline" /> の元の個数は <img src="tmp/edf6abb9fb79e3c90b5730fe145ed4d8.png" class="math-inline" /> 個です。

- 例：<img src="tmp/9abb08ba245573d07135446115b59007.png" class="math-inline" />（<img src="tmp/d6f25a0b3782112da787eda22e2feea4.png" class="math-inline" />）のとき、  
  <img src="tmp/e1fa47c9075271d02cf8ead81516d053.png" class="math-inline" /> の元は <img src="tmp/b18a410fc1b228763069c8fb5a8bb64a.png" class="math-inline" /> 個でした。

### 記号の由来

- <img src="tmp/e1fa47c9075271d02cf8ead81516d053.png" class="math-inline" /> の「P」は Power set の頭文字です。
- <img src="tmp/4edf2a90895c2e5712fabf97dcbecf90.png" class="math-inline" /> という記号は、「A の各元について『含めるか・含めないか』の2通りがある」ことから、 部分集合の総数が <img src="tmp/814e03bcc3063fe616bcb8bcc3124547.png" class="math-inline" /> になることに対応しています。

### 性質（ざっくり）

- 任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> について、<img src="tmp/306c68c76e187df22b04396302bbd30e.png" class="math-inline" /> かつ <img src="tmp/ecac6acd5329c85c8d0998772d7a9c32.png" class="math-inline" /> です。
- <img src="tmp/d3cd613dbef8bc0f1be0fb2d340867e0.png" class="math-inline" /> ならば <img src="tmp/a178caa5e649bf4c058e2a8d934fed00.png" class="math-inline" /> です。
- べき集合は「集合の集合」であり、集合論の公理系の中で厳密に定義されます。

### なぜ必要か？
べき集合は、主に次のような理由で必要とされ、使われています。

__1. 数学的な「構造」を扱う土台として必要__

数学では、ある集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に対して「その部分集合全体」を一つの新しい集合として扱いたい場面がよくあります。

- 例：位相空間（トポロジー）  
  位相空間とは「開集合の族」を指定することで定義されますが、  
  その「開集合の族」は、**全体集合のべき集合の部分集合**として与えられます。
  - つまり、<img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> の中から「開集合」とみなすものを選び出して、位相を定義します。

- 例：測度論・確率論  
  確率空間では、「事象」を標本空間 <img src="tmp/083ab9161ae5726daac5d5ba2e71af7f.png" class="math-inline" /> の部分集合として扱います。  
  したがって、**事象全体の集合は <img src="tmp/9b55b42a880435d34933b84bb8bb6dba.png" class="math-inline" /> の部分集合**として定義されます（完全加法族・σ-集合体など）。

このように、べき集合は「集合の集合（集合族）」を扱う際の**基本の舞台**として必要です。

__2. 計算機科学・離散数学での応用__

__有限集合のべき集合と組み合わせ__

有限集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> のべき集合 <img src="tmp/e1fa47c9075271d02cf8ead81516d053.png" class="math-inline" /> は、  
「A の元を選ぶ／選ばない」という**2択の組み合わせ**全体に対応します。

- 例：<img src="tmp/23f3388429fa25477a55b940700eda27.png" class="math-inline" /> のとき、  
  <img src="tmp/e1fa47c9075271d02cf8ead81516d053.png" class="math-inline" /> は
  

<div class="math-display-container"><img src="tmp/5ef283bd8b2d510e9af3af9afe606e86.png" class="math-display" /></div>


  となり、これは「a, b, c をそれぞれ含めるかどうか」の <img src="tmp/769dc63165bdc530f2a86b57d18d1298.png" class="math-inline" /> 通りのパターン全体です。

この性質は、
- 部分集合の列挙（組合せ最適化）
- 状態空間の表現（オートマトン、モデル検査）
などで利用されます。

__オートマトン理論（べき集合構成）__
決定性有限オートマトン（DFA）を非決定性有限オートマトン（NFA）から構成する際に、  
**NFA の状態集合のべき集合**を DFA の状態集合として使います。  
- NFA の「あり得る状態の集合」を 1 つの状態とみなすことで、DFA を構成します。

__3. 確率・情報・論理との関係__

__確率空間__

標本空間 <img src="tmp/083ab9161ae5726daac5d5ba2e71af7f.png" class="math-inline" /> の各「事象」は <img src="tmp/083ab9161ae5726daac5d5ba2e71af7f.png" class="math-inline" /> の部分集合です。  
したがって、**事象全体は <img src="tmp/9b55b42a880435d34933b84bb8bb6dba.png" class="math-inline" /> の部分集合**として定義されます。  
（実際には、すべての部分集合に確率を割り当てられない場合もあるので、σ-集合体という <img src="tmp/9b55b42a880435d34933b84bb8bb6dba.png" class="math-inline" /> の部分集合族を考えます。）

__情報量（ビット数）との対応__

有限集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> のべき集合の要素数は <img src="tmp/814e03bcc3063fe616bcb8bcc3124547.png" class="math-inline" /> です。  
これは、「A の元1つを特定するのに必要な情報量（ビット数）」と対応します。
- <img src="tmp/0d25d178b7971239fb0d782eb9be6877.png" class="math-inline" /> のとき、A の元1つを特定するには <img src="tmp/47b56f88c24fc3ad64077e11d01b8cc7.png" class="math-inline" /> ビット必要ですが、  
  <img src="tmp/e1fa47c9075271d02cf8ead81516d053.png" class="math-inline" /> の元1つ（＝A の部分集合1つ）を特定するには <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> ビット必要です。

__4. 抽象的な数学構造の定義__

- **ブール代数**：  
  べき集合 <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> は、和集合・共通部分・補集合などの演算により、  
  ブール代数の典型的な例になります。
- **順序集合・束（lattice）**：  
  <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> は包含関係 <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" /> に関して完備束になります。

このように、べき集合は「集合の構造」を調べるための**標準的なモデル**としても重要です。


## 集合族

集合族の数学的な定義は次の通りです。

### 1. 集合族の定義

**集合族**（family of sets）とは、「集合を元として持つ集合」のことです。

もう少し形式的には：

- ある集合 <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" />（**添字集合**）と、各 <img src="tmp/9986a8a5f7309c550d762fa0a9c779d4.png" class="math-inline" /> に対応する集合 <img src="tmp/b4cf280200862773bb25975b26b9248c.png" class="math-inline" /> が与えられたとき、これらの集合全体の集まり
  

<div class="math-display-container"><img src="tmp/20d5cb9aa2ba13b59e02aadbbc96a659.png" class="math-display" /></div>


  を**集合族**といいます。

- 記号としては、
  

<div class="math-display-container"><img src="tmp/e3053ff68d1cf1509bb7f9f3353e654c.png" class="math-display" /></div>


  のように書くことが多いです。

### 2. べき集合との関係

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**べき集合** <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> は、「<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合全体」の集合ですから、<img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> の元はすべて集合（<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合）です。

したがって、<img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> の**任意の部分集合**は、自動的に集合族になります。

- 例：  
  <img src="tmp/370d0b7a1f97fce98a2a6d59bc55410f.png" class="math-inline" /> のとき、  
  

<div class="math-display-container"><img src="tmp/6536420d0316d2aa3a7fb2adda68f0ad.png" class="math-display" /></div>


  は <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> の部分集合であり、集合族です。

### 3. 集合族の例

__例1：有限な集合族__

添字集合 <img src="tmp/eba27d24bf0c9fc21d4ea0d8a02add6d.png" class="math-inline" /> とし、


<div class="math-display-container"><img src="tmp/aee57d42e25eef7f60810a269e3e21da.png" class="math-display" /></div>


とすると、


<div class="math-display-container"><img src="tmp/c8976958990b57f202dc7c63bb16e175.png" class="math-display" /></div>


は集合族です。

__例2：自然数の部分集合の族__

添字集合 <img src="tmp/c770126685581d1b2c7cd93e05eea76f.png" class="math-inline" />（自然数全体）とし、


<div class="math-display-container"><img src="tmp/879c89df5c51a5c51310830a1f9a4da8.png" class="math-display" /></div>


とすると、


<div class="math-display-container"><img src="tmp/66aa0cd707085f03f5051311667ff451.png" class="math-display" /></div>


は集合族です。

__例3：空でない集合だけからなる族__

集合族は「空集合だけからなる集合」も含みますが、多くの文脈では「空でない集合からなる族」を考えることがあります。

- 例：  
  <img src="tmp/1ac53c678fe57536fa591b6e5e6d06b6.png" class="math-inline" /> は集合族で、どの元も空集合ではありません。

### 4. 和集合・共通部分との関係（参考）

集合族 <img src="tmp/927219c1b307f7f95d2fc635d02c6635.png" class="math-inline" /> が与えられたとき、その**和集合**と**共通部分**は次のように定義されます。

- **和集合**：
  

<div class="math-display-container"><img src="tmp/089894e4560a36cc24bac12200a28d1f.png" class="math-display" /></div>


- **共通部分**：
  

<div class="math-display-container"><img src="tmp/90bbd28f20a1d50d7cda1cf4de7f7966.png" class="math-display" /></div>



集合族を考えることで、無限個の集合に対する和集合・共通部分も自然に定義できます。

### 5. 集合族が必要な理由

集合族は、主に次のような理由で必要とされています。

__1. 「集合の集まり」を一つの対象として扱うため__

数学では、**複数の集合をまとめて扱いたい**場面がたくさんあります。

- 例：  
  「自然数 n ごとに集合 <img src="tmp/bb59ca86b348e5fe1764c39af957d722.png" class="math-inline" /> を考える」とき、これらを個別に扱うのではなく、
  

<div class="math-display-container"><img src="tmp/66aa0cd707085f03f5051311667ff451.png" class="math-display" /></div>


  という**集合族**としてまとめて扱うと、  
  - 和集合 <img src="tmp/322265bd82c9c4d59ec9bfbeb9b43b45.png" class="math-inline" />  
  - 共通部分 <img src="tmp/15b1fa226d0efbf35bad5b3ec6ca8c02.png" class="math-inline" />  
  などを一括で定義・議論できます。

このように、「集合の集まり」を一つの数学的対象（集合族）として扱うことで、無限個の集合に対しても統一的に操作できるようになります。

__2. 位相空間・測度論などで「構造」を定義するため__

__位相空間（トポロジー）__

位相空間とは、「どの部分集合を『開集合』とみなすか」を決めたものです。  
この「開集合の集まり」は、**全体集合の部分集合からなる集合族**です。

- 全体集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> のべき集合 <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> の部分集合族 <img src="tmp/056d48b5fe2947dd8a725fcd4b9b6236.png" class="math-inline" /> を「開集合族」と定義し、その <img src="tmp/056d48b5fe2947dd8a725fcd4b9b6236.png" class="math-inline" /> が満たすべき条件（任意個の和集合で閉じる、有限個の共通部分で閉じるなど）を課すことで、位相空間を定義します。

ここで「集合族」という概念がないと、「開集合の集まり」を数学的に厳密に扱えません。

__測度論・確率論__

確率空間では、「事象」を標本空間 <img src="tmp/083ab9161ae5726daac5d5ba2e71af7f.png" class="math-inline" /> の部分集合として扱います。  
しかし、**すべての部分集合に確率を割り当てられるとは限らない**ため、「確率が定義できる事象の集まり」として、<img src="tmp/9b55b42a880435d34933b84bb8bb6dba.png" class="math-inline" /> の部分集合族（σ-集合体）を考えます。

- この「事象の族」が集合族であり、その上に測度（確率）を定義することで、確率空間が構成されます。

__3. 無限和・無限積など「無限個の演算」を扱うため__

集合族 <img src="tmp/927219c1b307f7f95d2fc635d02c6635.png" class="math-inline" /> があると、添字集合 <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> が無限集合でも、

- 和集合 <img src="tmp/a9b05ef5bc1b66fc3f36d59f169af84a.png" class="math-inline" />
- 共通部分 <img src="tmp/097aaad7bf0f2209a97db40fc180edd4.png" class="math-inline" />

を自然に定義できます。

- 例：  
  <img src="tmp/c4d93baa960b37df87d627c8704af105.png" class="math-inline" />（実数全体）として、各 <img src="tmp/d1224dae84fa05efe4ea542bb4fcd258.png" class="math-inline" /> に対し <img src="tmp/4eed5092506637ed4a2dfcd26439670d.png" class="math-inline" />（実数直線の左半直線）とすると、  
  

<div class="math-display-container"><img src="tmp/ee9f993da12701eb69d2bc31f6063f39.png" class="math-display" /></div>


  などが定義できます。

このように、**無限個の集合に対する演算**を厳密に扱うために、集合族という枠組みが必要です。

__4. 数学的構造の一般化（フィルター・イデアルなど）__

集合族の特別なものとして、

- **フィルター**：ある種の「大きい集合」の集まり
- **イデアル**：ある種の「小さい集合」の集まり

などがあり、これらは

- 位相空間のコンパクト性
- モデル理論・集合論の強制法
- 測度論・確率論

など、さまざまな分野で重要な役割を果たします。

これらも「集合の集まり」として定義されるため、集合族の概念が土台になっています。

## 順序対

順序対（ordered pair）とは、**2つの対象を順序を付けて組にしたもの**です。

### 1. 順序対の直感的な意味

2つの対象 <img src="tmp/c1e88d0682c98b2b10bf8bc41b703bb2.png" class="math-inline" /> に対して、「1番目が <img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" />、2番目が <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> である組」を <img src="tmp/b01473ed500218be6ffa478d88033ca3.png" class="math-inline" /> と書き、これを**順序対**といいます。

重要な点は：

- **順序が意味を持つ**：  
  <img src="tmp/017eec20e3244699618c5a6bacbbdd78.png" class="math-inline" /> であることがあります（<img src="tmp/edf863102d3ee527ea5ee52b5a69f4bb.png" class="math-inline" /> のとき）。
- **成分が等しいときのみ等しい**：  
  <img src="tmp/632eba6ac109cc43a246f39054e9b527.png" class="math-inline" /> となるのは、<img src="tmp/fabe0eb49e78c56a07f4ffcc7f12d43b.png" class="math-inline" /> かつ <img src="tmp/9ac6973a9eaa61d82c1e9bd5173396ba.png" class="math-inline" /> のときに限ります。

例：
- 座標平面の点 <img src="tmp/b8e6559075e946b5e5fc11a136a89d21.png" class="math-inline" /> と <img src="tmp/ef7383fafa5089c21453dbaee3bac802.png" class="math-inline" /> は別の点です。
- 姓と名の組 <img src="tmp/5509252fc4b3e96b0a6f454dc5d2bb70.png" class="math-inline" /> と <img src="tmp/0a129e518251b142f48e7bd1f254b0fb.png" class="math-inline" /> は別のものとみなします。

### 2. 数学的な定義（クーラトフスキーの定義）

集合論では、順序対を集合として厳密に定義する必要があります。  
その一つの方法が**クーラトフスキー（Kuratowski）の定義**です。



<div class="math-display-container"><img src="tmp/420355e86348597659c78847cb5fd06c.png" class="math-display" /></div>



この定義により、順序対を「集合の集合」として扱えます。

この定義がうまくいく理由（直感的な説明）：
- <img src="tmp/c133a996a2b92daa0fdfb69c4f17cd7a.png" class="math-inline" /> と <img src="tmp/3700ed35ed509aed50f3e6c7e00bdb5d.png" class="math-inline" /> という2つの集合の情報から、「1番目の成分は <img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" />」「2番目の成分は <img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> か <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" />」と読み取れます。
- 特に、<img src="tmp/6c1658f93e555b5876a1c0c94f8d11db.png" class="math-inline" /> かつ <img src="tmp/3ba53bf6d0bd42e091865e3694e497d4.png" class="math-inline" /> から、<img src="tmp/fabe0eb49e78c56a07f4ffcc7f12d43b.png" class="math-inline" /> かつ <img src="tmp/9ac6973a9eaa61d82c1e9bd5173396ba.png" class="math-inline" /> が導かれ、<img src="tmp/e038f517777abded7b3750744162aef7.png" class="math-inline" /> が成り立ちます。

このようにして、順序対を純粋に集合の言葉で定義できます。

### 3. 通常の「組」との違い

日常語で「組」と言うと、順序を気にしないこともありますが、数学の**順序対**は常に順序を区別します。

- 例：  
  <img src="tmp/3700ed35ed509aed50f3e6c7e00bdb5d.png" class="math-inline" /> という集合（順序なし）は、<img src="tmp/bcd70a3567e8d2bf75709c8bbabfdd7f.png" class="math-inline" /> と同じです。  
  しかし、順序対 <img src="tmp/b01473ed500218be6ffa478d88033ca3.png" class="math-inline" /> と <img src="tmp/3e12ae606c685fd305ed9b5e5ee460f7.png" class="math-inline" /> は一般に異なります。

### 4. 順序対の一般化：n-組

2つの対象の順序対 <img src="tmp/b01473ed500218be6ffa478d88033ca3.png" class="math-inline" /> を一般化したものが、**n-組**（n-tuple）です。

- 3-組：<img src="tmp/299891b9d6cf90735a96b619a9dc7e20.png" class="math-inline" />  
- 一般の n-組：<img src="tmp/8b4030f64e79007ed224ed584aac284c.png" class="math-inline" />

これも順序が重要で、<img src="tmp/9f4a2be2cdfe01d726b90419abbbeca3.png" class="math-inline" /> となるのは、すべての <img src="tmp/62e13e7c1114087e103f0e72f9db52d7.png" class="math-inline" /> について <img src="tmp/e38eb8786c3394a03bda09cf37410414.png" class="math-inline" /> のときに限ります。

n-組も、順序対を入れ子にすることで集合として定義できます（例：<img src="tmp/a7d2ecbb0142f2b6a1f5193163c0ccf4.png" class="math-inline" /> など）。

### 5. 直積との関係

集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> の**直積** <img src="tmp/9aa33bf4c800bdd83164f41c1c878b1b.png" class="math-inline" /> は、「A の元 a と B の元 b の順序対 <img src="tmp/b01473ed500218be6ffa478d88033ca3.png" class="math-inline" /> 全体の集合」です。



<div class="math-display-container"><img src="tmp/bde02e731740a4040b874e69bab9172b.png" class="math-display" /></div>



つまり、順序対という概念があるからこそ、直積が定義できます。



## 直積

直積（デカルト積）の数学的な定義は次の通りです。

### 1. 2つの集合の直積

__定義__

2つの集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> に対して、「**A の元 a と B の元 b の組 (a, b) 全体の集合**」を、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の**直積**（デカルト積）といいます。

- 記号：<img src="tmp/9aa33bf4c800bdd83164f41c1c878b1b.png" class="math-inline" />
- 定義式：
  

<div class="math-display-container"><img src="tmp/bde02e731740a4040b874e69bab9172b.png" class="math-display" /></div>



ここで <img src="tmp/b01473ed500218be6ffa478d88033ca3.png" class="math-inline" /> は**順序対**（ordered pair）であり、 <img src="tmp/632eba6ac109cc43a246f39054e9b527.png" class="math-inline" /> となるのは <img src="tmp/fabe0eb49e78c56a07f4ffcc7f12d43b.png" class="math-inline" /> かつ <img src="tmp/9ac6973a9eaa61d82c1e9bd5173396ba.png" class="math-inline" /> のときに限ります。

### 2. n 個の集合の直積

__定義__

<img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> 個の集合 <img src="tmp/5faeb824a1be449c7e490bf469d9f9b5.png" class="math-inline" /> に対して、「各 <img src="tmp/b4cf280200862773bb25975b26b9248c.png" class="math-inline" /> から1つずつ元を取って並べた <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" />-組全体の集合」を、これらの集合の**直積**といいます。

- 記号：<img src="tmp/4a976bdf8567058a4dea3f9caffe1459.png" class="math-inline" /> または <img src="tmp/c984e3159fc34747a82ff0f1082c79c7.png" class="math-inline" />
- 定義式：
  

<div class="math-display-container"><img src="tmp/0a98222b5105b24654c017457ee4299c.png" class="math-display" /></div>



ここでも <img src="tmp/71ee7b66abe95c52ad7f7bec02c2cb99.png" class="math-inline" /> は順序付きの組であり、対応する成分がすべて等しいときのみ等しいとみなします。

### 3. 一般の集合族の直積

__定義__

集合族 <img src="tmp/927219c1b307f7f95d2fc635d02c6635.png" class="math-inline" />（添字集合 <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> と各 <img src="tmp/9986a8a5f7309c550d762fa0a9c779d4.png" class="math-inline" /> に対する集合 <img src="tmp/b4cf280200862773bb25975b26b9248c.png" class="math-inline" />）に対して、「各 <img src="tmp/9986a8a5f7309c550d762fa0a9c779d4.png" class="math-inline" /> ごとに <img src="tmp/b4cf280200862773bb25975b26b9248c.png" class="math-inline" /> の元を1つ選ぶ『選択関数』全体の集合」を、この集合族の**直積**といいます。

- 記号：<img src="tmp/64db53a1be2ae7b1abd8c7a73d2c6303.png" class="math-inline" />
- 定義式：
  

<div class="math-display-container"><img src="tmp/8c59b1e97dbe77d4fc74bc1f9c56846a.png" class="math-display" /></div>



直感的には、
- 添字 <img src="tmp/62e13e7c1114087e103f0e72f9db52d7.png" class="math-inline" /> ごとに「どの元を選ぶか」を決める関数 <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> を1つとると、  
  それが直積の1つの元に対応します。
- 有限個の直積 <img src="tmp/71ee7b66abe95c52ad7f7bec02c2cb99.png" class="math-inline" /> は、  
  <img src="tmp/a9be588af50ef980d2c50b061ef82172.png" class="math-inline" /> という関数と同一視できます。

### 4. 例

__例1：2次元座標平面__



<div class="math-display-container"><img src="tmp/33fdbfc2a167c98554f6c30f022d674a.png" class="math-display" /></div>


は、実数直線 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の2つの直積であり、  
通常の2次元座標平面と同一視されます。

__例2：3次元空間__



<div class="math-display-container"><img src="tmp/39ce9c66d8983ac8cd36de8e63f18f4b.png" class="math-display" /></div>



__例3：有限集合の直積__



<div class="math-display-container"><img src="tmp/d7db207bb47386e829ef6d70744519ec.png" class="math-display" /></div>


のとき、


<div class="math-display-container"><img src="tmp/398ae78ac81011cbeb93ede27f878832.png" class="math-display" /></div>



__例4：無限直積（関数空間として）__

添字集合 <img src="tmp/c770126685581d1b2c7cd93e05eea76f.png" class="math-inline" />、各 <img src="tmp/7dfbaea7e287db391efbb55fedf6b536.png" class="math-inline" /> とすると、


<div class="math-display-container"><img src="tmp/eeefb5941d11f4780c5cf57c4aaa2ac0.png" class="math-display" /></div>


は、「各自然数に対して 0 か 1 を割り当てる関数全体の集合」であり、  
これは**二進無限列の空間**とみなせます。

### 5. 直積の性質（ざっくり）

- 元の個数（有限集合の場合）：  
  <img src="tmp/280cfcccc3eb1564bb93266b279c9fba.png" class="math-inline" />、  
  一般に <img src="tmp/a0caca43e025ff7396d3bad70349081c.png" class="math-inline" />。
- 直積は**非可換**：一般に <img src="tmp/76a43d4ecadf74e27241088342dd484c.png" class="math-inline" />（ただし、順序対の順序を変える自然な全単射はあります）。
- 空集合との直積：  
  <img src="tmp/ee0ca403e6a5aab2dfbbd679242c0cd1.png" class="math-inline" />。

## 関係

数学における**関係**（relation）とは、ざっくり言うと「2つ（以上）の対象の間に成り立つ『関係性』を表すもの」です。

### 1. 関係の数学的な定義

__2項関係（最も基本的なもの）__

2つの集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> に対して、「A の元と B の元の間の関係」を、**直積 <img src="tmp/9aa33bf4c800bdd83164f41c1c878b1b.png" class="math-inline" /> の部分集合**として定義します。

- 形式的には：  
  集合 <img src="tmp/07ae8e6f6d946bbb34df737e94cce6d6.png" class="math-inline" /> を、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への**2項関係**といいます。
- 記号：  
  <img src="tmp/aa0b4e95b310fd021c874720d3c0e076.png" class="math-inline" /> が「関係 <img src="tmp/a0f7a58538051aad071147a20c796777.png" class="math-inline" /> にある」ことを  
  

<div class="math-display-container"><img src="tmp/67d39cede877d7f6bb84139a27f897ca.png" class="math-display" /></div>


  と書きます。

例：
- <img src="tmp/f658029dff54f97b2b3bedcc023c8a7b.png" class="math-inline" />（自然数）とし、<img src="tmp/e3477f680098a5c2c72f30e92c52a040.png" class="math-inline" /> とすると、 <img src="tmp/93f14fbe6cc1fde179f14a952593f671.png" class="math-inline" /> は「<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> は <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> より小さい」という関係を表します。

__同じ集合上の関係__

特に <img src="tmp/aaea37894ac3add2b714b6ff50ad2d60.png" class="math-inline" /> のとき、<img src="tmp/77f1b8b484bf0e9627dd0388a5753dee.png" class="math-inline" /> を「<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> 上の関係」といいます。

- 例：  
  <img src="tmp/8be68fc2a1c3bf559f55ec9d1027a5b0.png" class="math-inline" />（整数）とし、 <img src="tmp/79105b08b0a060072de7f3696345e49e.png" class="math-inline" /> は「偶奇が同じ」という関係です。

### 2. 関係の例

__数の関係__
- <img src="tmp/88bb41a50c3c079093e8b1f4b7e28a7d.png" class="math-inline" />（等しい）
- <img src="tmp/8e2ef8e3d3641c6b07743ff72f5835f4.png" class="math-inline" />（より小さい）
- <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" />（以下）
- <img src="tmp/710249df7ba4bebc84364a10b6e0ebe6.png" class="math-inline" />（割り切る：<img src="tmp/d6ffc2e34b97cc1509182a1aac583041.png" class="math-inline" /> は「m は n の約数」）

__集合の関係__
- <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" />（部分集合）
- <img src="tmp/703433fde77b69514cee13f320ecf601.png" class="math-inline" />（属する）

__日常的な関係__
- 「〜は〜の親である」
- 「〜は〜と同じクラスである」
- 「〜は〜より背が高い」

これらを数学的に扱うときは、対象の集合を決めて、  
その直積の部分集合として関係を定義します。

### 3. 関係の種類（重要な2つ）

__(1) 同値関係（equivalence relation）__

集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> 上の関係 <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> が次の3つを満たすとき、**同値関係**といいます。

1. **反射律**：任意の <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> について <img src="tmp/8a7e207de9b21e03f15828b5c1f0c59e.png" class="math-inline" />
2. **対称律**：<img src="tmp/2a8f50a3a7c1572afac59671da0586cc.png" class="math-inline" />
3. **推移律**：<img src="tmp/282a27126242cbc3ed0d34207676fd95.png" class="math-inline" /> かつ <img src="tmp/424315dad8a955f10ef966a7f9b12c74.png" class="math-inline" />

例：
- 「偶奇が同じ」（整数の集合上）
- 「同じ誕生日」（人の集合上）
- 「合同 modulo n」（整数の集合上：<img src="tmp/cb28250da1f631f8f6673c1625538ad5.png" class="math-inline" />）

同値関係があると、集合を「同値類」というグループに分けることができます。

__(2) 順序関係（order relation）__

集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> 上の関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> が次の3つを満たすとき、**半順序**（partial order）といいます。

1. **反射律**：<img src="tmp/91e916783df380da3731593c15075618.png" class="math-inline" />
2. **反対称律**：<img src="tmp/f5645db615a2af23d8c0eb8fbfb5bdeb.png" class="math-inline" /> かつ <img src="tmp/a94a4dcf6e1ebff8d159ac378ad22a76.png" class="math-inline" />
3. **推移律**：<img src="tmp/f5645db615a2af23d8c0eb8fbfb5bdeb.png" class="math-inline" /> かつ <img src="tmp/fc36cff11e63b66a6b7c8220050e04fc.png" class="math-inline" />

さらに、任意の <img src="tmp/6990a029c9f62b248fafa1caabeaf714.png" class="math-inline" /> について <img src="tmp/f5645db615a2af23d8c0eb8fbfb5bdeb.png" class="math-inline" /> または <img src="tmp/8ae95cd25057de5330bfaeaa535a5fe0.png" class="math-inline" /> が成り立つとき、  
**全順序**（total order）といいます。

例：
- 実数上の <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" />（大小関係）は全順序。
- 集合の包含関係 <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" /> は半順序（一般には全順序ではない）。

### 4. 関係の一般化（n項関係）

2項関係を一般化して、**n項関係**も定義できます。

- 集合 <img src="tmp/5faeb824a1be449c7e490bf469d9f9b5.png" class="math-inline" /> に対して、  
  <img src="tmp/cf8ece5d69860be26c16d7da2e37e112.png" class="math-inline" /> を n項関係といいます。

例：
- 「(x, y, z) は <img src="tmp/f22b1c141081bc393afed621c1de63a9.png" class="math-inline" /> を満たす」という関係は、  
  <img src="tmp/e21112601e6709342152bf3649179f3f.png" class="math-inline" /> の部分集合として定義される3項関係です。

## 写像 or 関数

写像（関数）について、数学的な定義と基本的な性質を説明します。

### 1. 写像（関数）の定義

__直感的な説明__

写像（map）または関数（function）とは、  
「ある集合の各元に対して、別の集合の元を**1つずつ**対応させる規則」のことです。

__数学的な定義__

2つの集合 <img src="tmp/3af4003804eded1b617e65dd187c4281.png" class="math-inline" /> に対して、  
次の条件を満たす**関係** <img src="tmp/79b01f8358ed2a92df902e0d870e6870.png" class="math-inline" /> を、  
<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から <img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> への**写像**といいます。

1. **全域性**（定義域の各元に対応が存在）  
   任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して、ある <img src="tmp/624d9c5f82f105740d90527a22154fa6.png" class="math-inline" /> が存在して <img src="tmp/50c2259a883f0058476e9b11b0c17d75.png" class="math-inline" />。
2. **一意性**（1つの入力に対し出力は1つ）  
   <img src="tmp/54c6b56a018621f4072d851eefaee86f.png" class="math-inline" /> かつ <img src="tmp/eb2fc6098de89eaffa8b6bd0fcfe6c63.png" class="math-inline" /> ならば <img src="tmp/d730147e44b7cc005325b3c6632a793a.png" class="math-inline" />。

このとき、  
- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> を**定義域**（domain）  
- <img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> を**終域**（codomain）  
といいます。

__記号__

- <img src="tmp/1309395d6ea8cb108f435d79f038359d.png" class="math-inline" />：<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から <img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> への写像
- <img src="tmp/ad5636b84e80af2c933cc5080d40a667.png" class="math-inline" />：<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に対応する <img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> の元 <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" />（<img src="tmp/50c2259a883f0058476e9b11b0c17d75.png" class="math-inline" />）

### 2. 写像の例

__例1：実数関数__

- <img src="tmp/8b8008b2eeada5dd1b543b844126b260.png" class="math-inline" />  
  → 各実数 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に対して、その2乗 <img src="tmp/db79bb72c5850b761f8aad4da65d9133.png" class="math-inline" /> を対応させる写像。

__例2：離散的な写像__

- <img src="tmp/3c62407c971abfad3f552ad7a4c3cdf6.png" class="math-inline" /> とし、  
  <img src="tmp/07ac5cef2d6506f2a0968e425fcaf31c.png" class="math-inline" /> と定める。  
  これは <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から <img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> への写像です。

__例3：定数写像__

- <img src="tmp/1309395d6ea8cb108f435d79f038359d.png" class="math-inline" /> が、すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して <img src="tmp/31d296d578b61d40c016cf911538febb.png" class="math-inline" />（固定された <img src="tmp/1d30d3c64762d4ad79d689cf1a272cbe.png" class="math-inline" />）となる写像。

__例4：包含写像__

- <img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> のとき、  
  <img src="tmp/118b7a7a0c8fea6126d77b1dccc07158.png" class="math-inline" /> は包含写像（恒等写像の制限）です。

### 3. 像（image）と逆像（inverse image）

__像__

写像 <img src="tmp/1309395d6ea8cb108f435d79f038359d.png" class="math-inline" /> と部分集合 <img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> に対して、  


<div class="math-display-container"><img src="tmp/a309288e2ebde003852d4805d5d9a10f.png" class="math-display" /></div>


を <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**像**（image）といいます。

- 特に、<img src="tmp/8bf2f372a47d22e5724adf03deac90ad.png" class="math-inline" /> を <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> の**値域**（range）といいます。

__逆像__

部分集合 <img src="tmp/138df74c459536e2527efbcbc3ccc4a2.png" class="math-inline" /> に対して、  


<div class="math-display-container"><img src="tmp/fe97d124b2b1600e24fec1bfd0deef0d.png" class="math-display" /></div>


を <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の**逆像**（inverse image）といいます。

- 逆像は、**f が全単射でなくても定義できます**（「逆写像」とは別物です）。

### 4. 単射・全射・全単射

写像 <img src="tmp/1309395d6ea8cb108f435d79f038359d.png" class="math-inline" /> について：

__単射（injective）__

- 定義：<img src="tmp/4f788ca24507211da3530282706cecca.png" class="math-inline" />  
  （異なる入力からは異なる出力が出る）
- 別表現：<img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> は「1対1」の対応。

__全射（surjective）__

- 定義：<img src="tmp/0fb9bd11b55f2b5243cd02ba773582b3.png" class="math-inline" />  
  （終域のどの元も、何らかの入力の像になっている）
- 別表現：<img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> は「上への（onto）」写像。

__全単射（bijective）__

- 定義：単射かつ全射。
- このとき、**逆写像** <img src="tmp/1d0cecfb871df02cbc954eacd82fd0d2.png" class="math-inline" /> が存在し、  
  <img src="tmp/64eb6df7bd0519909fb2100c3edc0921.png" class="math-inline" /> となります。

### 5. 合成写像

2つの写像 <img src="tmp/a50c30e9891bb0b361aa3bb50ef1d359.png" class="math-inline" /> に対して、  
その**合成写像** <img src="tmp/f95378f108221b8da2e38b96cd99f79f.png" class="math-inline" /> を


<div class="math-display-container"><img src="tmp/24e22931bd2f63b297040054a6406bd3.png" class="math-display" /></div>


で定義します。

- 合成は結合律 <img src="tmp/4427715db7c6a1704dbe364f40f79010.png" class="math-inline" /> を満たします。

### 6. 恒等写像

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> に対して、  


<div class="math-display-container"><img src="tmp/bb86f7e18cc0c71336b2c3b23b2a0324.png" class="math-display" /></div>


を**恒等写像**といいます。

- 任意の <img src="tmp/1309395d6ea8cb108f435d79f038359d.png" class="math-inline" /> について、  
  <img src="tmp/80c2f706ddf657d37b1d25adfca6082a.png" class="math-inline" />、<img src="tmp/c56ae043132d29daf2137361d0a63b87.png" class="math-inline" /> が成り立ちます。

### 7. 写像と関数の言葉の使い分け（補足）

- 数学では、**写像**と**関数**はほぼ同義として扱われることが多いです。
- ただし、値域が数（<img src="tmp/714f3cac59f4322d9cea069ca14139cd.png" class="math-inline" /> など）である場合に「関数」と呼ぶことが多いです。
- 英語では function, map, mapping などが使われます。

## 演習

### 問題

これまでの内容（べき集合・集合族・順序対・直積・関係・写像）に関する問題を出題します。  
証明が必要な問題を中心に、基礎から少し応用まで用意しました。

__問題1（べき集合の基本）__

集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が有限集合で <img src="tmp/0d25d178b7971239fb0d782eb9be6877.png" class="math-inline" /> のとき、  
べき集合 <img src="tmp/e1fa47c9075271d02cf8ead81516d053.png" class="math-inline" /> の元の個数が <img src="tmp/edf6abb9fb79e3c90b5730fe145ed4d8.png" class="math-inline" /> であることを証明せよ。

__問題2（べき集合と包含）__

集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/c1624919e7f11506b2560d516a73ecd1.png" class="math-display" /></div>



__問題3（集合族の和集合・共通部分）__

集合族 <img src="tmp/927219c1b307f7f95d2fc635d02c6635.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/07952f9d525b62011fc91b0d4e6aea18.png" class="math-display" /></div>



__問題4（順序対の性質）__

順序対 <img src="tmp/b01473ed500218be6ffa478d88033ca3.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/922071adf11c6feb471e2d4af2a7b509.png" class="math-display" /></div>


（クーラトフスキーの定義 <img src="tmp/aa0057ba2566b9a1609349e2d0a39653.png" class="math-inline" /> を用いてもよい。）

__問題5（直積の元の個数）__

有限集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/da20b10fdbb599c9cfbfc6662f322c2b.png" class="math-display" /></div>



__問題6（直積と空集合）__

集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/9aeb4c9d4d57f00d72b78dbd561aa6d3.png" class="math-display" /></div>



__問題7（関係の定義）__

集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> 上の関係 <img src="tmp/a0f7a58538051aad071147a20c796777.png" class="math-inline" /> を、<img src="tmp/77f1b8b484bf0e9627dd0388a5753dee.png" class="math-inline" /> として定義する。  
このとき、任意の <img src="tmp/6990a029c9f62b248fafa1caabeaf714.png" class="math-inline" /> について


<div class="math-display-container"><img src="tmp/45236b034b89e6624fcccade9ba3776e.png" class="math-display" /></div>


であることを、関係の定義に基づいて説明せよ。

__問題8（同値関係の定義）__

集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> 上の関係 <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> が同値関係であるとは、次の3条件を満たすことである：

1. 反射律：<img src="tmp/4605c4baf1f452cd3b7d28985242fa9a.png" class="math-inline" />
2. 対称律：<img src="tmp/2a8f50a3a7c1572afac59671da0586cc.png" class="math-inline" />
3. 推移律：<img src="tmp/0e10a8b34848e90a3459de621b5cce9f.png" class="math-inline" />

このとき、整数全体 <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> 上の関係


<div class="math-display-container"><img src="tmp/c7e2b6dc0f826443b8f8b5dedc7328c2.png" class="math-display" /></div>


が同値関係であることを証明せよ。

__問題9（写像の定義）__

写像 <img src="tmp/1309395d6ea8cb108f435d79f038359d.png" class="math-inline" /> を、関係 <img src="tmp/79b01f8358ed2a92df902e0d870e6870.png" class="math-inline" /> で

- 全域性：<img src="tmp/659d4107c884457c0cc748ed90e82ac1.png" class="math-inline" />
- 一意性：<img src="tmp/aceb532f446eceeff206bee56a3ebba1.png" class="math-inline" />

を満たすものとして定義する。  
このとき、任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して <img src="tmp/3fdae9fef92593e4e976f831339ce3a1.png" class="math-inline" /> が一意に定まることを示せ。

__問題10（単射・全射の判定）__

写像 <img src="tmp/0ecb74406b266dffcf367daa0f4a3185.png" class="math-inline" /> を <img src="tmp/aabef1d7124d7685af6b78e536ac5ec1.png" class="math-inline" /> で定義する。

1. <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> が単射であることを証明せよ。
2. <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> が全射であるかどうかを判定し、理由を述べよ。

__問題11（像と逆像の基本性質）__

写像 <img src="tmp/1309395d6ea8cb108f435d79f038359d.png" class="math-inline" /> と部分集合 <img src="tmp/f7e3eb75dcae83d946b6b460d9880e30.png" class="math-inline" />、<img src="tmp/2e786cb2e07287cbf8503e699629ae28.png" class="math-inline" /> について、次を示せ。

1. <img src="tmp/120a042b6ccbd0e7f3aeeba722c96754.png" class="math-inline" />
2. <img src="tmp/2e362c66b6ea76729d9e4fd5b1cd465d.png" class="math-inline" />

__問題12（合成写像の結合律）__

写像 <img src="tmp/6198a62b70361f3039f056ac458f1b64.png" class="math-inline" /> について、次を示せ。


<div class="math-display-container"><img src="tmp/53a928740ef4eb31e04d3abd95606866.png" class="math-display" /></div>



__問題13（べき集合と写像の関係）__

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> のべき集合を <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> とする。  
写像 <img src="tmp/1309395d6ea8cb108f435d79f038359d.png" class="math-inline" /> に対して、写像


<div class="math-display-container"><img src="tmp/54bc673194561794eed28dfc298f9af1.png" class="math-display" /></div>


を考える。

1. <img src="tmp/b73d7afeafc2e2c492925aca330e616c.png" class="math-inline" /> が well-defined である（すなわち、<img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> ならば <img src="tmp/71dcb5db5f5959443a1074f156abb937.png" class="math-inline" />）ことを示せ。
2. <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> が全射ならば <img src="tmp/b73d7afeafc2e2c492925aca330e616c.png" class="math-inline" /> も全射であることを示せ。

__問題14（直積と写像）__

集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> の直積 <img src="tmp/9aa33bf4c800bdd83164f41c1c878b1b.png" class="math-inline" /> と、写像


<div class="math-display-container"><img src="tmp/7546d3db13db7fc96eb0392dfc91a2f9.png" class="math-display" /></div>




<div class="math-display-container"><img src="tmp/51e40d1a142a8e1e89264cbcad9d0aa0.png" class="math-display" /></div>


（射影写像）を考える。

1. <img src="tmp/99dd984be5d1898ca66722174372b44c.png" class="math-inline" /> が全射であることを示せ。
2. 一般に <img src="tmp/99dd984be5d1898ca66722174372b44c.png" class="math-inline" /> は単射か？ 理由とともに答えよ。

__問題15（関係と写像）__

集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> 上の関係 <img src="tmp/77f1b8b484bf0e9627dd0388a5753dee.png" class="math-inline" /> が、ある写像 <img src="tmp/be4691334b8f1e59d5dd85f6c93651cb.png" class="math-inline" /> を用いて


<div class="math-display-container"><img src="tmp/0dbdc77d1a1882769657b1462431cd6b.png" class="math-display" /></div>


と書けるとする。このとき、<img src="tmp/a0f7a58538051aad071147a20c796777.png" class="math-inline" /> はどのような性質を持つか？  
（反射律・対称律・推移律の観点から考察せよ。）

### 解答

各問題に対する解答を順に示します。

__問題1（べき集合の基本）__

**主張**：有限集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> で <img src="tmp/0d25d178b7971239fb0d782eb9be6877.png" class="math-inline" /> のとき、<img src="tmp/6839850c66f38557815aa335487d6459.png" class="math-inline" />。

**証明**  
A の元を <img src="tmp/c51c203b7af94931654ccc85877101f4.png" class="math-inline" /> とする。  
A の部分集合 X は、各 <img src="tmp/dbe99c1297a68c6ec133c6551d6d847e.png" class="math-inline" /> について「含めるか・含めないか」の2通りで決まる。  
したがって、部分集合の総数は


<div class="math-display-container"><img src="tmp/3ba2a84d602de4eabea29f6bab0f9ba5.png" class="math-display" /></div>


である。よって <img src="tmp/6839850c66f38557815aa335487d6459.png" class="math-inline" />。□

__問題2（べき集合と包含）__

**主張**：<img src="tmp/02e7d967bd229717f3e6e20fb56591b9.png" class="math-inline" />。

**証明**  
<img src="tmp/d3cd613dbef8bc0f1be0fb2d340867e0.png" class="math-inline" /> を仮定する。  
任意の <img src="tmp/46cef2f81058f5a73c4103b8a5da6ed7.png" class="math-inline" /> をとると、<img src="tmp/62257db6fcfcc679dfd1432de9e1525d.png" class="math-inline" />。  
仮定より <img src="tmp/d3cd613dbef8bc0f1be0fb2d340867e0.png" class="math-inline" /> なので、<img src="tmp/4ca48da51d028387d91b80b37d9ebd28.png" class="math-inline" />。  
したがって <img src="tmp/4699a6a8350d5a02ca41551c7f1edc9b.png" class="math-inline" />。  
よって <img src="tmp/a178caa5e649bf4c058e2a8d934fed00.png" class="math-inline" />。□

__問題3（集合族の和集合・共通部分）__

**主張**：


<div class="math-display-container"><img src="tmp/6e4fb2b05837fd53ebf1b9721b95d33f.png" class="math-display" /></div>



**証明**（定義に基づく確認）

- 和集合の定義：
  

<div class="math-display-container"><img src="tmp/b85f02fa01ad1dd56a71e8c29d1096bc.png" class="math-display" /></div>


  これは右辺の条件そのものである。

- 共通部分の定義：
  

<div class="math-display-container"><img src="tmp/4c3128df47add219bc021457ef8ada15.png" class="math-display" /></div>


  これも右辺の条件そのものである。

したがって、等式は定義から直ちに成り立つ。□

__問題4（順序対の性質）__

**主張**：<img src="tmp/c90d5887eb67307d9244735865cd180c.png" class="math-inline" />。

**証明**（クーラトフスキーの定義 <img src="tmp/aa0057ba2566b9a1609349e2d0a39653.png" class="math-inline" /> を用いる）

- （⇒）<img src="tmp/aadab566f25f37e42fcf3bb5a058d41c.png" class="math-inline" /> とする。  
  このとき <img src="tmp/abe21b36d5473ac8ee5de1cda97f4c1f.png" class="math-inline" />。  
  集合が等しいので、  
  - <img src="tmp/6c1658f93e555b5876a1c0c94f8d11db.png" class="math-inline" /> または <img src="tmp/c6050da2b102ad99424efb75564d7fef.png" class="math-inline" />  
  - <img src="tmp/f7425e3aefb6b10de9c72e5f1e5d0558.png" class="math-inline" /> または <img src="tmp/6ff86d0ab29104b31052c0e20eec0ab7.png" class="math-inline" />  
  の組み合わせを考える。

  実際には、<img src="tmp/c133a996a2b92daa0fdfb69c4f17cd7a.png" class="math-inline" /> は1元集合、<img src="tmp/206dbbea548bc8db84d128a79a13834f.png" class="math-inline" /> は1元または2元集合なので、  
  整合性から <img src="tmp/6c1658f93e555b5876a1c0c94f8d11db.png" class="math-inline" /> かつ <img src="tmp/6ff86d0ab29104b31052c0e20eec0ab7.png" class="math-inline" /> となる。  
  よって <img src="tmp/fabe0eb49e78c56a07f4ffcc7f12d43b.png" class="math-inline" /> かつ <img src="tmp/8e3ffb3e821f55673384debf6c43e55e.png" class="math-inline" /> より <img src="tmp/9ac6973a9eaa61d82c1e9bd5173396ba.png" class="math-inline" />。

- （⇐）<img src="tmp/fabe0eb49e78c56a07f4ffcc7f12d43b.png" class="math-inline" /> かつ <img src="tmp/9ac6973a9eaa61d82c1e9bd5173396ba.png" class="math-inline" /> ならば、  
  <img src="tmp/c0aa4af247a01aa4a66c54bc2713f329.png" class="math-inline" />。

以上より主張が成り立つ。□

__問題5（直積の元の個数）__

**主張**：有限集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> について <img src="tmp/280cfcccc3eb1564bb93266b279c9fba.png" class="math-inline" />。

**証明**  
<img src="tmp/f6771a6911dcd59cabd920ab453bab84.png" class="math-inline" /> とする。  
直積の定義より


<div class="math-display-container"><img src="tmp/252c65b825e94ec48b21d1a42db5caef.png" class="math-display" /></div>


であり、<img src="tmp/091bc620661f7aa3321cb5ef25a0c479.png" class="math-inline" /> の組は <img src="tmp/fb6061356d995b6268fdca751c7e4f5c.png" class="math-inline" /> 通りある。  
順序対は <img src="tmp/14168b04464298f0ccedb6abe0d92c25.png" class="math-inline" /> なので、  
これらはすべて異なる。  
したがって <img src="tmp/e218c27cc1fc111278041f5f544b3379.png" class="math-inline" />。□

__問題6（直積と空集合）__

**主張**：<img src="tmp/37e3d266a2aa9a4ff3e7796d6b25a791.png" class="math-inline" />。

**証明**  
<img src="tmp/afe4b891bae76a448b347b307d33f92d.png" class="math-inline" /> とすると、定義より <img src="tmp/91aa48b65b7d642d9603ae0cdd3cc302.png" class="math-inline" /> かつ <img src="tmp/9f8fb204e75ba3eca2341d18d319956b.png" class="math-inline" />。  
しかし <img src="tmp/9f8fb204e75ba3eca2341d18d319956b.png" class="math-inline" /> は偽なので、<img src="tmp/afe4b891bae76a448b347b307d33f92d.png" class="math-inline" /> を満たす元は存在しない。  
したがって <img src="tmp/37e3d266a2aa9a4ff3e7796d6b25a791.png" class="math-inline" />。□

__問題7（関係の定義）__

**主張**：関係 <img src="tmp/77f1b8b484bf0e9627dd0388a5753dee.png" class="math-inline" /> に対し、<img src="tmp/af3ca19a19f2ec96a900e47ec414ab57.png" class="math-inline" />。

**説明**  
関係 <img src="tmp/a0f7a58538051aad071147a20c796777.png" class="math-inline" /> は「A の元の間の関係」を表すもので、  
数学的には「どの順序対が関係にあるか」を指定する集合として定義される。  
したがって、「a と b が関係 R にある」という言明は、  
「順序対 (a,b) が R に属する」ことと同値である。  
これが <img src="tmp/af3ca19a19f2ec96a900e47ec414ab57.png" class="math-inline" /> の意味である。□

__問題8（同値関係の定義）__

**主張**：<img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> 上の関係 <img src="tmp/e733257952e7aa275c7718eb4de5b5ac.png" class="math-inline" /> は偶数、は同値関係。

**証明**  
反射律・対称律・推移律を確認する。

- 反射律：任意の <img src="tmp/ee79df5c22ccdb1aac5069c6608d8bf1.png" class="math-inline" /> について <img src="tmp/c8047c975f625fec7d891dcc36b24b29.png" class="math-inline" /> は偶数なので <img src="tmp/9d922ae9f571874a7a84318705886893.png" class="math-inline" />。
- 対称律：<img src="tmp/fd43c40ff8a2aa5089e6f675c4358eff.png" class="math-inline" /> ならば <img src="tmp/1b9d47d8f9d4f1ac3edc1fa9eb40c04a.png" class="math-inline" /> は偶数。  
  このとき <img src="tmp/fa56cfd898d2384cb5e53b64eb8eae76.png" class="math-inline" /> も偶数なので <img src="tmp/0db5f0839bdc66b649cba6117fd60552.png" class="math-inline" />。
- 推移律：<img src="tmp/fd43c40ff8a2aa5089e6f675c4358eff.png" class="math-inline" /> かつ <img src="tmp/10fe0848907e6fccd2897a73229357f9.png" class="math-inline" /> ならば、  
  <img src="tmp/c977b06d26dc679d50e156f83c4cafef.png" class="math-inline" /> は偶数。  
  よって <img src="tmp/2ee102f0b509aa06476f127ed9fb5d6a.png" class="math-inline" /> も偶数なので <img src="tmp/ee806c69c3345963f055c90c7e428df6.png" class="math-inline" />。

以上より <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> は同値関係である。□

__問題9（写像の定義）__

**主張**：写像 <img src="tmp/79b01f8358ed2a92df902e0d870e6870.png" class="math-inline" /> に対し、任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について <img src="tmp/3fdae9fef92593e4e976f831339ce3a1.png" class="math-inline" /> が一意に定まる。

**証明**  
写像の定義より：

- 全域性：任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対し、ある <img src="tmp/624d9c5f82f105740d90527a22154fa6.png" class="math-inline" /> が存在して <img src="tmp/dd1040c860293a17fd6b218782c2b99e.png" class="math-inline" />。
- 一意性：<img src="tmp/8e55593cbc427963ae1da7c2fee46caa.png" class="math-inline" /> かつ <img src="tmp/6273dd199acd88dbe99826f96baca7bc.png" class="math-inline" /> ならば <img src="tmp/d730147e44b7cc005325b3c6632a793a.png" class="math-inline" />。

この2つより、各 <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して「<img src="tmp/dd1040c860293a17fd6b218782c2b99e.png" class="math-inline" /> となる <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" />」はちょうど1つ存在する。  
この唯一の <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> を <img src="tmp/3fdae9fef92593e4e976f831339ce3a1.png" class="math-inline" /> と定義する。  
したがって、任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対し <img src="tmp/3fdae9fef92593e4e976f831339ce3a1.png" class="math-inline" /> は一意に定まる。□

__問題10（単射・全射の判定）__

写像 <img src="tmp/28e8c94dbf2f80abae1d55131c44adb8.png" class="math-inline" /> について。

**1. 単射であることの証明**  
<img src="tmp/1ef74abf1825c0fab0c05717701a6f49.png" class="math-inline" /> とすると <img src="tmp/3a63b265a785673baf6e1535ad030651.png" class="math-inline" />。  
両辺を2で割って <img src="tmp/9c81f1664a8306ebf5167ca101ec0057.png" class="math-inline" />。  
よって <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> は単射。□

**2. 全射性の判定**  
<img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> は全射ではない。  
理由：例えば <img src="tmp/7223b5d73511ee58985c4bce02b667e1.png" class="math-inline" /> を考えると、<img src="tmp/bea3ff7e83f9f986ba38a5c4d93056c9.png" class="math-inline" /> となる整数 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は存在しない（<img src="tmp/5ae7a613e43435347b6fbb4c0fe603bc.png" class="math-inline" /> の整数解はない）。  
したがって、終域の元の中に像になっていないものがあるので、全射ではない。□

__問題11（像と逆像の基本性質）__

**1. 主張**：<img src="tmp/120a042b6ccbd0e7f3aeeba722c96754.png" class="math-inline" />。

**証明**  
任意の <img src="tmp/ec2ef8ba2e74a1765d1e620c595bab4a.png" class="math-inline" /> をとると、ある <img src="tmp/73fe4cd135f1bcb1a48184b22343989c.png" class="math-inline" /> が存在して <img src="tmp/f141e8674f7517ef30d06c8787a40a51.png" class="math-inline" />。  
<img src="tmp/7f0ec65989d2dde61ec52839d6e4f582.png" class="math-inline" /> より <img src="tmp/215e4c22c921254a6158f73d99574e57.png" class="math-inline" />。  
したがって <img src="tmp/e4624c8c74afff70db1e6786b3ce3091.png" class="math-inline" />。  
よって <img src="tmp/7732c9aeb8c6f59e78d26b901614ad54.png" class="math-inline" />。□

**2. 主張**：<img src="tmp/2e362c66b6ea76729d9e4fd5b1cd465d.png" class="math-inline" />。

**証明**  
任意の <img src="tmp/9bd16eb34738c9e7f9e89e9845888eb3.png" class="math-inline" /> をとると、<img src="tmp/1a1ca5c30848642389c09496a4458638.png" class="math-inline" />。  
<img src="tmp/8605ed3e6e54a5225a83b3cce0040b9a.png" class="math-inline" /> より <img src="tmp/b4b51d682638387c988e49406ccb2673.png" class="math-inline" />。  
したがって <img src="tmp/01a5dc32a2f56a9fe6a7d2a63acccf6a.png" class="math-inline" />。  
よって <img src="tmp/eec1720b4a8a24e62d97514170800c5b.png" class="math-inline" />。□

__問題12（合成写像の結合律）__

**主張**：<img src="tmp/4427715db7c6a1704dbe364f40f79010.png" class="math-inline" />。

**証明**  
任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、


<div class="math-display-container"><img src="tmp/badb0f5322c0719fff618991b5366e0e.png" class="math-display" /></div>


よって両者は等しい。  
写像が等しいとは、すべての入力に対して出力が等しいことなので、  
<img src="tmp/4427715db7c6a1704dbe364f40f79010.png" class="math-inline" /> が成り立つ。□

__問題13（べき集合と写像の関係）__

**1. 主張**：<img src="tmp/b73d7afeafc2e2c492925aca330e616c.png" class="math-inline" /> は well-defined（<img src="tmp/b4daee443e3f83e2c23a214aa267b5ce.png" class="math-inline" />）。

**証明**  
<img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> とする。  
任意の <img src="tmp/6d5971a29cbb9f1f63fb7fa35eddd6fd.png" class="math-inline" /> をとると、ある <img src="tmp/91aa48b65b7d642d9603ae0cdd3cc302.png" class="math-inline" /> が存在して <img src="tmp/5ebc769c60c5f00f40aad9ed6b78dafa.png" class="math-inline" />。  
<img src="tmp/1309395d6ea8cb108f435d79f038359d.png" class="math-inline" /> より <img src="tmp/c2b8927d954dfab809a7ab1adf8457fd.png" class="math-inline" />。  
したがって <img src="tmp/624d9c5f82f105740d90527a22154fa6.png" class="math-inline" />。  
よって <img src="tmp/71dcb5db5f5959443a1074f156abb937.png" class="math-inline" /> であり、<img src="tmp/b73d7afeafc2e2c492925aca330e616c.png" class="math-inline" /> は well-defined。□

**2. 主張**：<img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> が全射ならば <img src="tmp/b73d7afeafc2e2c492925aca330e616c.png" class="math-inline" /> も全射。

**証明**  
<img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> は全射なので、任意の <img src="tmp/624d9c5f82f105740d90527a22154fa6.png" class="math-inline" /> に対してある <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> が存在して <img src="tmp/3d733d7f5e6af7e9a093de56cc699da0.png" class="math-inline" />。  
特に、任意の <img src="tmp/d0bd8098773cf53de37079024c87f416.png" class="math-inline" /> をとると、  
<img src="tmp/508b3c9e2c476b74c71fed2ea3564542.png" class="math-inline" /> とおけば、  
<img src="tmp/815ac16c64b216bb2ca0ca47249c324e.png" class="math-inline" /> となる（∵ <img src="tmp/7c48a25b77bab09baf3332d1c48d1033.png" class="math-inline" />）。  
したがって <img src="tmp/b73d7afeafc2e2c492925aca330e616c.png" class="math-inline" /> は全射。□

__問題14（直積と写像）__

**1. 主張**：<img src="tmp/99dd984be5d1898ca66722174372b44c.png" class="math-inline" /> は全射。

**証明**（<img src="tmp/45fe16a465fe893aa6b4f28d72b1abd5.png" class="math-inline" /> について）  
任意の <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> をとる。  
ある <img src="tmp/1c7b9d06bbb9ff1ebb4d49f09efce87f.png" class="math-inline" /> を固定し（B が空でないと仮定）、<img src="tmp/07b814fce1ab0941a410e2d515b80b23.png" class="math-inline" /> とすると、  
<img src="tmp/9a5fc93ad2b5fb479612af02113fa8b6.png" class="math-inline" />。  
したがって任意の <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> は <img src="tmp/45fe16a465fe893aa6b4f28d72b1abd5.png" class="math-inline" /> の像である。  
よって <img src="tmp/45fe16a465fe893aa6b4f28d72b1abd5.png" class="math-inline" /> は全射。  
同様に <img src="tmp/fb423669fcbe969a28857409cdd5dd64.png" class="math-inline" /> も全射。□

**2. 主張**：一般に <img src="tmp/99dd984be5d1898ca66722174372b44c.png" class="math-inline" /> は単射ではない。

**理由**  
<img src="tmp/6f02951b12d2160a8993dc7f49f366d0.png" class="math-inline" /> とすると、  
例えば <img src="tmp/16e06b8dc533fa1fa688223b213bbc40.png" class="math-inline" />（ただし <img src="tmp/66b6797cf549c163995b169b5848a89d.png" class="math-inline" />）は異なる元だが、  
<img src="tmp/f56e97a353f9c619174d60735ac1dcc0.png" class="math-inline" /> となる。  
よって <img src="tmp/45fe16a465fe893aa6b4f28d72b1abd5.png" class="math-inline" /> は単射ではない。  
同様に <img src="tmp/fb423669fcbe969a28857409cdd5dd64.png" class="math-inline" /> も単射ではない。□

__問題15（関係と写像）__

**主張**：<img src="tmp/d3e43cfe42ad7922737b6d08aaa51b51.png" class="math-inline" /> のとき、R の性質を考察せよ。

**解答**

- **反射律**：一般には成り立たない。  
  反例：<img src="tmp/595f68a95b8f195a8238e57fadbc48bc.png" class="math-inline" /> とすると、  
  <img src="tmp/a829820357fe2214cc3298b118668691.png" class="math-inline" /> なので <img src="tmp/d48b963ef97944e723933ee912b81a63.png" class="math-inline" /> は偽。

- **対称律**：一般には成り立たない。  
  反例：上と同じ例で <img src="tmp/27d046de23fbb0c426577f9eb8bce39a.png" class="math-inline" /> だが <img src="tmp/8bc9199951f7a17bd536f303278b08c8.png" class="math-inline" /> も成り立つので、  
  この例では対称だが、一般には <img src="tmp/3d733d7f5e6af7e9a093de56cc699da0.png" class="math-inline" /> だからといって <img src="tmp/c8977f13a93a4336e2e46e2f26edefc5.png" class="math-inline" /> とは限らない。  
  例えば <img src="tmp/470530b256156440d6ff40703cb15e92.png" class="math-inline" />（A が整数の部分集合など）とすると対称でない。

- **推移律**：一般には成り立たない。  
  反例：<img src="tmp/9c02e1471e48cbeda9db1cf7b259a542.png" class="math-inline" /> とすると、  
  <img src="tmp/cb0b847646a93eefcf3ae6b33b2aa1e3.png" class="math-inline" /> だが <img src="tmp/84caf01f284d88fc5298f0ddd487570d.png" class="math-inline" />（∵ <img src="tmp/92b74071623da0b20628b85cf2ba690c.png" class="math-inline" />）。

**結論**：  
R は一般に反射律・対称律・推移律のいずれも満たさない。  
ただし、f が恒等写像なら R は相等関係（=）になり、同値関係となる。  
また、f が全単射でかつ対合（<img src="tmp/6a563e3a52ef88382ce588d338f499fc.png" class="math-inline" />）なら対称律が成り立つが、一般にはそうとは限らない。□



<div style="page-break-before:always"></div>


# 添数づけられた族と選択公理

## 添数づけられた族

「添数づけられた族（indexed family）」は、集合論・数学でよく使われる概念で、**集合の集まり（族）に、添数（インデックス）を付けて扱うための仕組み**です。

### 1. 直感的なイメージ

- 普通の「集合の族」は、単に「集合をいくつか集めたもの」です。
  - 例：{A, B, C} は3つの集合 A, B, C からなる族。
- 一方、「添数づけられた族」は、**各集合にラベル（添数）を付けて**、「どの集合がどの添数に対応するか」を明示したものです。
  - 例：A₁, A₂, A₃ のように、1, 2, 3 という添数で集合を区別する。

この「添数」は、自然数とは限らず、一般の集合（インデックス集合）から取れます。

### 2. 形式的な定義

__インデックス集合（添字集合）__

まず、**添数全体の集合**を用意します。これを**インデックス集合（index set）** と呼び、通常 I などで表します。

- 例：I = {1, 2, 3} や I = ℕ（自然数全体）など。

__添数づけられた族の定義__

**インデックス集合 I で添数づけられた集合の族**とは、  
「I の各元 i に対して、ある集合 Aᵢ を対応させる写像」のことです。

より正確には：

- ある集合 X を全体集合とする。
- インデックス集合 I がある。
- 写像  
  

<div class="math-display-container"><img src="tmp/95a3083e003ab6ab67966dd22b54b496.png" class="math-display" /></div>

  
  が与えられる（ここで <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> は X のべき集合）。
- このとき、各 i ∈ I に対して  
  

<div class="math-display-container"><img src="tmp/9fd01b1c18d6c30c34e12b48efa62753.png" class="math-display" /></div>

  
  とおくと、集合の集まり  
  

<div class="math-display-container"><img src="tmp/5dca0890f6786f390818190ba1d3b6c6.png" class="math-display" /></div>

  
  を **I で添数づけられた集合の族**と呼ぶ。

記号としては、

- <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" />
- <img src="tmp/927219c1b307f7f95d2fc635d02c6635.png" class="math-inline" />

などと書きます。

### 3. 具体例

__例1：有限インデックス集合__

- I = {1, 2, 3}
- A₁ = {1, 2}, A₂ = {3, 4}, A₃ = {5, 6}

このとき、<img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> は、添数 1, 2, 3 でラベルされた3つの集合の族です。

__例2：自然数を添数とする族__

- I = ℕ（自然数全体）
- Aₙ = {n, n+1} と定義する。

このとき、<img src="tmp/f61df9fb8bb5aea5d7d5503ae37a346e.png" class="math-inline" /> は A₁ = {1, 2}, A₂ = {2, 3}, A₃ = {3, 4}, … という無限個の集合の族です。

__例3：実数を添数に使う__

- I = ℝ（実数全体）
- 各 r ∈ ℝ に対して、Aᵣ = {x ∈ ℝ | x ≥ r} と定義する。

このとき、<img src="tmp/b528c2e01a2eea3e9310c18c602a0299.png" class="math-inline" /> は、実数 r を添数とする「半直線」の族です。

### 4. 添数づけられた族を使うメリット

1. **同じ集合が複数回現れても区別できる**  
   - 普通の集合の族では、同じ集合が2回現れても1つとみなされますが、  
     添数づけられた族では、**添数が違えば別の元**とみなせます。
   - 例：A₁ = {1, 2}, A₂ = {1, 2} でも、添数が違うので区別されます。

2. **無限族を扱いやすい**  
   - 自然数 ℕ や実数 ℝ など、無限集合をインデックス集合にすることで、無限個の集合をシステマティックに扱えます。

3. **和集合・共通部分・直積などを定義しやすい**  
   - 添数づけられた族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> に対して、
     - 和集合：<img src="tmp/a9b05ef5bc1b66fc3f36d59f169af84a.png" class="math-inline" />
     - 共通部分：<img src="tmp/097aaad7bf0f2209a97db40fc180edd4.png" class="math-inline" />
     - 直積：<img src="tmp/dfb0684c0df64178490392db6ad4b469.png" class="math-inline" />  
     などが自然に定義できます。

## 集合族の和

「集合族の和」には、大きく分けて2つの意味があります。

1. **有限個の集合の和（普通の和集合）**
2. **一般の集合族（添数づけられた族を含む）の和**

順に説明します。

### 1. 有限個の集合の和（普通の和集合）

__定義（2つの集合の場合）__

2つの集合 A, B に対して、**和集合（union）** <img src="tmp/2a2d668fa15d8d000993fbeec5fc1f82.png" class="math-inline" /> は、



<div class="math-display-container"><img src="tmp/14c3d9580f61b40e3e2d23c218b4cd0a.png" class="math-display" /></div>



と定義されます。  
つまり、「A か B の少なくとも一方に属する元全体」です。

__3つ以上の有限個の集合の場合__

同様に、有限個の集合 A₁, A₂, …, Aₙ に対して、



<div class="math-display-container"><img src="tmp/e7d0200dfbc4c9cd638d220df08ed1d0.png" class="math-display" /></div>



と定義します。

### 2. 一般の集合族の和

ここからが本題です。  
「集合族」とは、**集合を要素とする集まり**のことです。

__2.1 集合族の定義（形式的）__

集合 X を全体集合とし、その部分集合の集まり <img src="tmp/86b43e0269ceada40fe0d8cc410941f5.png" class="math-inline" /> を考えます。  
<img src="tmp/86b43e0269ceada40fe0d8cc410941f5.png" class="math-inline" /> が**集合族**であるとは、

- <img src="tmp/8e4fc75ac44e929400395b944f364a0f.png" class="math-inline" />  
  （<img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> は X のべき集合）

という意味です。

例：
- <img src="tmp/63ea67b85bbfc6d8fdc46cf4ca25d350.png" class="math-inline" />  
  ここで A, B, C は X の部分集合。

__2.2 集合族の和の定義__

集合族 <img src="tmp/86b43e0269ceada40fe0d8cc410941f5.png" class="math-inline" /> に対して、その**和集合**（union）は



<div class="math-display-container"><img src="tmp/2c9f203e0b02f84ecbc09a75471f758d.png" class="math-display" /></div>



と定義されます。

言葉で言うと：

> 「族 <img src="tmp/86b43e0269ceada40fe0d8cc410941f5.png" class="math-inline" /> に属する少なくとも1つの集合に含まれる元全体」

です。

__2.3 添数づけられた族の場合__

集合族が**添数づけられた族** <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> の形で与えられていることも多いです。  
この場合、和集合は



<div class="math-display-container"><img src="tmp/d70bf9309a954ba9a14416114aa38cb7.png" class="math-display" /></div>



と書かれます。

これは、先ほどの定義



<div class="math-display-container"><img src="tmp/3417b7a58e697c2c6a7325a22f3eb7bb.png" class="math-display" /></div>



と同じ意味です。

### 3. 具体例

__例1：有限族__

- X = ℝ（実数全体）
- A = [0, 1], B = [1, 2], C = [2, 3]
- <img src="tmp/63ea67b85bbfc6d8fdc46cf4ca25d350.png" class="math-inline" />

このとき、



<div class="math-display-container"><img src="tmp/eefb62a423d27b096933d2f6411d628e.png" class="math-display" /></div>



です。

__例2：自然数を添数とする無限族__

- I = ℕ（自然数全体）
- 各 n ∈ ℕ に対して、Aₙ = [n, n+1]（閉区間）
- 族 <img src="tmp/f61df9fb8bb5aea5d7d5503ae37a346e.png" class="math-inline" />

このとき、



<div class="math-display-container"><img src="tmp/4a84cf92581be6836fb0d8b84b0a843a.png" class="math-display" /></div>



となります（1以上の実数全体）。

__例3：実数を添数とする族__

- I = ℝ
- 各 r ∈ ℝ に対して、Aᵣ = {x ∈ ℝ | x ≥ r}
- 族 <img src="tmp/b528c2e01a2eea3e9310c18c602a0299.png" class="math-inline" />

このとき、



<div class="math-display-container"><img src="tmp/1370f75b4f54b26c3e929e99c3534bad.png" class="math-display" /></div>



です（任意の実数 x に対して、r ≤ x となる r を取れば x ∈ Aᵣ となるため）。

### 4. まとめ

- **有限個の集合の和**：  
  <img src="tmp/f4a64a0ec826a7ea03ffc05d0bff524c.png" class="math-inline" />
- **一般の集合族 <img src="tmp/86b43e0269ceada40fe0d8cc410941f5.png" class="math-inline" /> の和**：  
  <img src="tmp/ea1fe28217e0f987a03672345a866a72.png" class="math-inline" />
- **添数づけられた族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> の和**：  
  <img src="tmp/5c7e8d752a6ee5f094572460585e6daa.png" class="math-inline" />

これらはすべて、「少なくとも1つの集合に属する元全体」という同じ考え方を、有限・無限・一般の族に対して統一的に表現したものです。

## 共通部分

「集合の共通部分」には、大きく分けて2つの状況があります。

1. **有限個の集合の共通部分**
2. **一般の集合族（添数づけられた族を含む）の共通部分**

順に説明します。

### 1. 有限個の集合の共通部分

__定義（2つの集合の場合）__

2つの集合 A, B に対して、**共通部分（intersection）** <img src="tmp/93840aa29d13dd6fd5056013d05f866b.png" class="math-inline" /> は、



<div class="math-display-container"><img src="tmp/b7d73c6cfc313133c051bf3f93becf2b.png" class="math-display" /></div>



と定義されます。  
つまり、「A にも B にも同時に属する元全体」です。

__3つ以上の有限個の集合の場合__

同様に、有限個の集合 A₁, A₂, …, Aₙ に対して、



<div class="math-display-container"><img src="tmp/7ffe1c4024f0bd507b8625f1842fa790.png" class="math-display" /></div>



と定義します。

### 2. 一般の集合族の共通部分

「集合族」とは、**集合を要素とする集まり**のことです。

__2.1 集合族の定義（形式的）__

集合 X を全体集合とし、その部分集合の集まり <img src="tmp/86b43e0269ceada40fe0d8cc410941f5.png" class="math-inline" /> を考えます。  
<img src="tmp/86b43e0269ceada40fe0d8cc410941f5.png" class="math-inline" /> が**集合族**であるとは、

- <img src="tmp/8e4fc75ac44e929400395b944f364a0f.png" class="math-inline" />  
  （<img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> は X のべき集合）

という意味です。

例：
- <img src="tmp/63ea67b85bbfc6d8fdc46cf4ca25d350.png" class="math-inline" />  
  ここで A, B, C は X の部分集合。

__2.2 集合族の共通部分の定義__

集合族 <img src="tmp/86b43e0269ceada40fe0d8cc410941f5.png" class="math-inline" /> に対して、その**共通部分**（intersection）は



<div class="math-display-container"><img src="tmp/74d679796dddef74416f473b4ba8f45d.png" class="math-display" /></div>



と定義されます。

言葉で言うと：

> 「族 <img src="tmp/86b43e0269ceada40fe0d8cc410941f5.png" class="math-inline" /> に属する**すべての**集合に含まれる元全体」

です。

__2.3 添数づけられた族の場合__

集合族が**添数づけられた族** <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> の形で与えられていることも多いです。  
この場合、共通部分は



<div class="math-display-container"><img src="tmp/16af919be2f91624736e90317dbc63db.png" class="math-display" /></div>



と書かれます。

これは、先ほどの定義



<div class="math-display-container"><img src="tmp/1efae44c0ef9f35c69a5d3c06d5d3044.png" class="math-display" /></div>



と同じ意味です。

### 3. 具体例

__例1：有限族__

- X = ℝ（実数全体）
- A = [0, 2], B = [1, 3], C = [1, 2]
- <img src="tmp/63ea67b85bbfc6d8fdc46cf4ca25d350.png" class="math-inline" />

このとき、



<div class="math-display-container"><img src="tmp/c6ce7fe3296ac4c21274464583150e1a.png" class="math-display" /></div>



です。

__例2：自然数を添数とする無限族__

- I = ℕ（自然数全体）
- 各 n ∈ ℕ に対して、Aₙ = [−1/n, 1/n]（閉区間）
- 族 <img src="tmp/f61df9fb8bb5aea5d7d5503ae37a346e.png" class="math-inline" />

このとき、



<div class="math-display-container"><img src="tmp/5a784809445deb9d8fbb4531a4a3f65f.png" class="math-display" /></div>



となります（0だけがすべての区間に含まれる）。

__例3：空でない集合族の共通部分__

- X = ℝ
- <img src="tmp/24421e58f91fb6238e62226c981161f1.png" class="math-inline" />  
  （正の実数 r ごとに区間 [0, r] を考える）

このとき、



<div class="math-display-container"><img src="tmp/d0c10a806c886b6e54cc7afd3bdb1702.png" class="math-display" /></div>



です（0だけがすべての [0, r] に含まれる）。

### 4. 空集合族の共通部分についての注意

集合族 <img src="tmp/86b43e0269ceada40fe0d8cc410941f5.png" class="math-inline" /> が**空集合**（<img src="tmp/427ef75a0a0b41c24cb6d52742d7ca65.png" class="math-inline" />）の場合、  
「すべての A ∈ <img src="tmp/86b43e0269ceada40fe0d8cc410941f5.png" class="math-inline" /> について x ∈ A」という条件は、  
「A が存在しないので、条件は自動的に真」と解釈されます。

そのため、**空族の共通部分は全体集合 X** と定義されることが多いです。

- <img src="tmp/28b13cc8e76c9f12f0c6b5c979626699.png" class="math-inline" />

これは、和集合の空族の場合（<img src="tmp/b5d6038370908245249a60751336dc05.png" class="math-inline" />）と対照的です。


## 集合族の直積

「集合族の直積」は、**添数づけられた族** <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> に対して定義される概念です。  
ここでは、次の順に説明します。

1. 直積の直感的なイメージ
2. 形式的な定義（写像としての定義）
3. 有限個の集合の直積（デカルト積）との関係
4. 具体例
5. 選択公理との関係（簡単に）

因みに"デカルト積"とも呼ばれます。同じ意味です。

### 1. 直感的なイメージ

添数づけられた族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> の**直積**とは、

> 「各 i ∈ I に対して、Aᵢ の元を1つずつ選んでできる『選択の仕方』全体の集合」

です。

- 各 i ごとに「どの元を選ぶか」を決めると、1つの「選択関数」ができます。
- そのような関数全体の集合が、直積 <img src="tmp/dfb0684c0df64178490392db6ad4b469.png" class="math-inline" /> です。

### 2. 形式的な定義

__2.1 前提：添数づけられた族__

インデックス集合 I と、各 i ∈ I に対する集合 Aᵢ からなる族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> を考えます。

__2.2 直積の定義__

族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> の**直積（product）** <img src="tmp/dfb0684c0df64178490392db6ad4b469.png" class="math-inline" /> は、次のように定義されます。



<div class="math-display-container"><img src="tmp/631dd23676ded5d543045f87e14ca302.png" class="math-display" /></div>



言葉で言うと：

- まず、すべての Aᵢ の和集合 <img src="tmp/a9b05ef5bc1b66fc3f36d59f169af84a.png" class="math-inline" /> を取る。
- その上で、「I からその和集合への関数 f」のうち、
  - 各 i に対して f(i) ∈ Aᵢ を満たすもの全体を集めた集合が直積。

この f は、**各 i ごとに Aᵢ から1つ元を選ぶ関数**とみなせます。

### 3. 有限個の集合の直積（デカルト積）との関係

__3.1 I = {1, 2, …, n} の場合__

I = {1, 2, …, n} とすると、直積は



<div class="math-display-container"><img src="tmp/3002309fc6df9cdd4596dd3b6105d2f3.png" class="math-display" /></div>



となります。

このとき、各 f は

- f(1) ∈ A₁
- f(2) ∈ A₂
- …
- f(n) ∈ Aₙ

を満たすので、f を**n-組** <img src="tmp/8b4030f64e79007ed224ed584aac284c.png" class="math-inline" /> と同一視できます。  
ここで aᵢ = f(i) です。

したがって、



<div class="math-display-container"><img src="tmp/6db085aca83fc06fb91166ef564a9cd8.png" class="math-display" /></div>



となり、これは通常の**デカルト積**と同じものです。

__3.2 一般の I に対する直積は「無限組」の一般化__

- I が有限集合のとき：直積は「有限組」の集合
- I が無限集合のとき：直積は「無限組」の集合

とみなせます。  
ただし、無限個の「組」を扱うには、**関数として定義する**のが数学的に自然です。

### 4. 具体例

__例1：有限直積（I = {1, 2}）__

- A₁ = {1, 2}, A₂ = {3, 4}

このとき、



<div class="math-display-container"><img src="tmp/c51712ac43ccd061c16c8145751fdd46.png" class="math-display" /></div>



です（通常の2次元デカルト積と同じ）。

__例2：無限直積（I = ℕ）__

- I = ℕ（自然数全体）
- 各 n ∈ ℕ に対して Aₙ = {0, 1}

このとき、



<div class="math-display-container"><img src="tmp/d438060a085bd398369db9e8c8de93f8.png" class="math-display" /></div>



は、「各自然数 n に対して 0 か 1 を割り当てる関数」全体の集合です。  
これは、**0と1からなる無限列**全体の集合と同一視できます。

__例3：実数の無限直積__

- I = ℝ
- 各 r ∈ ℝ に対して Aᵣ = ℝ（実数全体）

このとき、



<div class="math-display-container"><img src="tmp/aadd85552fe54b811f821da733ea357c.png" class="math-display" /></div>



は、「各実数 r に対して実数を1つ対応させる関数」全体の集合です。  
これは、**実数上の実数値関数全体の集合**とみなせます。

### 5. なんで必要となった？

直積の考え方が必要となった主な理由は、次の4点にまとめられます。

1. **複数の対象を「組」として扱うため**  
   - 2つ以上の集合から1つずつ元を取って並べた「組」を数学的に定式化する必要があった。  
   - 例：平面上の点＝(x座標, y座標) という2つの実数の組。

2. **無限個の対象を統一的に扱うため**  
   - 有限個ならタプル (a₁, a₂, …, aₙ) で済むが、無限個は「…」で書けない。  
   - そこで、各添数 i に対して元を対応させる**関数**として「無限組」を表現する一般化が必要になった。

3. **関数空間・新しい数学的構造を作るため**  
   - 直積 <img src="tmp/3c49288130690dc02c85f7ae3e22bcae.png" class="math-inline" /> は「I から X への関数全体」とみなせ、**関数空間**の自然な定式化になる。  
   - 群・環・位相空間などの**直積**を取ることで、新しい構造（直積群・直積環・直積位相空間など）を構成できる。

4. **選択公理・存在証明の舞台として必要だったため**  
   - 選択公理は「空でない集合の族の直積は空でない」という形で述べられ、  
     直積の概念がなければ選択公理を厳密に定式化できない。  
   - 多くの存在定理（無限列・無限選択を伴うもの）は、直積の元（選択関数）の存在に帰着される。

要するに、直積は  
- 「組」の一般化  
- 無限個の対象・関数空間の扱い  
- 新しい構造の構成  
- 選択公理や存在証明の厳密化  

のために不可欠な概念として必要とされました。

## 選択公理

**選択公理（Axiom of Choice, AC）** は、集合論の公理の1つで、**「無限個の集合から同時に1つずつ元を選べる」** ことを保証する公理です。

### 1. 直感的なイメージ

- 有限個の集合 A₁, A₂, …, Aₙ がすべて空でないなら、それぞれから1つずつ元 a₁ ∈ A₁, a₂ ∈ A₂, …, aₙ ∈ Aₙ を選ぶことは、直感的に「明らか」にできます。
- しかし、**無限個**の集合の族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> がすべて空でないとき、「各 i ごとに Aᵢ から1つ元を選ぶ」という操作を、有限の場合のように「明らか」とは言えません。

選択公理は、この**無限個の同時選択**を認める公理です。

### 2. 形式的な定義

__2.1 集合族の直積との関係__

前の回答で説明したように、集合族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> の**直積**は



<div class="math-display-container"><img src="tmp/631dd23676ded5d543045f87e14ca302.png" class="math-display" /></div>



と定義されます。  
ここで f は「各 i に対して Aᵢ から1つ元を選ぶ関数」です。

__2.2 選択公理の主張__

**選択公理**は、次のように述べられます。

> 任意の集合族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> について、  
> すべての i ∈ I に対して Aᵢ ≠ ∅ ならば、  


<div class="math-display-container"><img src="tmp/298743cf744aaa46a0c32544b987820d.png" class="math-display" /></div>


> である。

言葉で言うと：

- 「空でない集合からなる任意の族に対して、その直積は空でない」  
  ＝ 「空でない集合の族からは、同時に1つずつ元を選ぶことができる」

### 3. 具体例で見る選択公理の必要性

__例1：有限個の集合__

- A = {1, 2}, B = {3, 4}, C = {5, 6}

このとき、直積 A × B × C は



<div class="math-display-container"><img src="tmp/845b10a8a601fcc70d90befb8bc4b6f9.png" class="math-display" /></div>



と、**具体的に書き下せます**。  
有限個なら、選択公理なしでも「選べる」ことが明らかです。

__例2：無限個の集合__

- I = ℕ（自然数全体）
- 各 n ∈ ℕ に対して Aₙ = ℕ（自然数全体）

このとき、直積 <img src="tmp/f8e65c12091d0553a539534e3c40ef7c.png" class="math-inline" /> は

- 「各自然数 n に対して自然数を1つ対応させる関数」全体の集合

ですが、そのような関数を**具体的に1つ書き下す**ことはできません。  
（「f(n) = n」と決めれば1つ作れますが、それは特定の選び方の1つに過ぎません。）

選択公理は、

> 「とにかく、そのような関数が少なくとも1つ存在する」

と主張するものです。

### 4. 選択公理と同値な命題（代表例）

選択公理は、数学の多くの分野で使われる重要な定理と**同値**であることが知られています。  
代表的なものを挙げます。

__4.1 整列可能定理（Well-ordering theorem）__

> 任意の集合は、適切な順序を入れることで**整列集合**にできる。

- 整列集合とは、「空でない部分集合が必ず最小元を持つ」ような順序集合です。
- 例えば、実数全体 ℝ に「通常の大小関係」を入れると整列集合にはなりませんが、選択公理を使うと、ℝ にも**何らかの**整列順序が存在することが示せます。

__4.2 ツォルンの補題（Zorn’s lemma）__

> 帰納的順序集合は極大元を持つ。

- 帰納的順序集合とは、「任意の全順序部分集合が上界を持つ」ような順序集合です。
- ツォルンの補題は、線形代数の「基底の存在」や、環論の「極大イデアルの存在」など、  
  多くの存在証明で使われます。

__4.3 ハウスドルフの極大原理__

> 任意の半順序集合には、極大な全順序部分集合が存在する。

これもツォルンの補題と同値な形の1つです。

### 5. 選択公理が「自明」でない理由

- 有限個の選択は、人間の直感でも「明らか」に思えます。
- しかし、**無限個の同時選択**は、「具体的な規則で選ぶ」ことができない場合も多く、その存在を保証するには**公理**が必要になります。
- 選択公理を認めると、  
  - 実数全体 ℝ に整列順序が存在する  
  - ベクトル空間に基底が存在するなど、直感に反する（ように見える）結果も導かれます。

そのため、選択公理は**ZFC集合論**の公理の1つとして採用されていますが、  
「本当に認めてよいのか」という哲学的議論の対象にもなっています。


>__公理__  
>著者は数学で出てくる**公理（axiom）** とは。。。と思うことが結構ありました。公理が何かはクリアにしたいところです。
>ずばり公理とは、ある理論の中で**証明なしに正しいとみなす前提**のことです。
>もう少し詳しく言うと：
>- 数学や論理学では、何かを証明するときに「それより前の事実」に頼ります。
>- しかし、どこまでも遡ると、「これ以上遡れない出発点」が必要になります。
>- その「出発点」として、**証明せずに認める命題**を**公理**と呼びます。
>__1. 公理の役割__  
>__1.1 理論の土台__  
>公理は、その理論の**基礎的なルール**や**前提条件**を定めるものです。
>- 例：ユークリッド幾何学では、
>  - 「2点を通る直線は1本だけ引ける」
>  - 「すべての直角は互いに等しい」
>  といった命題が公理として設定されています。
>これらの公理から、三角形の合同条件や三平方の定理などが**証明**されます。
>__1.2 無定義用語との関係__  
>公理は、しばしば**無定義用語（primitive terms）** とセットで導入されます。
>- 例：集合論では、「集合」「元（要素）」「属する（∈）」といった概念を厳密には定義せず、  
>  その代わりに、それらが満たすべき性質を**公理**として与えます。
>- つまり、「集合とは何か」を直接定義するのではなく、  
>  「集合が満たすべきルール」を公理として定めることで、間接的に特徴づけます。
>__2. 公理の例__  
>__2.1 ユークリッド幾何学の公理__  
>ユークリッドの『原論』では、次のような公理（公準）が挙げられています。
>1. 任意の2点を通る直線を引ける。
>2. 有限直線を連続して一直線に延長できる。
>3. 任意の点を中心とし、任意の半径の円を描ける。
>4. すべての直角は互いに等しい。
>5. （平行線公理）直線とその上にない1点が与えられたとき、その点を通り元の直線と交わらない直線はただ1本存在する。
>これらを出発点として、幾何学の定理が導かれます。
>__2.2 集合論の公理（ZFC）__  
>現代数学の基礎としてよく使われる**ZFC集合論**では、次のような公理があります（一部）。
>- **外延性の公理**：2つの集合が同じ元を持つなら、それらは等しい。
>- **対の公理**：任意の2つの集合に対して、それらだけを元とする集合が存在する。
>- **和集合の公理**：集合の族に対して、その要素すべての和集合が存在する。
>- **べき集合の公理**：任意の集合に対して、その部分集合全体の集合（べき集合）が存在する。
>- **無限公理**：無限集合が存在する。
>- **選択公理**：空でない集合からなる任意の族に対して、その直積は空でない。
>これらから、自然数・実数・関数・順序数など、数学のさまざまな概念が構成されます。
>__3. 公理の性質__  
>__3.1 証明不能な前提__  
>公理は、その理論の内部では**証明されません**。
>- 公理それ自体を証明しようとすると、  
>  さらに別の前提（より基本的な公理）が必要になります。
>- どこかで「これ以上遡らない前提」を置かないと、無限後退に陥ります。
>- そのため、**あるレベルで「これが前提です」と宣言する必要**があり、それが公理です。
>__3.2 真偽ではなく「採用するかどうか」の問題__  
>公理は、**絶対的な真理**というより、**理論を構築するためのルール**です。
>- ある公理系を採用すると、その上で成り立つ定理の体系が得られます。
>- 別の公理系を採用すると、別の体系が得られます。
>- 例：ユークリッド幾何学の「平行線公理」を変えると、非ユークリッド幾何学が得られます。
>したがって、「この公理は正しいか？」よりも、「この公理を採用すると、どのような数学ができるか？」という観点が重要になります。
>__3.3 無矛盾性・独立性__  
>公理系について、次のような性質が問題になります。
>- **無矛盾性**：公理から矛盾（A と ¬A が両方証明される）が導かれないこと。
>- **独立性**：ある公理が、他の公理から証明できないこと。  
>  （例：平行線公理は、ユークリッドの他の公理から独立であることが知られています。）

## Dependent Choice（従属選択公理）

**Dependent Choice（従属選択公理）** は、選択公理（AC）より弱いが、可算選択公理（ACω）より強い中間的な公理です。  
主に**再帰的な選択プロセス**を扱う際に使われます。

### 1. 直感的なイメージ

通常の選択公理（AC）は、

> 「空でない集合の族から、同時に1つずつ元を選べる」

という強い主張です。

Dependent Choice は、これより**制限された形**で、

> 「各ステップでの選択が、前のステップの選択に依存するような『無限列』を作れる」

ことを保証します。

例：
- 1歩目：A₀ から a₀ を選ぶ。
- 2歩目：a₀ に依存して決まる集合 A₁(a₀) から a₁ を選ぶ。
- 3歩目：a₀, a₁ に依存して決まる集合 A₂(a₀, a₁) から a₂ を選ぶ。
- …

このように、「次に選べる集合が、それまで選んだものに依存する」状況で、  
**無限に続く列 (a₀, a₁, a₂, …) が存在する**と主張するのが Dependent Choice です。

### 2. 形式的な定義

Dependent Choice は、通常次のように述べられます。

> 集合 X とその上の二項関係 R ⊂ X × X が与えられ、任意の x ∈ X に対して、ある y ∈ X が存在して (x, y) ∈ R となるとする（つまり R は「全域的」）。  
> このとき、X の点の無限列 (x₀, x₁, x₂, …) で、すべての n について (xₙ, xₙ₊₁) ∈ R を満たすものが存在する。

言葉で言うと：

- X 上の関係 R が「どの点からも必ずどこかへ行ける」なら、「R に沿って無限に進み続ける列」が存在する。

この R が「次に選べる候補」を表し、xₙ に対する「次に選べる集合」が {y ∈ X | (xₙ, y) ∈ R} とみなせます。

### 3. 他の選択公理との関係

選択公理の強弱関係は、おおよそ次のようになります。



<div class="math-display-container"><img src="tmp/21614c0f748cf456c153f868f339c454.png" class="math-display" /></div>



- **AC（Axiom of Choice）**：任意の空でない集合の族に対して選択関数が存在。
- **DC（Dependent Choice）**：依存関係のある無限選択列が存在。
- **ACω（Countable Choice）**：可算個の空でない集合の族に対して選択関数が存在。

つまり、

- AC は非常に強い（無限個の独立な選択を許す）。
- DC は「依存関係のある無限列」に限定した形。
- ACω は「可算個の独立な選択」に限定した形。

DC は、ACω より強く、AC より弱いことが知られています。

### 4. 具体例でのイメージ

__例1：実数列の構成__

- X = ℝ（実数全体）
- 関係 R を、「x と y の差が 1 より小さい」と定義：
  

<div class="math-display-container"><img src="tmp/ce214dc72483e6b0d29d5240d6d3434d.png" class="math-display" /></div>


- 任意の実数 x に対して、|x − y| < 1 となる y は無数に存在するので、R は全域的。

DC を使うと、

- 任意の初期値 x₀ から始めて、
- 各ステップで「前の値から距離 1 未満の点」を選び続ける無限列 (x₀, x₁, x₂, …) が存在する

ことが保証されます。

__例2：木（tree）上の無限経路__

- X をある木（tree）の頂点集合とし、
- R を「親子関係」（子へ進む関係）とする。

もし木が「どの頂点からも必ず子が存在する」（無限に分岐し続ける）なら、DC により、**根から始まる無限経路**が存在することが保証されます。

これは、**Königの補題**（可算無限の木で、各頂点が有限分岐なら無限経路が存在）の証明などで本質的に使われます。

### 5. どのような場面で使われるか

Dependent Choice は、次のような分野でよく現れます。

- **記述集合論**：実数上のボレル集合・解析的集合の性質を調べる際。
- **実解析・関数解析**：バナッハ空間上のベールの範疇定理の証明など。
- **再帰理論**：無限に続く計算・選択プロセスを扱うとき。
- **モデル理論**：ある構造から別の構造への「無限ステップの構成」を行うとき。

特に、「依存関係のある無限列」を扱う証明で、  
AC を使うと強すぎるが、ACω では足りない、という場面で DC がちょうどよい強さになります。


## 演習

以下、直積・選択公理・Dependent Choice に関する演習問題を出題します。  
難易度は易しいものから少し考えるものまで混ぜています。

### 問題

__1. 直積に関する問題__

__問題1-1（定義の確認）__

集合族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> の直積 <img src="tmp/dfb0684c0df64178490392db6ad4b469.png" class="math-inline" /> を、選択関数を用いて定義せよ。

__問題1-2（有限直積と通常の直積の関係）__

I = {1, 2, 3} とし、A₁ = {1, 2}, A₂ = {3, 4}, A₃ = {5, 6} とする。

1. 直積 <img src="tmp/ac4ac22cab06a1b17f76dacffd66d462.png" class="math-inline" /> の要素をすべて列挙せよ。
2. それを通常の3次元デカルト積 A₁ × A₂ × A₃ と同一視できることを説明せよ。

__問題1-3（無限直積の例）__

I = ℕ（自然数全体）とし、各 n ∈ ℕ に対して Aₙ = {0, 1} とする。

1. 直積 <img src="tmp/b8f55f55fe81b17e52809b91eef9eda0.png" class="math-inline" /> の1つの要素を具体的に1つ構成せよ。
2. この直積の要素全体は、どのような数学的対象とみなせるか説明せよ。

__2. 選択公理（AC）に関する問題__

__問題2-1（選択公理の主張）__

選択公理を、「集合族の直積」を用いて述べよ。

__問題2-2（有限選択はACなしで可能）__

I = {1, 2, 3} とし、各 i ∈ I に対して Aᵢ ≠ ∅ とする。

選択公理を用いずに、直積 <img src="tmp/ac4ac22cab06a1b17f76dacffd66d462.png" class="math-inline" /> が空でないことを示せ。  
（ヒント：有限個なら具体的に選べる。）

__問題2-3（AC と同値な命題）__

次の命題が選択公理と同値であることを、直観的に説明せよ。

> 任意の集合は、適切な順序を入れることで整列集合にできる。  
> （整列可能定理）

__3. Dependent Choice（DC）に関する問題__

__問題3-1（DC の主張）__

Dependent Choice を、「二項関係と無限列」の言葉で述べよ。

__問題3-2（DC の具体例）__

X = ℝ（実数全体）とし、二項関係 R を



<div class="math-display-container"><img src="tmp/ce214dc72483e6b0d29d5240d6d3434d.png" class="math-display" /></div>



で定義する。

1. R が「全域的」であることを示せ（すなわち、任意の x ∈ ℝ に対して (x, y) ∈ R となる y が存在する）。
2. Dependent Choice を用いると、任意の x₀ ∈ ℝ から始まる無限列 (x₀, x₁, x₂, …) で、すべての n について (xₙ, xₙ₊₁) ∈ R を満たすものが存在することを説明せよ。

__問題3-3（DC と AC の強弱）__

次の包含関係が成り立つ理由を直観的に説明せよ。



<div class="math-display-container"><img src="tmp/21614c0f748cf456c153f868f339c454.png" class="math-display" /></div>



ここで ACω は「可算個の空でない集合の族に対して選択関数が存在する」という可算選択公理とする。

__4. 少し発展的な問題__

__問題4-1（直積と関数空間）__

I を任意の集合、X を固定された集合とする。  
直積 <img src="tmp/3c49288130690dc02c85f7ae3e22bcae.png" class="math-inline" /> が、どのような関数空間と自然に同一視できるか説明せよ。

__問題4-2（DC を使った存在証明のイメージ）__

次の状況を考える。

- 頂点集合 V を持つ無限の木（tree）があり、
- 各頂点には少なくとも1つの子が存在する（すなわち木は無限に伸び続ける）。

Dependent Choice を用いて、「根から始まる無限経路」が存在することを説明せよ。  
（ヒント：R を「親から子へ進む関係」とみなす。）

### 解答

__1. 直積に関する問題__

__問題1-1（定義の確認）__

集合族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> の直積 <img src="tmp/dfb0684c0df64178490392db6ad4b469.png" class="math-inline" /> は、次のように定義される。



<div class="math-display-container"><img src="tmp/631dd23676ded5d543045f87e14ca302.png" class="math-display" /></div>



ここで、f は「各 i ∈ I に対して Aᵢ の元を1つ選ぶ関数」であり、**選択関数**と呼ばれる。

__問題1-2（有限直積と通常の直積の関係）__

I = {1, 2, 3}, A₁ = {1, 2}, A₂ = {3, 4}, A₃ = {5, 6} とする。

1. 直積 <img src="tmp/ac4ac22cab06a1b17f76dacffd66d462.png" class="math-inline" /> の要素は、選択関数 f: {1,2,3} → {1,2,3,4,5,6} で、f(1)∈A₁, f(2)∈A₂, f(3)∈A₃ を満たすもの全体である。  
   これを3-組 (f(1), f(2), f(3)) と同一視すると、要素は次の8通り。

   

<div class="math-display-container"><img src="tmp/461e2f582920b25c1fa38a8d299d296e.png" class="math-display" /></div>



2. 通常の3次元デカルト積は  
   

<div class="math-display-container"><img src="tmp/9b68bfa663e433cc0b6092c6e38e915b.png" class="math-display" /></div>


   であり、これは上で列挙した集合と一致する。  
   したがって、有限個の直積 <img src="tmp/461c2510b1f6fbe0c1e842d3ddd3c7ee.png" class="math-inline" /> は、n-組の集合 A₁ × … × Aₙ と自然に同一視できる。

__問題1-3（無限直積の例）__

I = ℕ, 各 n ∈ ℕ に対して Aₙ = {0, 1} とする。

1. 直積 <img src="tmp/b8f55f55fe81b17e52809b91eef9eda0.png" class="math-inline" /> の1つの要素として、例えば  
   

<div class="math-display-container"><img src="tmp/20c121a0648845895c5b81591f25daff.png" class="math-display" /></div>


   と定めれば、f は選択関数であり、直積の要素である。

2. この直積の要素全体は、  
   「各自然数 n に対して 0 か 1 を割り当てる関数」全体の集合である。  
   これは、**0 と 1 からなる無限列**全体の集合と同一視できる。  
   言い換えると、**2進無限列**全体の集合である。

__2. 選択公理（AC）に関する問題__

__問題2-1（選択公理の主張）__

選択公理は、集合族の直積を用いて次のように述べられる。

> 任意の集合族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> について、  
> すべての i ∈ I に対して Aᵢ ≠ ∅ ならば、  


<div class="math-display-container"><img src="tmp/298743cf744aaa46a0c32544b987820d.png" class="math-display" /></div>


> である。

すなわち、「空でない集合からなる任意の族に対して、その直積は空でない」。

__問題2-2（有限選択はACなしで可能）__

I = {1, 2, 3}, 各 Aᵢ ≠ ∅ とする。

- A₁ ≠ ∅ より、ある a₁ ∈ A₁ が存在する。
- A₂ ≠ ∅ より、ある a₂ ∈ A₂ が存在する。
- A₃ ≠ ∅ より、ある a₃ ∈ A₃ が存在する。

これらを成分とする3-組 (a₁, a₂, a₃) は、  
選択関数 f: {1,2,3} → ⋃ Aᵢ を f(1)=a₁, f(2)=a₂, f(3)=a₃ と定めたものに対応する。  
この f はすべての i について f(i) ∈ Aᵢ を満たすので、直積 <img src="tmp/ac4ac22cab06a1b17f76dacffd66d462.png" class="math-inline" /> の要素である。

したがって、有限個の場合は**具体的に元を選べる**ため、選択公理なしでも直積が空でないことが示せる。

__問題2-3（AC と同値な命題）__

> 任意の集合は、適切な順序を入れることで整列集合にできる。（整列可能定理）

**直観的な説明**：

- 選択公理は「空でない集合の族から同時に1つずつ元を選べる」ことを保証する。
- 整列可能定理は「任意の集合 X に整列順序（空でない部分集合が最小元を持つ順序）が存在する」と主張する。
- 整列順序を構成するには、X の部分集合から「最小元候補」を繰り返し選び続ける必要があり、これは**無限回の選択**を伴う。
- この「無限回の選択」を正当化するには選択公理が必要であり、逆に選択公理から整列可能定理が導かれる。
- したがって、両者は同値である。

__3. Dependent Choice（DC）に関する問題__

__問題3-1（DC の主張）__

Dependent Choice は、次のように述べられる。

> 集合 X とその上の二項関係 R ⊂ X × X が与えられ、  
> 任意の x ∈ X に対して、ある y ∈ X が存在して (x, y) ∈ R となるとする（R は全域的）。  
> このとき、X の点の無限列 (x₀, x₁, x₂, …) で、  
> すべての n について (xₙ, xₙ₊₁) ∈ R を満たすものが存在する。

__問題3-2（DC の具体例）__

X = ℝ, R = {(x, y) ∈ ℝ² | |x − y| < 1} とする。

1. 任意の x ∈ ℝ に対して、例えば y = x とすれば |x − y| = 0 < 1 なので (x, y) ∈ R。  
   よって R は全域的である（実際には各 x に対して無数の y が存在する）。

2. Dependent Choice を仮定すると、  
   - 任意の初期点 x₀ ∈ ℝ を固定する。
   - R は全域的だから、各ステップで「現在の点から距離 1 未満の点」を選び続けることができる。
   - DC により、そのような無限列 (x₀, x₁, x₂, …) が存在することが保証される。

__問題3-3（DC と AC の強弱）__



<div class="math-display-container"><img src="tmp/21614c0f748cf456c153f868f339c454.png" class="math-display" /></div>



**直観的な説明**：

- **AC ⇒ DC**：  
  選択公理は「任意の空でない集合の族から同時に1つずつ元を選べる」という非常に強い主張。  
  Dependent Choice は「依存関係のある無限列」の存在のみを要求するので、AC から導かれる。

- **DC ⇒ ACω**：  
  DC は「依存関係のある無限選択列」の存在を保証するが、ACω は「可算個の独立な選択」のみを要求する。  
  依存関係を「自明」（常に同じ集合族から選ぶ）とすれば、DC から ACω が導かれる。

したがって、AC が最も強く、DC はその中間、ACω が最も弱い。

__4. 少し発展的な問題__

__問題4-1（直積と関数空間）__

I を任意の集合、X を固定された集合とする。

直積 <img src="tmp/3c49288130690dc02c85f7ae3e22bcae.png" class="math-inline" /> は、



<div class="math-display-container"><img src="tmp/a56304cabec32b7e4d92e5ca76be0e11.png" class="math-display" /></div>



であるが、各 Aᵢ = X なので、これは単に



<div class="math-display-container"><img src="tmp/623de35a69de3be4065199a6c6f8b666.png" class="math-display" /></div>



と一致する。

したがって、<img src="tmp/3c49288130690dc02c85f7ae3e22bcae.png" class="math-inline" /> は、**I から X への関数全体の集合**、すなわち**関数空間** <img src="tmp/45db5f1fd4dc24ae42d4016240959282.png" class="math-inline" /> と自然に同一視できる。

__問題4-2（DC を使った存在証明のイメージ）__

- 頂点集合 V の無限木で、各頂点に少なくとも1つの子が存在するとする。
- 二項関係 R を「親から子へ進む関係」と定義する：  
  (v, w) ∈ R ⇔ w は v の子。

このとき、

- 各頂点 v には少なくとも1つの子が存在するので、R は全域的。
- Dependent Choice を仮定すると、任意の根 r から始まる無限列 (r = v₀, v₁, v₂, …) で、  
  すべての n について (vₙ, vₙ₊₁) ∈ R を満たすものが存在する。
- これは「根から始まる無限経路」の存在を意味する。

したがって、DC を用いると、「各頂点が子を持つ無限木には、根から始まる無限経路が存在する」ことが示せる。


<div style="page-break-before:always"></div>


# 同値関係


## 直和分割
集合論における**直和分割（disjoint union decomposition / partition into a disjoint union）** とは、ある集合を「互いに交わらない（共通部分が空である）部分集合」の和集合として表すことです。

### 1. 直和分割の定義

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> に対し、その部分集合の族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" />（ただし <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> は添字集合）が次の2条件を満たすとき、<img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**直和分割**であるといいます。

1. **被覆条件（カバー）**  
   

<div class="math-display-container"><img src="tmp/b4406c2538a44900b5c12a3899dea37b.png" class="math-display" /></div>


   すなわち、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> のどの元も、少なくとも一つの <img src="tmp/b4cf280200862773bb25975b26b9248c.png" class="math-inline" /> に属する。

2. **互いに素（pairwise disjoint）**  
   任意の <img src="tmp/e54da2ff63cec8e07fbd3b72880468a5.png" class="math-inline" /> で <img src="tmp/fb50bc9b051b9ef7941637f336762cfa.png" class="math-inline" /> ならば
   

<div class="math-display-container"><img src="tmp/ace25167a9dbde30db4840959b793dc3.png" class="math-display" /></div>


   が成り立つ。

このとき、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> は <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> の**直和（disjoint union）** であるといい、


<div class="math-display-container"><img src="tmp/2e770dd22358761851dfa8d5e074a6a9.png" class="math-display" /></div>


と書くことがあります（<img src="tmp/3cd3ec77c006bededfa37bb9c90f967e.png" class="math-inline" /> は直和の記号）。

### 2. 簡単な例

- 自然数の集合 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> を「偶数全体」と「奇数全体」に分ける：
  

<div class="math-display-container"><img src="tmp/dbeb67e477a60005d951de56b49e659f.png" class="math-display" /></div>


  とすると、
  

<div class="math-display-container"><img src="tmp/e6696e76577245f73ee4e43a409ce155.png" class="math-display" /></div>


  なので、<img src="tmp/e81cd7ed1b1c1a13d1be0d216439cdee.png" class="math-inline" /> は <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> の直和分割です。

- 実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> を負の数・ゼロ・正の数に分ける：
  

<div class="math-display-container"><img src="tmp/2146de52d0a2e8e72d00e45e7a72bec7.png" class="math-display" /></div>


  とすると、
  

<div class="math-display-container"><img src="tmp/2fe50f4e955650a470b2f613eb67b0c8.png" class="math-display" /></div>


  なので、<img src="tmp/61f4cf28876bfc6c88bd674851e273aa.png" class="math-inline" /> は <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の直和分割です。

### 3. 同値関係との関係

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の**同値関係 <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" />** があるとき、各同値類


<div class="math-display-container"><img src="tmp/da59b353ecdb92377691494016d9e0ec.png" class="math-display" /></div>


を集めた集合族 <img src="tmp/43a6e71d740c61ada4bcc03f617fabc3.png" class="math-inline" /> は、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の直和分割になります。  
このようにして得られる直和分割を**商集合（quotient set）** と呼び、


<div class="math-display-container"><img src="tmp/a1bbb10fd32e87610a9f693060298bc0.png" class="math-display" /></div>


と書きます。


### 4. 注意点

- 「直和」という言葉は、**集合の直和（disjoint union）** と**ベクトル空間の直和（direct sum）** など、文脈によって意味が少し異なることがあります。集合論の文脈では、上記のように「互いに交わらない和集合」を指すのが普通です。
- 添字集合 <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> が有限とは限らず、無限集合でも構いません。

もし、直和分割と関連する概念（同値関係・商集合・直積との違いなど）についても知りたい場合は、その点も補足できます。

## 同値関係

**同値関係（equivalence relation）** とは、集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の元どうしの関係 <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> で、次の3つの条件を満たすものをいいます。

### 1. 同値関係の定義

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の元 <img src="tmp/e8a97f6a2e8c46626a47117edfcb7399.png" class="math-inline" /> について、関係 <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> が以下を満たすとき、<img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の**同値関係**です。

1. **反射律（reflexivity）**  
   任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して
   

<div class="math-display-container"><img src="tmp/5239a7e9a1571f946a383f18885ec523.png" class="math-display" /></div>


   が成り立つ。

2. **対称律（symmetry）**  
   任意の <img src="tmp/6ecf568da0898bbcac56f45f3573860b.png" class="math-inline" /> に対して
   

<div class="math-display-container"><img src="tmp/c2da2551e55c08d47678067310e7882a.png" class="math-display" /></div>


   が成り立つ。

3. **推移律（transitivity）**  
   任意の <img src="tmp/534c08c8ec262bfb3176a6084d63777a.png" class="math-inline" /> に対して
   

<div class="math-display-container"><img src="tmp/c3d5171325826a0883cd17d357114aba.png" class="math-display" /></div>


   が成り立つ。

### 2. 具体例

__例1：整数の「mod 3」による合同関係__  
整数全体 <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> において、


<div class="math-display-container"><img src="tmp/cbbc85da6d0277823c81fc93168564b9.png" class="math-display" /></div>


と定義すると、<img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> は同値関係です。

- 反射律：<img src="tmp/90933e3e57e49fc57d990261547e00d9.png" class="math-inline" /> は3の倍数 → <img src="tmp/8a7e207de9b21e03f15828b5c1f0c59e.png" class="math-inline" />
- 対称律：<img src="tmp/f2d7c0906b3a89c8f04112dcc2760dbc.png" class="math-inline" /> が3の倍数なら <img src="tmp/91601778c1f6c9fe86978e7bfb3e49c6.png" class="math-inline" /> も3の倍数 → <img src="tmp/2ec2ef9df6cbd65190af0c22804c5441.png" class="math-inline" />
- 推移律：<img src="tmp/f2d7c0906b3a89c8f04112dcc2760dbc.png" class="math-inline" /> と <img src="tmp/4cdad60c1d1808d0537a1b8b8d59faaf.png" class="math-inline" /> が3の倍数なら、<img src="tmp/2421b05945831f901f7ba0210cf8881d.png" class="math-inline" /> も3の倍数 → <img src="tmp/282a27126242cbc3ed0d34207676fd95.png" class="math-inline" /> かつ <img src="tmp/1d012d2fe96c328e3191171601992312.png" class="math-inline" />

__例2：平面上の「同じx座標を持つ点」__  
平面 <img src="tmp/b8e63d739e17c94e714c0f375c6651dd.png" class="math-inline" /> の点 <img src="tmp/c53e93989898a64cac83cc0adc748721.png" class="math-inline" /> に対し、


<div class="math-display-container"><img src="tmp/7fe27d2e348bbeffb14a72440511d7bd.png" class="math-display" /></div>


とすると、これは同値関係です。

- 反射律：明らかに <img src="tmp/787eec981cdd4b6038c227f5ec7e807c.png" class="math-inline" />
- 対称律：<img src="tmp/4d9ae0adc7d06a05edc8b4fef9b77312.png" class="math-inline" />
- 推移律：<img src="tmp/70e997ed2e3dd20c1838e199543accba.png" class="math-inline" /> かつ <img src="tmp/b102777cbc5d0aca9cd5e0ab4dbb855c.png" class="math-inline" />

__例3：集合の「同じ要素数」__

有限集合の族において、「要素数が等しい」という関係も同値関係です。

### 3. 同値類と商集合

同値関係 <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> があると、各元 <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して


<div class="math-display-container"><img src="tmp/da59b353ecdb92377691494016d9e0ec.png" class="math-display" /></div>


という集合を考えることができます。これを <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> の**同値類（equivalence class）** といいます。

- 同値類は互いに交わらず、かつ <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 全体を覆います（＝直和分割になります）。
- 同値類全体の集合
  

<div class="math-display-container"><img src="tmp/a1bbb10fd32e87610a9f693060298bc0.png" class="math-display" /></div>


  を**商集合（quotient set）** といいます。

例えば、整数の mod 3 の同値関係では、同値類は


<div class="math-display-container"><img src="tmp/7031c6cd64d5510310b88c5d3b5c7385.png" class="math-display" /></div>


の3つで、商集合は


<div class="math-display-container"><img src="tmp/7eb9fb553187a6d2c18efb8690adb3f6.png" class="math-display" /></div>


となります。

### 4. 必要性

同値関係は、**「同じ種類のもの同士をまとめて、本質的な違いだけに注目する」** ための仕組みです。  
以下、代表的な使われ方と「なぜ必要か」を説明します。

__1. 数学的な場面での使われ方__

__(1) 整数の合同（mod n）__
- 関係：<img src="tmp/478c8e2df02ea0addfe0b31ebdcaa38c.png" class="math-inline" /> は <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> の倍数
- なぜ必要か：  
  整数全体を「n で割った余り」ごとに分類し、**無限にある整数を有限個のタイプにまとめる**ことで、計算や構造の理解を簡単にします。  
  例：mod 3 では、整数は「余り 0, 1, 2」の3種類に分かれ、商集合は <img src="tmp/206157b9c6c5806912cb045907fd995e.png" class="math-inline" /> という有限集合になります。

__(2) 分数（有理数）の定義__
- 関係：<img src="tmp/ca9964a08339f56df5f42f037328a441.png" class="math-inline" />（ただし <img src="tmp/43cfa20f13f1f090edeea6138128538e.png" class="math-inline" />）
- なぜ必要か：  
  <img src="tmp/0ccca668cb33fc05c30e8f70f7427a36.png" class="math-inline" /> と <img src="tmp/763c87519410f35111ca81e12a09a6ab.png" class="math-inline" /> は「見た目は違うが同じ数」です。  
  同値関係を使って「値が等しい分数」を1つの塊（同値類）とみなすことで、**分数の本質（比の値）だけを取り出した有理数**を厳密に定義できます。

__(3) ベクトル空間の商空間__
- 関係：ベクトル <img src="tmp/19e0af95034149031af53581c8fbc097.png" class="math-inline" />（<img src="tmp/5546d3ae443ac0823c814acaa9c34322.png" class="math-inline" /> は部分空間）
- なぜ必要か：  
  部分空間 <img src="tmp/5546d3ae443ac0823c814acaa9c34322.png" class="math-inline" /> の方向を「無視して」、**残りの方向だけを見る**ために使います。  
  例：平面 <img src="tmp/b8e63d739e17c94e714c0f375c6651dd.png" class="math-inline" /> で x軸方向を無視すると、商空間は「y座標だけが本質」の1次元空間になります。

__(4) 位相空間の商空間（同値類で点を貼り合わせる）__
- 関係：円周上の点を「0 と 1 を同一視」するなど
- なぜ必要か：  
  トーラスや射影平面など、**複雑な図形を単純な図形から“貼り合わせ”で作る**ときに使います。

__2. 数学以外の場面での使われ方（概念的な例）__

__(1) 「同じ種類」の分類__
- 関係：2つの図形が「合同」である、2つの単語が「同義語」である、など
- なぜ必要か：  
  細かい違いを無視して、「本質的に同じもの」を1つのグループとして扱うことで、**議論をシンプルにし、パターンを見つけやすく**します。

__(2) 抽象化・モデル化__
- 関係：実世界の複雑な対象を「同じ状態」とみなす（例：同じ気温・同じ天気の日を1つのタイプとみなす）
- なぜ必要か：  
  個々の違いにこだわらず、**重要な特徴だけに注目してモデル化**するために使います。

__3. なぜ同値関係が必要か__

同値関係が必要な理由は、主に次の3点です。

1. **無限・複雑なものを有限・単純な形にまとめる**  
   （例：整数全体 → mod n の有限個の類）

2. **本質的な違いだけに注目し、細かい違いを無視する**  
   （例：分数の表し方の違いを無視して「値」だけを見る）

3. **新しい数学的対象（商集合・商空間）を構成する基礎になる**  
   （例：有理数、商ベクトル空間、商位相空間など）

つまり、同値関係は「同じとみなしてよいもの」を明確に定義し、**抽象化・単純化・新しい構造の構築**を可能にする、数学の基本的な道具です。

もし、特定の例（分数の定義や商空間など）をもう少し詳しく見たい場合は、その例を指定してください。


### 5. 補足

- 同値関係は「同じ種類のもの同士をグルーピングする」ための抽象的な枠組みです。
- 反射律・対称律・推移律のどれか一つでも欠けると、同値関係とは呼べません。
- 同値関係を定めることは、集合を「互いに交わらない部分集合（同値類）」に分けることと本質的に同じです。

もし、商集合や直和分割との関係、あるいは同値関係ではない例（半順序など）についても知りたい場合は、その点も補足できます。


>__商集合__  
>**商集合（quotient set）** とは、ある集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> とその上の同値関係 <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> が与えられたとき、
>- 各元 <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して、その**同値類** <img src="tmp/6c705196a0d344c187210c791260a70a.png" class="math-inline" /> を考え、
>- それら同値類全体を集めた集合
>のことです。記号では


<div class="math-display-container"><img src="tmp/a1bbb10fd32e87610a9f693060298bc0.png" class="math-display" /></div>


と書きます。
>直感的には：
>- 同値関係で「同じとみなせる元」を1つの塊（同値類）にまとめ、
>- その塊たちを新しい「点」として並べた集合
>が商集合です。
例：整数 <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> と mod 3 の合同関係では、


<div class="math-display-container"><img src="tmp/2abedb85f3a07a39364b480625f3ab8f.png" class="math-display" /></div>


の3つの同値類があり、商集合は


<div class="math-display-container"><img src="tmp/7eb9fb553187a6d2c18efb8690adb3f6.png" class="math-display" /></div>


となります。
>つまり、商集合は「同値関係によって元をグルーピングした結果の集合」です。

## 同値関係の強さ

「同値関係の強さ」は、**二つの同値関係の間の包含関係**として数学的に定義されます。

### 1. 定義

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上に二つの同値関係 <img src="tmp/05a314db85b6d9f7cfc6ae7c25919005.png" class="math-inline" /> と <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> があるとします。

- <img src="tmp/05a314db85b6d9f7cfc6ae7c25919005.png" class="math-inline" /> が <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> より**強い（stronger）**、あるいは**細かい（finer）** とは、
  

<div class="math-display-container"><img src="tmp/3dd0a21d34201f75f6c31c566ce5455c.png" class="math-display" /></div>


  が成り立つことをいいます。

- このとき、<img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> は <img src="tmp/05a314db85b6d9f7cfc6ae7c25919005.png" class="math-inline" /> より**弱い（weaker）**、あるいは**粗い（coarser）**　といいます。

つまり、<img src="tmp/05a314db85b6d9f7cfc6ae7c25919005.png" class="math-inline" /> の方が「より多くのペアを同値とみなす」関係になっているとき、<img src="tmp/05a314db85b6d9f7cfc6ae7c25919005.png" class="math-inline" /> は <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> より**強い（細かい）**　と定義されます。


### 2. 具体例

__例1：整数の合同関係__

- <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" />：<img src="tmp/112b53f392022573a73d7b1a3c74c068.png" class="math-inline" /> は 2 の倍数（mod 2）
- <img src="tmp/5886c8769f6d19ce51778c73f25fbf99.png" class="math-inline" />：<img src="tmp/a6613701a702367d77f785763e4bb0f0.png" class="math-inline" /> は 6 の倍数（mod 6）

このとき、


<div class="math-display-container"><img src="tmp/a399bbfb14cfec0ec9ad3af8f134ccbd.png" class="math-display" /></div>


なので、<img src="tmp/5886c8769f6d19ce51778c73f25fbf99.png" class="math-inline" /> は <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> より**強い（細かい）** です。

- mod 6 の同値類：<img src="tmp/beccfe391773bdba3e75985da5647b70.png" class="math-inline" />（6個の類）
- mod 2 の同値類：偶数全体、奇数全体（2個の類）

mod 6 の方が「より細かく」整数を分けているので、強い方です。

__例2：平面上の点__

- <img src="tmp/d1b500bc3e45fcff51bc61f4b54841bd.png" class="math-inline" />：<img src="tmp/145cce13c4f1a0ad40511d3eef3e7bf8.png" class="math-inline" />（x座標が同じ）
- <img src="tmp/0cab5541919d1c15367377973be9f307.png" class="math-inline" />：<img src="tmp/eca33c15653cf5551b55a277d554365a.png" class="math-inline" />（完全に同じ点）

このとき、


<div class="math-display-container"><img src="tmp/37a8174aabb2dd88fed1dd8f65641e08.png" class="math-display" /></div>


なので、<img src="tmp/0cab5541919d1c15367377973be9f307.png" class="math-inline" /> は <img src="tmp/d1b500bc3e45fcff51bc61f4b54841bd.png" class="math-inline" /> より**強い（細かい）** です。

### 3. 同値類の観点からの解釈

- <img src="tmp/05a314db85b6d9f7cfc6ae7c25919005.png" class="math-inline" /> が <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> より**強い**とき、<img src="tmp/05a314db85b6d9f7cfc6ae7c25919005.png" class="math-inline" /> の同値類は <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> の同値類よりも**細かく**分割されています。
- 言い換えると、<img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> の各同値類は、<img src="tmp/05a314db85b6d9f7cfc6ae7c25919005.png" class="math-inline" /> のいくつかの同値類の**和集合**として書けます。

先ほどの mod 6 と mod 2 の例では、

- mod 6 の同値類：<img src="tmp/9583cd7aa37ce9cd4bafb49f5651ee6c.png" class="math-inline" />
- mod 2 の同値類：<img src="tmp/987231eab87d87cd636dfac711c75442.png" class="math-inline" />

となっており、mod 2 の各類は mod 6 の類の合併になっています。

### 4. 順序関係としての見方

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の同値関係全体の集合を <img src="tmp/483a5a3729b20fbe620bb2ef6af72ded.png" class="math-inline" /> とすると、

- 「<img src="tmp/05a314db85b6d9f7cfc6ae7c25919005.png" class="math-inline" /> は <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> より強い（細かい）」という関係は、<img src="tmp/483a5a3729b20fbe620bb2ef6af72ded.png" class="math-inline" /> 上の**半順序**（反射的・反対称・推移的）になっています。
- 最も弱い（粗い）同値関係は「すべての元を同一視する」関係（＝ <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 全体が1つの同値類）。
- 最も強い（細かい）同値関係は「自分自身としか同値でない」関係（＝各元が別々の同値類）。

### 5. まとめ

- 「同値関係の強さ」は、包含関係 <img src="tmp/008675b8405a2acac9c1a4edb48417ee.png" class="math-inline" /> で定義される。
- 強い方（細かい方）は、より多くのペアを同値とみなし、同値類の分割が細かい。
- 弱い方（粗い方）は、同値類が大きくなり、強い方の同値類をまとめた形になる。

もし、この順序構造や、同値関係の交わり・結び（meet/join）についても知りたい場合は、その点も補足できます。

## 被覆

**被覆（cover）** とは、ある集合を「その部分集合の集まり」で覆う（カバーする）という概念です。主に集合論や位相空間論で使われます。

### 1. 集合の被覆の定義

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> と、その部分集合の族（集合の集まり）<img src="tmp/c2e8e2ca063f65696dcab120056b001a.png" class="math-inline" /> を考えます（<img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> は添字集合）。

<img src="tmp/7910f9a324a076c1f3fe6f03ca20245a.png" class="math-inline" /> が <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**被覆**であるとは、


<div class="math-display-container"><img src="tmp/2e4d9e4301d462d2e15023a675aff3b6.png" class="math-display" /></div>


が成り立つことをいいます。

言い換えると：

- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> のどの元 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> も、少なくとも一つの <img src="tmp/2f062a38e7476e54f32b0cdea57d822d.png" class="math-inline" /> に属する。
- つまり、<img src="tmp/7910f9a324a076c1f3fe6f03ca20245a.png" class="math-inline" /> の要素たちの和集合が <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> を**覆い尽くす**。

このとき、各 <img src="tmp/2f062a38e7476e54f32b0cdea57d822d.png" class="math-inline" /> を**被覆の要素**と呼びます。

### 2. 具体例

__例1：実数直線の区間による被覆__

実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> に対して、


<div class="math-display-container"><img src="tmp/0a6e899df49d2892ab120aaa7c70248c.png" class="math-display" /></div>


とおくと、<img src="tmp/4e35a44f19b75174ec977617f2860fbb.png" class="math-inline" /> は <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の被覆です。  
任意の実数 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は、ある整数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> に対して <img src="tmp/a89dcfd08280f31cd53f03475546c88c.png" class="math-inline" /> を満たすので、必ずどれかの <img src="tmp/fd8458adf0019850e97dfe8f95168e27.png" class="math-inline" /> に含まれます。

__例2：有限集合の被覆__

<img src="tmp/85bcfdc85cb185d1adb8653d097a4429.png" class="math-inline" /> とし、


<div class="math-display-container"><img src="tmp/fdc19463d3287f2200a4e7b4d7398a19.png" class="math-display" /></div>


とすると、<img src="tmp/93636448ef038b709daa91af5a32a3df.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の被覆です。  
実際、<img src="tmp/b12b85b4b6b569ea273c45f5eaf545cd.png" class="math-inline" />、<img src="tmp/1f4522af478b17aa034653c6d3e522e3.png" class="math-inline" />、<img src="tmp/ad6251cdb23b8c23451ef4d8ac2db96d.png" class="math-inline" />、<img src="tmp/d9cc7da969de6e9feef18e3fcffe69ec.png" class="math-inline" /> となり、すべての元がカバーされています。

### 3. 被覆の種類（位相空間の場合）

位相空間 <img src="tmp/60830441dff6a79e579becf819e169f2.png" class="math-inline" /> において、被覆にはいくつかの種類があります。

__(1) 開被覆（open cover）__

各 <img src="tmp/2f062a38e7476e54f32b0cdea57d822d.png" class="math-inline" /> が開集合であるような被覆を**開被覆**といいます。  
例1の <img src="tmp/449c9c62d5e111684b550f4db7b54b34.png" class="math-inline" /> は <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の開区間なので、開被覆です。

__(2) 閉被覆（closed cover）__

各 <img src="tmp/2f062a38e7476e54f32b0cdea57d822d.png" class="math-inline" /> が閉集合であるような被覆を**閉被覆**といいます。

__(3) 有限被覆（finite cover）__

添字集合 <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> が有限集合である被覆を**有限被覆**といいます。  
例2は有限被覆です。

__(4) 部分被覆（subcover）__

<img src="tmp/7910f9a324a076c1f3fe6f03ca20245a.png" class="math-inline" /> が <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の被覆であるとき、その部分族 <img src="tmp/3eb73a8802699afa01b079a03eebcca7.png" class="math-inline" /> で、それ自身も <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の被覆になっているものを**部分被覆**といいます。

### 4. 被覆と直和分割の違い

- **被覆**：和集合が <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> を覆えばよく、<img src="tmp/2f062a38e7476e54f32b0cdea57d822d.png" class="math-inline" /> 同士が交わっていても構いません。
- **直和分割**：被覆であり、かつどの2つも交わらない（互いに素）という条件が追加されます。

つまり、直和分割は「重なりがない特別な被覆」です。

### 5. まとめ

- 被覆とは、集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合の族 <img src="tmp/194ac4b974dd2eeb4320efc22008de61.png" class="math-inline" /> で、その和集合が <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 全体を含むもの。
- 位相空間では、開被覆・閉被覆・有限被覆・部分被覆など、さまざまなバリエーションが定義される。
- 直和分割は「互いに素な被覆」であり、被覆の特別な場合。

もし、位相空間における「コンパクト性」と「任意の開被覆が有限部分被覆を持つ」ことの関係など、もう少し応用的な話も知りたい場合は、その点も補足できます。

## 演習

### 問題

これまでの内容を踏まえて、以下の演習問題に取り組んでみてください。  
必要に応じて、定義や具体例を思い出しながら解いてみてください。


__問1（直和分割）__

集合 <img src="tmp/6775221494f231f7ee7835f231ca4fe4.png" class="math-inline" /> を考えます。  
次の集合族は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の直和分割になっているかどうか判定し、理由も述べてください。

1. <img src="tmp/8edab61d51a9bd839b6e48990aa4d6cc.png" class="math-inline" />
2. <img src="tmp/293c67970695a381a12d344aca09d917.png" class="math-inline" />
3. <img src="tmp/ff84e928b13e62ff96f4f2b3a3d3daf3.png" class="math-inline" />

__問2（同値関係の定義）__

整数全体 <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> において、次の関係 <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> は同値関係かどうか判定し、理由も述べてください。

1. <img src="tmp/478c8e2df02ea0addfe0b31ebdcaa38c.png" class="math-inline" /> は偶数
2. <img src="tmp/1ab9c5bf2dea4a324d565465a07e1d51.png" class="math-inline" />
3. <img src="tmp/7e09fd5d4e2a7a994bb780a54f6eaa6c.png" class="math-inline" />

__問3（同値類と商集合）__

整数全体 <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> において、関係  


<div class="math-display-container"><img src="tmp/cbbc85da6d0277823c81fc93168564b9.png" class="math-display" /></div>


は同値関係です（mod 3 の合同関係）。

1. 同値類 <img src="tmp/d758cf20bc4d74ce75f8810eb43f9fce.png" class="math-inline" /> を具体的に集合として書き下してください。
2. 商集合 <img src="tmp/0f38325fd4ddb7415e3e342c9595fcb9.png" class="math-inline" /> を書き下してください。

__問4（同値関係の強さ）__

整数全体 <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> において、次の2つの同値関係を考えます。

- <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" />：<img src="tmp/112b53f392022573a73d7b1a3c74c068.png" class="math-inline" /> は 2 の倍数
- <img src="tmp/9c37ec2d90fb571b537bec06863bb97d.png" class="math-inline" />：<img src="tmp/c55c371de83e5c37dc1b1b487d4f9077.png" class="math-inline" /> は 4 の倍数

1. <img src="tmp/9c37ec2d90fb571b537bec06863bb97d.png" class="math-inline" /> は <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> より強い（細かい）か、弱い（粗い）か、あるいは比較不能か、理由とともに答えてください。
2. 同値類 <img src="tmp/1f9e11e9d0d8bdbb71011504521247da.png" class="math-inline" /> と <img src="tmp/2bff1cbb7328aba07bf0fad36b1e2fb8.png" class="math-inline" /> の包含関係を調べ、どのように「細かい／粗い」が現れているか説明してください。

__問5（被覆）__

集合 <img src="tmp/a550ae2978e7fbc854d0854b920cd24c.png" class="math-inline" /> を考えます。次の集合族は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の被覆になっているかどうか判定し、理由も述べてください。

1. <img src="tmp/910ca2ef4081ba958c45828b19aa329f.png" class="math-inline" />
2. <img src="tmp/bf88512208a488e71f240eb0ed27e539.png" class="math-inline" />
3. <img src="tmp/751841b7d294be56eda4d735ffee7879.png" class="math-inline" />

__問6（被覆と直和分割の関係）__

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合の族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> が <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**直和分割**であるとき、それは必ず <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**被覆**でもあると言えるかどうか、理由とともに答えてください。

### 解答

以下、解答と簡単な解説です。

__問1（直和分割）__

**直和分割の条件**：
1. 和集合が <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 全体になる：<img src="tmp/01e4c25671e9297e7cbe6e864927a827.png" class="math-inline" />
2. どの2つも交わらない：<img src="tmp/2cf2380a2bd26ce42fd5c837d2467040.png" class="math-inline" />

__1.__ 

<img src="tmp/8edab61d51a9bd839b6e48990aa4d6cc.png" class="math-inline" />
- 和集合：<img src="tmp/b7c9fc0ed21e46a4bf0589a0cd85f9e8.png" class="math-inline" />
- どの2つも交わらない：<img src="tmp/30341deb5f5d3cd7576904bbc5d31cd2.png" class="math-inline" />

→ **直和分割である**。

__2.__ 

<img src="tmp/293c67970695a381a12d344aca09d917.png" class="math-inline" />
- 和集合：<img src="tmp/f7139330bb06edd7db3e3d0a1403352a.png" class="math-inline" />
- しかし、<img src="tmp/2047cce751e6e740b091596f2cfb90c4.png" class="math-inline" />、<img src="tmp/e81c4d7aed04142023dfdcf01e52c088.png" class="math-inline" />

→ **直和分割ではない**（互いに素でない）。

__3.__ 

<img src="tmp/ff84e928b13e62ff96f4f2b3a3d3daf3.png" class="math-inline" />
- 和集合：<img src="tmp/4a2606a22f1e688a7886128b5252e796.png" class="math-inline" />
- どの2つも交わらない（1要素集合同士も当然交わらない）

→ **直和分割である**。

__問2（同値関係の定義）__

同値関係の条件：反射律・対称律・推移律。

__1.__ 

<img src="tmp/478c8e2df02ea0addfe0b31ebdcaa38c.png" class="math-inline" /> は偶数
- 反射律：<img src="tmp/90933e3e57e49fc57d990261547e00d9.png" class="math-inline" /> は偶数 → <img src="tmp/8a7e207de9b21e03f15828b5c1f0c59e.png" class="math-inline" />
- 対称律：<img src="tmp/f2d7c0906b3a89c8f04112dcc2760dbc.png" class="math-inline" /> が偶数なら <img src="tmp/91601778c1f6c9fe86978e7bfb3e49c6.png" class="math-inline" /> も偶数 → <img src="tmp/2ec2ef9df6cbd65190af0c22804c5441.png" class="math-inline" />
- 推移律：<img src="tmp/f2d7c0906b3a89c8f04112dcc2760dbc.png" class="math-inline" /> と <img src="tmp/4cdad60c1d1808d0537a1b8b8d59faaf.png" class="math-inline" /> が偶数なら、<img src="tmp/2421b05945831f901f7ba0210cf8881d.png" class="math-inline" /> も偶数 → <img src="tmp/282a27126242cbc3ed0d34207676fd95.png" class="math-inline" /> かつ <img src="tmp/1d012d2fe96c328e3191171601992312.png" class="math-inline" />

→ **同値関係である**（偶奇による mod 2 の合同関係）。

__2.__ 

<img src="tmp/1ab9c5bf2dea4a324d565465a07e1d51.png" class="math-inline" />
- 反射律：<img src="tmp/eca18b4797b9da05b2b0dd8f0b47cb47.png" class="math-inline" /> は偽 → <img src="tmp/49cf6b1ad7904ebed84ddd985011e87c.png" class="math-inline" />
- 対称律：<img src="tmp/908fe3a938b201138192a7612e1fa289.png" class="math-inline" /> なら <img src="tmp/b8581ad6c73c79ac00088f9f4e87935b.png" class="math-inline" /> は偽 → 対称律を満たさない

→ **同値関係ではない**（反射律・対称律が不成立）。

__3.__ 

<img src="tmp/7e09fd5d4e2a7a994bb780a54f6eaa6c.png" class="math-inline" />
- 反射律：<img src="tmp/7cfadb4a02af8f859b044eaef36b539d.png" class="math-inline" /> → <img src="tmp/8a7e207de9b21e03f15828b5c1f0c59e.png" class="math-inline" />
- 対称律：<img src="tmp/87a0dc8955825e9be72ca52967517a4f.png" class="math-inline" /> → <img src="tmp/2ec2ef9df6cbd65190af0c22804c5441.png" class="math-inline" />
- 推移律：<img src="tmp/cda4af2e6a0bf2d8cdf56797bc27b344.png" class="math-inline" /> かつ <img src="tmp/2d85ff7a00be48142bec67365af4b453.png" class="math-inline" /> → 推移律も成立

→ **同値関係である**（絶対値が等しい元同士を同一視）。

__問3（同値類と商集合）__

関係：<img src="tmp/478c8e2df02ea0addfe0b31ebdcaa38c.png" class="math-inline" /> は 3 の倍数。

__1.__ 

同値類 <img src="tmp/d758cf20bc4d74ce75f8810eb43f9fce.png" class="math-inline" />
- <img src="tmp/7b1da6472fb3238858d12339e01ce312.png" class="math-inline" />
- <img src="tmp/6daa88d8bd2d6e6c7d5ca9ec2cef6912.png" class="math-inline" />
- <img src="tmp/b34f5575566a36228a974a95f5cdc6e1.png" class="math-inline" />

__2.__ 

商集合 <img src="tmp/0f38325fd4ddb7415e3e342c9595fcb9.png" class="math-inline" />


<div class="math-display-container"><img src="tmp/7eb9fb553187a6d2c18efb8690adb3f6.png" class="math-display" /></div>


（3で割った余りが 0, 1, 2 の3つの同値類からなる集合）

__问4（同値関係の強さ）__

__1.__ 

<img src="tmp/9c37ec2d90fb571b537bec06863bb97d.png" class="math-inline" /> と <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> の強さの比較
- <img src="tmp/9c37ec2d90fb571b537bec06863bb97d.png" class="math-inline" />：<img src="tmp/c55c371de83e5c37dc1b1b487d4f9077.png" class="math-inline" /> は 4 の倍数
- <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" />：<img src="tmp/112b53f392022573a73d7b1a3c74c068.png" class="math-inline" /> は 2 の倍数

<img src="tmp/800f033cf3c2e0b35e35385521b458d0.png" class="math-inline" /> は 4 の倍数 <img src="tmp/bb94cf00d1abfeea2e50677d8704cb49.png" class="math-inline" /> は 2 の倍数 <img src="tmp/a172503ba7c08b43f548d4785db257a6.png" class="math-inline" />  
したがって、


<div class="math-display-container"><img src="tmp/97b2bad8db29f2278202be9f0b2ed459.png" class="math-display" /></div>


が成り立つので、<img src="tmp/9c37ec2d90fb571b537bec06863bb97d.png" class="math-inline" /> は <img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" /> より**強い（細かい）**。

__2.__ 

<img src="tmp/1f9e11e9d0d8bdbb71011504521247da.png" class="math-inline" /> と <img src="tmp/2bff1cbb7328aba07bf0fad36b1e2fb8.png" class="math-inline" /> の包含関係
- <img src="tmp/da1a363319cb629d59474ba84403405e.png" class="math-inline" />
- <img src="tmp/1eaaa72752f75ab389538af2ef05c21d.png" class="math-inline" />

明らかに <img src="tmp/6769a57203a02a3681afa8fec61c7a3e.png" class="math-inline" /> です。  
強い方（<img src="tmp/9c37ec2d90fb571b537bec06863bb97d.png" class="math-inline" />）の同値類が、弱い方（<img src="tmp/d814e960e8c43ae4e70215f6fc910020.png" class="math-inline" />）の同値類に**含まれる**形になっています。  
つまり、強い方の分割はより細かく、弱い方はそれを「まとめた」粗い分割になっています。

__問5（被覆）__

**被覆の条件**：和集合が <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 全体を含むこと（<img src="tmp/06409d719e1d176ce836df21c2f5fa55.png" class="math-inline" />）。

__1.__

<img src="tmp/910ca2ef4081ba958c45828b19aa329f.png" class="math-inline" />
- 和集合：<img src="tmp/ad86d35f9f9e7dc5d4bb4076c76b7d47.png" class="math-inline" />

→ **被覆である**（実際には直和分割でもある）。

__2.__ 

<img src="tmp/bf88512208a488e71f240eb0ed27e539.png" class="math-inline" />
- 和集合：<img src="tmp/3a5235a77617702aec9d5c804deb4094.png" class="math-inline" />

→ **被覆である**（ただし直和分割ではない：<img src="tmp/59514e3c8b563905308e8607ab9c6f32.png" class="math-inline" />）。

__3.__ 

<img src="tmp/751841b7d294be56eda4d735ffee7879.png" class="math-inline" />
- 和集合：<img src="tmp/ef8f459ed411cba3307805946e6f7693.png" class="math-inline" />  
  要素 3 が含まれないので、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 全体を覆っていない。

→ **被覆ではない**。

__問6（被覆と直和分割の関係）__

**主張**：集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合の族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> が <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**直和分割**であるとき、それは必ず <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**被覆**でもあると言えるか？

**答え**：**はい、必ず被覆でもある**。

**理由**：
- 直和分割の定義から、  
  

<div class="math-display-container"><img src="tmp/b4406c2538a44900b5c12a3899dea37b.png" class="math-display" /></div>


  が成り立ちます。
- 被覆の定義は「和集合が <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 全体を含む」ことなので、この条件は満たされています。
- 直和分割は「被覆＋互いに素」という条件なので、被覆であることは自動的に含まれます。

したがって、直和分割であれば必ず被覆です（逆は一般には成り立ちません：被覆でも直和分割でない例は問5(2)など）。

<div style="page-break-before:always"></div>


# 集合の対等

## 対等

集合論における**対等（equinumerous）** とは、**2つの集合の要素の個数が「同じ」であること**を厳密に表す概念です。  
有限集合なら「要素数が等しい」ことに対応し、無限集合にも自然に拡張できます。


### 1. 対等の定義

2つの集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> が**対等**であるとは、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への**全単射（bijection）** が存在することをいいます。

記号では、


<div class="math-display-container"><img src="tmp/7fd52966b4d7cf9bc33df0f21c5855f0.png" class="math-display" /></div>


と書くことがあります（ここでの <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> は同値関係の記号とは別物ですが、文脈で区別します）。

より正確に書くと：

- 写像 <img src="tmp/18766859ca7c11d2de922641bfec7ae1.png" class="math-inline" /> が存在して、
  - **単射**：<img src="tmp/95eeefe7834f4827f831f9b47340281f.png" class="math-inline" />
  - **全射**：任意の <img src="tmp/1c7b9d06bbb9ff1ebb4d49f09efce87f.png" class="math-inline" /> に対し、ある <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> が存在して <img src="tmp/446c2f591e2c7f9282046b67f22723e2.png" class="math-inline" />

このような <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> が存在するとき、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は**対等**であるといいます。


### 2. 具体例

__例1：有限集合__
- <img src="tmp/c9ada39e81417f791e82174ed94a18d8.png" class="math-inline" />
- <img src="tmp/07ac5cef2d6506f2a0968e425fcaf31c.png" class="math-inline" /> は全単射
- よって <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は対等（どちらも要素数 3）

__例2：無限集合__
- <img src="tmp/525dc8519742e5a21dd862a038661e1f.png" class="math-inline" />（正の偶数の集合）
- <img src="tmp/48c7fe82db4d8f85bc6f63469efcce92.png" class="math-inline" /> は <img src="tmp/90a3668c129c3ebee169b6d5f6ae1a28.png" class="math-inline" /> の全単射
- よって <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は対等（どちらも可算無限）

__例3：対等でない例__
- <img src="tmp/96f28457ab748480ca3cd37a4097a413.png" class="math-inline" />
- <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への全単射は存在しない（要素数が異なる）
- よって <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は対等ではない

### 3. 対等と濃度（cardinality）
この部分は後ほど詳細に扱っていきます。
関連する内容ということでご承知おきください。

- 対等である集合は、**同じ濃度（cardinality）を持つ**といいます。
- 有限集合の場合、濃度は単に「要素の個数」です。
- 無限集合の場合も、対等であるかどうかで「大きさが同じか」を判断します（例：自然数全体と有理数全体は対等、実数全体はそれらより「大きい」）。


### 4. 注意点

- 「対等」は**集合のサイズ（濃度）が同じ**という意味であり、集合が「等しい」とは別概念です。
  - 例：<img src="tmp/7952a039a96398415c8e6161bcd97f66.png" class="math-inline" /> と <img src="tmp/4f7b41f2ab308327d5a2cd5ba5a8d4bc.png" class="math-inline" /> は等しくないが、対等である。
- 対等関係は、集合全体のクラス上の**同値関係**になります（反射律・対称律・推移律を満たす）。

## 濃度

**濃度（cardinality）** とは、集合の「大きさ」や「要素の個数」を表す数学的な概念です。  
有限集合では単に「要素の数」に対応し、無限集合にも自然に拡張されます。


### 1. 濃度の直感的な意味

- 有限集合の場合：  
  集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の濃度は、その中に含まれる要素の個数です。  
  例：<img src="tmp/07cc1897e2aa76f97cf2f059400351ff.png" class="math-inline" /> の濃度は 3、記号で <img src="tmp/892eaef769c0f4c9a24b108e22014a36.png" class="math-inline" /> などと書きます。

- 無限集合の場合：  
  「要素の数」を直接数えることはできませんが、**他の集合との対等関係（全単射の有無）** を通じて「大きさが同じか・どちらが大きいか」を比較します。


### 2. 濃度の厳密な定義（対等による）

2つの集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> が**対等（equinumerous）** であるとは、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への**全単射**が存在することをいいます（前の回答で説明した通り）。

このとき、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は**同じ濃度を持つ**と定義します。

- 有限集合の場合：  
  要素数が等しければ全単射が作れるので、「濃度が同じ」＝「要素数が同じ」。

- 無限集合の場合：  
  例：自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と偶数全体は全単射 <img src="tmp/956e10b149367ff8642c84f4ca342f92.png" class="math-inline" /> で結べるので、同じ濃度（可算無限）です。


### 3. 濃度の記号と例

- 有限集合の濃度：自然数 <img src="tmp/18991571b007a883e99c3893b3740bdf.png" class="math-inline" /> で表す。  
  例：<img src="tmp/12ac6e3b1da5a40228077c84f43ad64f.png" class="math-inline" />

- 可算無限濃度：<img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />（アレフ・ゼロ）  
  - 自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" />、整数全体 <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />、有理数全体 <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> などがこの濃度を持つ。

- 連続体濃度：<img src="tmp/0a2f3f25776381173161ff6d7ffeb331.png" class="math-inline" />  
  - 実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" />、区間 <img src="tmp/039474a1a493fb3c2d8f4adbf24f971e.png" class="math-inline" /> などがこの濃度を持つ。  
  - 可算無限より「大きい」ことが知られています（カントールの対角線論法）。

### 4. 濃度の比較

集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> について：

- <img src="tmp/46c4dd564b86222198c1bd3d0f7100f9.png" class="math-inline" />：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の間に全単射が存在する（対等）。
- <img src="tmp/aaa17504be8f59cf8a0e9609d3d6bf03.png" class="math-inline" />：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への単射が存在する（<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> より大きくない）。
- <img src="tmp/cdf89d42f0059ea8c63230b802884b42.png" class="math-inline" />：<img src="tmp/aaa17504be8f59cf8a0e9609d3d6bf03.png" class="math-inline" /> だが <img src="tmp/9e6bd43e02ea4b6ecd6932084ac781de.png" class="math-inline" />（<img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の方が真に大きい）。

このように、濃度は**全単射・単射の存在**を通じて比較されます。


### 5. なぜ濃度が必要か

- 有限集合では「個数」で十分ですが、無限集合では「個数を数える」ことができません。
- そこで、**全単射の有無**を「大きさが同じか」の基準にすることで、無限集合のサイズを厳密に比較できます。
- これにより、「自然数と有理数は同じくらい多いが、実数はもっと多い」といったことが数学的に証明できます。


## 濃度の大小

**濃度の大小**とは、集合の「大きさ」を比較するための数学的な関係です。  
有限集合では単に「要素数が多い／少ない」に対応し、無限集合にも自然に拡張されます。

### 1. 濃度の大小の定義

2つの集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> について、濃度 <img src="tmp/2335a214056244469e5092738bfe7d89.png" class="math-inline" /> の大小は次のように定義されます。

__(1) <img src="tmp/aaa17504be8f59cf8a0e9609d3d6bf03.png" class="math-inline" />__
- 定義：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への**単射（injection）** が存在するとき、<img src="tmp/aaa17504be8f59cf8a0e9609d3d6bf03.png" class="math-inline" /> と書く。
- 意味：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の要素を <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の要素に「重複なく」対応づけられる  
  → <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の「大きさ」は <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> より大きくない（<img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の方が同じかそれ以上）。

__(2) <img src="tmp/46c4dd564b86222198c1bd3d0f7100f9.png" class="math-inline" />__  
- 定義：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への**全単射（bijection）** が存在するとき、<img src="tmp/46c4dd564b86222198c1bd3d0f7100f9.png" class="math-inline" /> と書く。
- 意味：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は**対等**（equinumerous）で、同じ濃度を持つ。

__(3) <img src="tmp/cdf89d42f0059ea8c63230b802884b42.png" class="math-inline" />__
- 定義：<img src="tmp/aaa17504be8f59cf8a0e9609d3d6bf03.png" class="math-inline" /> だが <img src="tmp/9e6bd43e02ea4b6ecd6932084ac781de.png" class="math-inline" /> のとき、<img src="tmp/cdf89d42f0059ea8c63230b802884b42.png" class="math-inline" /> と書く。
- 意味：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への単射はあるが、全単射は存在しない  
  → <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の方が**真に大きい**濃度を持つ。

### 2. 具体例

__例1：有限集合__
- <img src="tmp/96f28457ab748480ca3cd37a4097a413.png" class="math-inline" />
- <img src="tmp/d3a38460eab6d642235acbbd5bcb8690.png" class="math-inline" /> の単射は存在する（例：<img src="tmp/4e55d02058eb7ced87be7e47d2249875.png" class="math-inline" />）
- しかし全単射は存在しない（要素数が異なる）
- よって <img src="tmp/cdf89d42f0059ea8c63230b802884b42.png" class="math-inline" />

__例2：無限集合（可算無限と連続体濃度）__
- <img src="tmp/024d16c0e3fdfd485f9bd70c7df42ce1.png" class="math-inline" />（可算無限）
- <img src="tmp/45db033be1b7b63527830e74e43579b7.png" class="math-inline" />（連続体濃度）
- <img src="tmp/39a463630d85c3be0bd2f7745e08c8f8.png" class="math-inline" /> の単射は明らかに存在する（自然数をそのまま実数とみなす）
- しかしカントールの対角線論法により、<img src="tmp/39a463630d85c3be0bd2f7745e08c8f8.png" class="math-inline" /> の全単射は存在しないことが知られている
- よって <img src="tmp/d802509ea469d1ac38885691297ef6c3.png" class="math-inline" />

__例3：同じ無限濃度__
- <img src="tmp/05bedd17488afc9fbca66cd269648904.png" class="math-inline" />  
  （自然数・整数・有理数はすべて可算無限で、互いに全単射が構成できる）

### 3. 濃度の大小が持つ数学的意味

__(1) 「大きさ」の厳密な比較__
- 有限集合では「個数」で比較できますが、無限集合では「個数を数える」ことができません。
- そこで、**単射・全単射の存在**を「大きさが同じか・どちらが大きいか」の基準にすることで、無限集合のサイズを厳密に比較できます。

__(2) 無限の階層の存在を示す__
- 自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> は、どちらも無限ですが、<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の方が「真に大きい」ことが証明できます。
- さらに、任意の集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> に対して、そのべき集合 <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> は <img src="tmp/bbe53ea0985487882e8fe1d73d29fd4e.png" class="math-inline" /> を満たします（カントールの定理）。  
  これにより、**無限に大きい濃度の階層**が存在することがわかります。

__(3) 同値関係・順序関係としての構造__
- 「<img src="tmp/46c4dd564b86222198c1bd3d0f7100f9.png" class="math-inline" />」は対等関係であり、反射律・対称律・推移律を満たす**同値関係**です。
- 「<img src="tmp/aaa17504be8f59cf8a0e9609d3d6bf03.png" class="math-inline" />」は反射的・推移的であり、濃度全体のクラス上の**前順序（preorder）** になります（反対称性も成り立ち、実際には**全順序的な性質**を持ちますが、選択公理を仮定すると整列可能定理により「すべての濃度は比較可能」とみなせます）。

### 4. なぜこの定義なのか（直感との一致）

- 有限集合の場合：  
  <img src="tmp/aaa17504be8f59cf8a0e9609d3d6bf03.png" class="math-inline" /> は「<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の要素数 ≤ <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の要素数」と完全に一致します。
- 無限集合の場合：  
  「<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> を <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> に埋め込める（単射がある）」なら、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> より大きくない、という直感を数学的に表現したものです。

この定義により、**有限・無限を問わず一貫した「大きさ」の比較**が可能になります。

### 5. Cantor-Bersteinの定理

**Cantor–Bernsteinの定理**（シュレーダー–ベルンシュタインの定理とも呼ばれます）は、**濃度の比較に関する基本的で強力な定理**です。

__1. 定理の主張__

2つの集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> について、

- <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への**単射**が存在し、
- <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> から <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> への**単射**が存在する

ならば、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への**全単射**が存在する、すなわち


<div class="math-display-container"><img src="tmp/a99c01ae90b09ede83ac002b0c7ed682.png" class="math-display" /></div>


が成り立つ、というのが定理の主張です。

記号で書くと：


<div class="math-display-container"><img src="tmp/31e82312da5012223b9cdf85bd413948.png" class="math-display" /></div>



__2. 直感的な意味__

- <img src="tmp/aaa17504be8f59cf8a0e9609d3d6bf03.png" class="math-inline" />：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> を <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> に「埋め込める」 → <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> より大きくない
- <img src="tmp/d373dfe3cbf50ac6f699126718368531.png" class="math-inline" />：<img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> を <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に「埋め込める」 → <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> より大きくない

この2つが同時に成り立つなら、**どちらかが真に大きいということはあり得ず、結局は同じ大きさ（同じ濃度）になる**、というのが定理の内容です。

有限集合では当たり前に感じられますが、無限集合でも成り立つことが重要です。

__3. 具体例__

__例：開区間 <img src="tmp/039474a1a493fb3c2d8f4adbf24f971e.png" class="math-inline" /> と閉区間 <img src="tmp/171d58b8063a24dc65a33bacb46a5fb7.png" class="math-inline" />__

- <img src="tmp/2d0feefcecce3c2d791ccd49c319ff23.png" class="math-inline" /> なので、包含写像により単射 <img src="tmp/b8ff1d5355907fc6d79070913d1dc126.png" class="math-inline" /> が存在  
  → <img src="tmp/558c85caa0128574499b53bcdaf4cf78.png" class="math-inline" />
- 一方、<img src="tmp/984d5c7a625959f11a70f581c0de1118.png" class="math-inline" /> の単射も構成できます（例：<img src="tmp/d874cc20e6dd25616cab7dba4700c90d.png" class="math-inline" />）  
  → <img src="tmp/30866dae3e51259b01c2ab10e07ac948.png" class="math-inline" />

Cantor–Bernsteinの定理より、


<div class="math-display-container"><img src="tmp/a892a7e72a7841b3ac9571ead36a8128.png" class="math-display" /></div>


つまり、開区間と閉区間は**同じ濃度（連続体濃度）** を持つことがわかります。

__4. なぜ重要なのか__

1. **濃度の順序関係が「反対称的」になる**  
   <img src="tmp/aaa17504be8f59cf8a0e9609d3d6bf03.png" class="math-inline" /> かつ <img src="tmp/d373dfe3cbf50ac6f699126718368531.png" class="math-inline" /> なら <img src="tmp/46c4dd564b86222198c1bd3d0f7100f9.png" class="math-inline" /> が成り立つので、濃度の比較は**半順序**（実際には全順序的な性質）を持ちます。

2. **全単射を直接構成しなくても濃度が等しいとわかる**  
   単射を2つ見つけるだけで「同じ濃度」と結論できるため、**全単射を具体的に構成する手間を省ける**ことが多いです。

3. **無限集合のサイズ比較の基礎**  
   可算無限・連続体濃度など、さまざまな無限の大きさを比較する際の基本的な道具になります。

__5. 証明__

以下、Cantor–Bernsteinの定理の証明を示します。

---

__証明の準備__

- 単射 <img src="tmp/18766859ca7c11d2de922641bfec7ae1.png" class="math-inline" />、単射 <img src="tmp/73c2f0b6e20ec26e8fbf5771ffb7c389.png" class="math-inline" /> が与えられているとします。
- 目標：全単射 <img src="tmp/5db05b173c24f6642712dd700182b10e.png" class="math-inline" /> を構成する。

証明のアイデアは、**<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> を2つの部分に分け、片方では <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> を、もう片方では <img src="tmp/f9d9a07414833b4ee7408dd0ff6f70bc.png" class="math-inline" /> を使う**ことです。  
ただし <img src="tmp/ce689b6baffcda72072471af44754299.png" class="math-inline" /> は単射であっても全射とは限らないので、「逆写像」はそのままでは定義できません。そこで、**逆像**を使って慎重に扱います。

__証明のステップ__

__ステップ1：写像 <img src="tmp/dd37be2f03ec754bda0ed31f5348f1ba.png" class="math-inline" /> の定義__

<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の部分集合 <img src="tmp/62257db6fcfcc679dfd1432de9e1525d.png" class="math-inline" /> に対して、


<div class="math-display-container"><img src="tmp/133da950a287bfe374f2c30e6387b6fe.png" class="math-display" /></div>


と定義します。ここで、

- <img src="tmp/879d42c69cad9727b6fb4ab90d064474.png" class="math-inline" />
- <img src="tmp/6c57cf195c10067fb562b316a2873de2.png" class="math-inline" />：<img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> のうち <img src="tmp/8bf2f372a47d22e5724adf03deac90ad.png" class="math-inline" /> に含まれない部分
- <img src="tmp/4d0398f653f5efb6bd2513545c33e71e.png" class="math-inline" />
- <img src="tmp/b94612b79e9b73489425131ce473dc0b.png" class="math-inline" />：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> のうち、<img src="tmp/b1ade167f65c693a66250c969f5f29e9.png" class="math-inline" /> に含まれない部分

つまり、<img src="tmp/549a532aeb6a33fc32a92e0a711faf51.png" class="math-inline" /> は「<img src="tmp/8bf2f372a47d22e5724adf03deac90ad.png" class="math-inline" /> に写されなかった <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の元を <img src="tmp/ce689b6baffcda72072471af44754299.png" class="math-inline" /> で <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に戻したもの」の**補集合**です。

この <img src="tmp/f4feb358fd4f859b1f04ca157e8c3be8.png" class="math-inline" /> は**単調増加**です：


<div class="math-display-container"><img src="tmp/0a80763a21889a77c51642485065963b.png" class="math-display" /></div>


（証明略：包含関係を追えば確認できます）

__ステップ2：不動点の存在__

単調増加写像 <img src="tmp/dd37be2f03ec754bda0ed31f5348f1ba.png" class="math-inline" /> に対し、次の集合を考えます：


<div class="math-display-container"><img src="tmp/0de71c4b977217a8dcdbd4bb69ce9557.png" class="math-display" /></div>


（「<img src="tmp/680ffea20bb1191d156f50c730ace733.png" class="math-inline" /> を満たすすべての <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の和集合」）

このとき、次が成り立ちます：

1. <img src="tmp/d40e1f83b028a3a2837c4e1b93f7ca1f.png" class="math-inline" />
2. <img src="tmp/fedc379e4e45d6f28c0f9c74843539e2.png" class="math-inline" />

したがって、<img src="tmp/a3becf6a07d2e71ce6a9b8dc61ed6ff9.png" class="math-inline" /> となり、<img src="tmp/7da569b3f6d6a9ad1defdb55cccf0f5f.png" class="math-inline" /> は <img src="tmp/f4feb358fd4f859b1f04ca157e8c3be8.png" class="math-inline" /> の**不動点**です。

（証明の概略：単調性と和集合の定義から、包含関係を両方向に示せます。詳細は集合論の教科書を参照してください。）

__ステップ3：全単射 <img src="tmp/5db05b173c24f6642712dd700182b10e.png" class="math-inline" /> の構成__

不動点 <img src="tmp/678311a930ea5760e815ec2ffb03f81b.png" class="math-inline" /> に対して、次のように <img src="tmp/5db05b173c24f6642712dd700182b10e.png" class="math-inline" /> を定義します：


<div class="math-display-container"><img src="tmp/ef60190c2d589db90548f30b94c437a9.png" class="math-display" /></div>



ここで、<img src="tmp/8e4ca2cbca8f005a31aa453f02694cad.png" class="math-inline" /> は「<img src="tmp/80dd932a0edea1bb9557d89013935230.png" class="math-inline" /> となる <img src="tmp/df42cc6c7de0e8ac0ad614aa3a1cb5ed.png" class="math-inline" />」の意味です。  
<img src="tmp/ce689b6baffcda72072471af44754299.png" class="math-inline" /> は単射なので、そのような <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は高々1つですが、存在するかどうかが問題です。  
実は、<img src="tmp/68829d5bae4e9ca00166dbf53773c3d0.png" class="math-inline" /> のとき、必ずそのような <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> が存在することが、不動点の条件から導かれます。

より正確に：

- <img src="tmp/19b5ea5df7a14ab69f8d0e7860b65c11.png" class="math-inline" />
- よって <img src="tmp/1e9233086fe05504c1c96bdc7435c82e.png" class="math-inline" />
- したがって、任意の <img src="tmp/68829d5bae4e9ca00166dbf53773c3d0.png" class="math-inline" /> に対し、ある <img src="tmp/d82c2b7c5e116cf34edb9f73044ecba7.png" class="math-inline" /> が一意に存在して <img src="tmp/80dd932a0edea1bb9557d89013935230.png" class="math-inline" />
- この <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> を <img src="tmp/8e4ca2cbca8f005a31aa453f02694cad.png" class="math-inline" /> と書く

この定義により、<img src="tmp/1e7b05d73158083a7c4dbc84cbead35a.png" class="math-inline" /> は well-defined です。

__ステップ4：<img src="tmp/1e7b05d73158083a7c4dbc84cbead35a.png" class="math-inline" /> が全単射であることの確認__

**(a) 単射であること**

- <img src="tmp/907ebd0782af0294c0917742ced44109.png" class="math-inline" /> で <img src="tmp/b4cf181ca82785c2a99ff11b0b99ab58.png" class="math-inline" /> と仮定します。
- <img src="tmp/3ef38f84c23bda0b5445b19f87776622.png" class="math-inline" /> がともに <img src="tmp/7da569b3f6d6a9ad1defdb55cccf0f5f.png" class="math-inline" /> に属する場合：  
  <img src="tmp/5cccbfa8b7eff28ed0cc13687feabfc2.png" class="math-inline" /> で、<img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> は単射なので <img src="tmp/70e997ed2e3dd20c1838e199543accba.png" class="math-inline" />。
- <img src="tmp/3ef38f84c23bda0b5445b19f87776622.png" class="math-inline" /> がともに <img src="tmp/89b032cdaa0b17da44bc09671c89d2ab.png" class="math-inline" /> に属する場合：  
  <img src="tmp/a86f0d738725c6633903366d2eb86c33.png" class="math-inline" /> で、<img src="tmp/ce689b6baffcda72072471af44754299.png" class="math-inline" /> は単射なので <img src="tmp/f9d9a07414833b4ee7408dd0ff6f70bc.png" class="math-inline" /> も単射的 → <img src="tmp/70e997ed2e3dd20c1838e199543accba.png" class="math-inline" />。
- 一方が <img src="tmp/7da569b3f6d6a9ad1defdb55cccf0f5f.png" class="math-inline" />、他方が <img src="tmp/89b032cdaa0b17da44bc09671c89d2ab.png" class="math-inline" /> に属する場合は起こり得ません。  
  なぜなら、<img src="tmp/854b8ecba3d72079a42b6580cfd85e29.png" class="math-inline" />、<img src="tmp/56c99502f88c9d8328d9109afd69ca9e.png" class="math-inline" /> であり、  
  <img src="tmp/bcff0cd09d358dfcbc36d0967c66221b.png" class="math-inline" /> だからです。

よって <img src="tmp/1e7b05d73158083a7c4dbc84cbead35a.png" class="math-inline" /> は単射。

**(b) 全射であること**

任意の <img src="tmp/df42cc6c7de0e8ac0ad614aa3a1cb5ed.png" class="math-inline" /> を取ります。

- もし <img src="tmp/87c4acd3afcd2a43128f3f3009620f51.png" class="math-inline" /> なら、ある <img src="tmp/e456c2b4ad127b7e6806bf6588c0a83b.png" class="math-inline" /> が存在して <img src="tmp/ad5636b84e80af2c933cc5080d40a667.png" class="math-inline" />。このとき <img src="tmp/648669fb24ba8360fd404ecc8cb5973f.png" class="math-inline" />。
- もし <img src="tmp/d82c2b7c5e116cf34edb9f73044ecba7.png" class="math-inline" /> なら、<img src="tmp/035c420daec49c300ed1d9c68e54042c.png" class="math-inline" />。  
  このとき <img src="tmp/7dc58d513c17e1007253b859c4ec4e5d.png" class="math-inline" />。

したがって、任意の <img src="tmp/df42cc6c7de0e8ac0ad614aa3a1cb5ed.png" class="math-inline" /> に対して <img src="tmp/aa24017d263de81c4707472d0cc8c095.png" class="math-inline" /> となる <img src="tmp/91aa48b65b7d642d9603ae0cdd3cc302.png" class="math-inline" /> が存在するので、<img src="tmp/1e7b05d73158083a7c4dbc84cbead35a.png" class="math-inline" /> は全射。

以上より、<img src="tmp/5db05b173c24f6642712dd700182b10e.png" class="math-inline" /> は全単射です。

__結論__

単射 <img src="tmp/18766859ca7c11d2de922641bfec7ae1.png" class="math-inline" /> と単射 <img src="tmp/73c2f0b6e20ec26e8fbf5771ffb7c389.png" class="math-inline" /> が存在するとき、上記の構成により全単射 <img src="tmp/5db05b173c24f6642712dd700182b10e.png" class="math-inline" /> が得られる。  
したがって、


<div class="math-display-container"><img src="tmp/31e82312da5012223b9cdf85bd413948.png" class="math-display" /></div>


が成り立ち、Cantor–Bernsteinの定理が証明されました。

---

### 6. Cantorの定理

**Cantorの定理**は、**任意の集合に対して、そのべき集合（部分集合全体の集合）の濃度は、元の集合の濃度より真に大きい**ことを主張する定理です。  
無限集合であっても「もっと大きい無限」が存在することを示す、集合論の基本的で重要な結果です。

__1. 定理の主張__

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**べき集合（power set）** を


<div class="math-display-container"><img src="tmp/60f2c1c4172f0ed78c9a83484e82bfc1.png" class="math-display" /></div>


と書きます（<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合全体の集合）。

このとき、Cantorの定理は次のように述べられます：

> 任意の集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> に対し、  


<div class="math-display-container"><img src="tmp/d6267c985a364c57ad1e1e9d8a54b62a.png" class="math-display" /></div>


> が成り立つ。  
> すなわち、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> への**全射は存在しない**。

特に、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が無限集合であっても、<img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> はそれより真に大きい濃度を持ちます。

__2. 直感的な意味__

- 有限集合の場合：  
  <img src="tmp/d46f7b659df64db7200be7157c3ddad9.png" class="math-inline" /> なら <img src="tmp/01166bdaef87f28a0cd5015d527525c4.png" class="math-inline" /> であり、<img src="tmp/4925ac9beb1c8d09bdd00eb932e45a23.png" class="math-inline" /> は明らかです。
- 無限集合の場合：  
  例えば <img src="tmp/6e82299800e7ce873f7e86662e3dd8de.png" class="math-inline" />（自然数全体）とすると、<img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" /> は <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> より真に大きい濃度（連続体濃度以上）を持ちます。  
  つまり、「自然数の部分集合全体」は自然数全体より「はるかに多い」ということです。

この定理により、**無限にも大小の階層があり、いくらでも大きい無限が存在する**ことがわかります。



__3. 証明の概要（対角線論法）__

---

Cantorの定理の証明は、**対角線論法（diagonal argument）** を用います。

__ステップ1：全射が存在しないことを示す__

目標：<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> への**全射は存在しない**ことを示す。  
（全射がなければ、全単射も存在せず、したがって <img src="tmp/bbe53ea0985487882e8fe1d73d29fd4e.png" class="math-inline" />）

背理法で示します。

- 仮定：ある全射 <img src="tmp/c6a89df6b7f7789245cb5adfe18ce1e6.png" class="math-inline" /> が存在すると仮定する。
- このとき、各 <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して <img src="tmp/1aebe0f63b409f59bbc806bf91c5ad20.png" class="math-inline" /> が定まる。

__ステップ2：対角集合の構成__

次の集合 <img src="tmp/24cb8c0badb9ec1fac4ef6cec65fd202.png" class="math-inline" /> を考えます：


<div class="math-display-container"><img src="tmp/e7046516f4b6eadcb29ce82e40bd6c6f.png" class="math-display" /></div>


つまり、<img src="tmp/af178e094b74c7f04f930b0ca331bbf7.png" class="math-inline" /> は「自分自身を要素として含まない <img src="tmp/3fdae9fef92593e4e976f831339ce3a1.png" class="math-inline" /> に対応する <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> 全体」です。

- <img src="tmp/af178e094b74c7f04f930b0ca331bbf7.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合なので、<img src="tmp/460407ed3dc9b2b8994aa869cb6f8375.png" class="math-inline" />。
- <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> は全射なので、ある <img src="tmp/16c4d8899695ea86518b08b4c9d9088b.png" class="math-inline" /> が存在して <img src="tmp/5f089b2f6d468fc62135d4866c307be9.png" class="math-inline" /> となるはずです。

__ステップ3：矛盾の導出__

この <img src="tmp/6f11db88b00387f71e09066eaf1485f8.png" class="math-inline" /> について、次の2通りを考えます。

1. <img src="tmp/c1b7d32acc62a8b10986d5b18a32e098.png" class="math-inline" /> の場合  
   - <img src="tmp/af178e094b74c7f04f930b0ca331bbf7.png" class="math-inline" /> の定義より、<img src="tmp/d4b2f5157c6f5fad105355d2435ed141.png" class="math-inline" />。
   - しかし <img src="tmp/5f089b2f6d468fc62135d4866c307be9.png" class="math-inline" /> なので、<img src="tmp/e56c4f1634385fc45fa844285a491a1f.png" class="math-inline" /> となり矛盾。

2. <img src="tmp/e56c4f1634385fc45fa844285a491a1f.png" class="math-inline" /> の場合  
   - <img src="tmp/af178e094b74c7f04f930b0ca331bbf7.png" class="math-inline" /> の定義より、<img src="tmp/19d68bd72e9266a87ee76b6f489bfe03.png" class="math-inline" />。
   - しかし <img src="tmp/5f089b2f6d468fc62135d4866c307be9.png" class="math-inline" /> なので、<img src="tmp/c1b7d32acc62a8b10986d5b18a32e098.png" class="math-inline" /> となり矛盾。

どちらの場合も矛盾が生じます。  
したがって、最初の仮定「全射 <img src="tmp/c6a89df6b7f7789245cb5adfe18ce1e6.png" class="math-inline" /> が存在する」は誤りです。

__ステップ4：濃度の比較__

- 包含写像 <img src="tmp/082be666b7331967ecc446b5c898de12.png" class="math-inline" /> は <img src="tmp/24177eee36cbc03bdc4939c4c3c7a4e6.png" class="math-inline" /> の単射なので、<img src="tmp/866c7d95cf55649605be9d6179fb6aa3.png" class="math-inline" />。
- 一方、全射は存在しないので、全単射も存在せず、<img src="tmp/dd6845e29c6c6125f4c1c9910b930bd4.png" class="math-inline" />。

よって、


<div class="math-display-container"><img src="tmp/d6267c985a364c57ad1e1e9d8a54b62a.png" class="math-display" /></div>


が成り立ちます。

---

__4. 定理の意義__

1. **無限の階層の存在**  
   - 自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> に対して、<img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" /> はより大きい濃度を持つ。
   - さらに <img src="tmp/70a7a79a47870995c050d6aee6e72d69.png" class="math-inline" /> はそれより大きく、この操作を繰り返すことで**無限に大きい濃度の列**が得られる。

2. **連続体仮説との関係**  
   - <img src="tmp/024d16c0e3fdfd485f9bd70c7df42ce1.png" class="math-inline" />、<img src="tmp/f55d0b50760106e9e76b5cdc08a5147d.png" class="math-inline" /> であり、これは実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の濃度（連続体濃度）に等しいことが知られています。
   - 連続体仮説は「<img src="tmp/0ed95fbd8a8f7e23c6c080c57a0da502.png" class="math-inline" /> か？」という問いであり、Cantorの定理はその出発点となります。

3. **数学の基礎への影響**  
   - 「すべての集合の濃度は比較可能か？」「より大きい無限が常に存在するか？」といった問題は、集合論の公理（ZFC）や選択公理・連続体仮説と深く結びついています。

## 有限集合、無限集合

**有限集合**と**無限集合**は、集合の「大きさ」を区別する基本的な概念です。  
数学的には、**自然数との全単射の有無**によって定義されます。

### 1. 有限集合の定義

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が**有限集合**であるとは、ある自然数 <img src="tmp/561220853468740bb75d3a76b2451355.png" class="math-inline" /> に対して、


<div class="math-display-container"><img src="tmp/dab8832ce97dddf8c4cb4e804fa6645c.png" class="math-display" /></div>


が成り立つことをいいます。ここで <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> は**対等**（全単射が存在する）を表します。

言い換えると：

- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から <img src="tmp/8b997ccc696c3df57e4702a76cebd9fe.png" class="math-inline" /> への**全単射**が存在する
- このとき、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**濃度（要素数）** は <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> であるといい、<img src="tmp/d46f7b659df64db7200be7157c3ddad9.png" class="math-inline" /> と書く

例：
- <img src="tmp/4aeaf5c0819e1f2c5922c5b5b07561ea.png" class="math-inline" /> は <img src="tmp/7952a039a96398415c8e6161bcd97f66.png" class="math-inline" /> と全単射があるので有限集合、<img src="tmp/9f69280e88971e2a890e09712784aee4.png" class="math-inline" />
- 空集合 <img src="tmp/36682ee60666b219d2be3577c1bfda62.png" class="math-inline" /> は <img src="tmp/ac530716190fa3ec148b9fc37d3fc192.png" class="math-inline" /> に対応し、有限集合とみなす（<img src="tmp/09448b7458b60cd3273616cd22b910f9.png" class="math-inline" />）

### 2. 無限集合の定義

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が**無限集合**であるとは、**有限集合でない**こと、すなわちどの自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> に対しても


<div class="math-display-container"><img src="tmp/913baa5db883b5801382207b45242200.png" class="math-display" /></div>


が成り立つことをいいます。

より実用的な同値定義として：

- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が無限集合  
  <img src="tmp/6a454feba3850900425336c9be01d5bd.png" class="math-inline" /> <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から自身への**単射であって全射でない写像**が存在する  
  <img src="tmp/6a454feba3850900425336c9be01d5bd.png" class="math-inline" /> <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> はある真部分集合と対等（同じ濃度を持つ）

例：
- 自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" />：写像 <img src="tmp/c89bb5a598fabbbbdeadb45ff7505ff4.png" class="math-inline" /> は単射だが全射でない → 無限集合
- 実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" />：同様に無限集合

### 3. 可算無限と非可算無限

無限集合の中でも特に重要な区別があります。

__(1) 可算無限集合（countably infinite）__
- 自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と対等な無限集合
- 例：<img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />（整数全体）、<img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" />（有理数全体）など
- 濃度は <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />（アレフ・ゼロ）と書く

__(2) 非可算無限集合（uncountably infinite）__
- <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と対等でない無限集合
- 例：<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" />（実数全体）、<img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" />（自然数のべき集合）など
- 実数全体の濃度は**連続体濃度** <img src="tmp/0a2f3f25776381173161ff6d7ffeb331.png" class="math-inline" /> と呼ばれる

### 4. 有限集合と無限集合の性質の違い

__(1) 部分集合との関係__
- 有限集合：真部分集合は必ず**より小さい濃度**を持つ  
  （例：<img src="tmp/7952a039a96398415c8e6161bcd97f66.png" class="math-inline" /> の真部分集合の濃度は 0,1,2）
- 無限集合：真部分集合であっても**同じ濃度**を持つことがある  
  （例：<img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と偶数全体は同じ濃度 <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />）

__(2) 和集合・直積の振る舞い__
- 有限集合の有限和・有限直積はまた有限集合
- 無限集合との和集合・直積は多くの場合無限集合（特に可算無限＋有限＝可算無限 など）

__(3) 選択公理との関係__
- 有限集合の族から元を選ぶのは自明
- 無限集合の族から同時に元を選ぶには**選択公理**が必要になる場合が多い

## 可算集合、非可算集合

**可算集合**と**非可算集合**は、無限集合を「どれくらい大きいか」で分類する概念です。  
数学的には、**自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> との全単射の有無**によって定義されます。

### 1. 可算集合（countable set）

__定義__

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が**可算集合**であるとは、次のいずれか（同値）を満たすことをいいます。

1. <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が有限集合である
2. <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と対等である（全単射が存在する）

特に 2 のケースを**可算無限集合**と呼びます。

- 記号：可算無限集合の濃度を <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />（アレフ・ゼロ）と書く
- 意味：要素を <img src="tmp/66548bcfbfcd9b7109a6fe192e58e2f1.png" class="math-inline" /> と「番号付け」できる

__具体例__

- <img src="tmp/44fef71d7ab69623d8fd8c42d9014024.png" class="math-inline" />：自明に可算無限（恒等写像が全単射）
- <img src="tmp/8ee71e1e6524404fb8c4f212503bda9e.png" class="math-inline" />（整数全体）  
  全単射の例：<img src="tmp/3223e8ee9db6c446f19203f5187ecbce.png" class="math-inline" /> など → 可算無限
- <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" />（有理数全体）  
  - 有理数は既約分数 <img src="tmp/f4f4ff9a31e576a624cc206bae56eadb.png" class="math-inline" /> で表せるので、格子点 <img src="tmp/ef7738125e68ffab265fd2ecfd0cc710.png" class="math-inline" /> と対応づけられる  
  - 格子点は自然数と全単射が作れる → 可算無限

### 2. 非可算集合（uncountable set）

__定義__

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が**非可算集合**であるとは、**可算集合でない**こと、すなわち

- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> は有限集合ではなく、
- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> は <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と対等でもない

ことをいいます。

言い換えると：

- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> への全単射は存在しない
- 濃度は <img src="tmp/9c4f266646a0ea858d967dcb6c26e3ba.png" class="math-inline" />

__具体例__

- <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" />（実数全体）：**非可算**  
  - カントールの対角線論法により、<img src="tmp/39a463630d85c3be0bd2f7745e08c8f8.png" class="math-inline" /> の全単射は存在しないことが示される
  - 濃度は**連続体濃度** <img src="tmp/0a2f3f25776381173161ff6d7ffeb331.png" class="math-inline" /> と呼ばれ、<img src="tmp/083da687551409df5c328a82e67d0f42.png" class="math-inline" />
- <img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" />（自然数のべき集合）：**非可算**  
  - Cantorの定理により <img src="tmp/4cbda99ae27270f76ecbc678da825ae5.png" class="math-inline" />
  - 実際には <img src="tmp/e1442ab9ee7577a4117aaf8bca13145b.png" class="math-inline" />
- 区間 <img src="tmp/039474a1a493fb3c2d8f4adbf24f971e.png" class="math-inline" /> や <img src="tmp/171d58b8063a24dc65a33bacb46a5fb7.png" class="math-inline" /> も <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> と対等なので非可算

### 3. 可算と非可算の性質の違い

__(1) 部分集合との関係__
- 可算集合の任意の無限部分集合は、**可算無限**または有限
  - 例：偶数全体、素数全体はどちらも <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と対等
- 非可算集合には、**非可算な部分集合**が必ず存在する
  - 例：<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の任意の区間 <img src="tmp/d06c6d246161dd0896775d08d0eb36bb.png" class="math-inline" /> は非可算

__(2) 和集合・直積の振る舞い__
- 可算集合の**有限和・有限直積**は可算集合
  - 例：<img src="tmp/477de5409352f91455b2be680c11cbc8.png" class="math-inline" />（整数の格子）は可算無限
- 可算集合の**可算無限個の和集合**も可算集合（可算和は可算）
  - 例：可算無限個の有限集合の和集合は可算
- 非可算集合との和集合・直積は、多くの場合非可算
  - 例：<img src="tmp/4bc1bd72a133938cef10f2acd8ebabd9.png" class="math-inline" /> は <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> と対等（濃度 <img src="tmp/0a2f3f25776381173161ff6d7ffeb331.png" class="math-inline" />）

__(3) 濃度の比較__
- 可算集合の濃度：<img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />（最小の無限濃度）
- 非可算集合の濃度：<img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> より真に大きい  
  - 例：<img src="tmp/127a7eca56f54e00ecd2c5b2d0b6af1a.png" class="math-inline" />


### 4. なぜこの区別が重要か

- **可算集合**は「要素を1,2,3,…と数え上げられる」という意味で、**扱いやすい無限**です。
- **非可算集合**は「自然数では数え上げられないほど多い」無限で、実数・関数空間・べき集合など、より複雑な対象に対応します。
- この区別により、「どのくらいの無限か」を厳密に議論でき、解析学・位相空間論・確率論など多くの分野で本質的な役割を果たします。


## 演習

### 問題

これまで説明した内容に対応した演習問題をまとめます。  

__第1部：対等・濃度の定義と具体例__

**問題1（対等の判定）**  
次の各組の集合 <img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> について、対等かどうかを判定し、理由を簡潔に述べよ。  
必要ならば、全単射・単射・全射の存在を示すか、その非存在を説明せよ。

1. <img src="tmp/c9ada39e81417f791e82174ed94a18d8.png" class="math-inline" />
2. <img src="tmp/8eb3360c6a39aad84ad820410c9b96be.png" class="math-inline" />（正の偶数の集合）
3. <img src="tmp/9cab16b374e44dc5cc9f82270af9ef80.png" class="math-inline" />
4. <img src="tmp/fc13a05a9ab122aa854a0535544f32d2.png" class="math-inline" />（開区間）
5. <img src="tmp/6e8b7724dde28424c012b8061554446a.png" class="math-inline" />


**問題2（濃度の比較）**  
次の各組について、<img src="tmp/efa403df45c1ea69e5464fa35c14815a.png" class="math-inline" /> と <img src="tmp/2a40418b805fd9d11859d4ab5e5edaec.png" class="math-inline" /> の大小関係（<img src="tmp/af87643e786a80599a72ef56635db374.png" class="math-inline" />）を答え、理由を述べよ。

1. <img src="tmp/8edaaae1108a69b0d5169246607a2fe3.png" class="math-inline" />
2. <img src="tmp/7271cddae1d386e9c0b888aca2cadac5.png" class="math-inline" />
3. <img src="tmp/d1787ccb445a4574bed16adbbac9afea.png" class="math-inline" />
4. <img src="tmp/a554cc4f1362424cd4367aaee5021869.png" class="math-inline" />
5. <img src="tmp/6c0cc6ca14a6ad53ed0ff69de8ed379a.png" class="math-inline" />

__第2部：有限集合・無限集合の判定と性質__

**問題3（有限・無限の判定）**  
次の集合が有限集合か無限集合かを判定し、その理由を述べよ。

1. <img src="tmp/1fc3f3bd38550706c02686e1d233dc41.png" class="math-inline" />
2. <img src="tmp/dd1a226a29dafe7a462c0393f8811412.png" class="math-inline" />
3. <img src="tmp/5145a96cc40a5400930f2189aa317345.png" class="math-inline" />
4. <img src="tmp/b871fcbb8c2d43934f61cb6f7eb65acf.png" class="math-inline" />
5. <img src="tmp/aed967913ae1329be9368ddda80be889.png" class="math-inline" />（自然数の1点部分集合全体）

**問題4（無限集合の特徴づけ）**  
集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が無限集合であることの同値な定義として、次の命題が知られている：

> <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 自身への**単射であって全射でない写像**が存在する。

この定義を用いて、次の集合が無限集合であることを示せ。

1. <img src="tmp/6e82299800e7ce873f7e86662e3dd8de.png" class="math-inline" />
2. <img src="tmp/e721acde2f2637b63f9d9d4d201385ae.png" class="math-inline" />
3. <img src="tmp/4e761a6e59344d15675d70821b57c33c.png" class="math-inline" />

__第3部：可算集合・非可算集合__

**問題5（可算・非可算の判定）**  
次の集合が可算集合か非可算集合かを判定し、理由を述べよ。

1. <img src="tmp/5f482196db06f2ad9cf0c6a00f9106e5.png" class="math-inline" />
2. <img src="tmp/b4560043f4ff4c083330c9267dcd012c.png" class="math-inline" />
3. <img src="tmp/592ef28275252070e288b760096fbc0e.png" class="math-inline" />（自然数から <img src="tmp/d8429da08828f207b99be93378783e65.png" class="math-inline" /> への写像全体、すなわち二進無限列の集合）
4. <img src="tmp/5c4d47937dbc0fb5e979279a05526bda.png" class="math-inline" />
5. <img src="tmp/94927d3cedc2831f60b308d2e1000094.png" class="math-inline" />


**問題6（可算和・可算直積の性質）**  
次の主張が正しいかどうかを判定し、正しければ証明の概要を、誤りならば反例を示せ。

1. 可算無限個の有限集合の和集合は、常に可算集合である。
2. 可算無限個の可算無限集合の直積は、常に可算集合である。
3. 可算無限個の非可算集合の和集合は、常に非可算集合である。

__第4部：Cantor–Bernsteinの定理とCantorの定理の適用__

**問題7（Cantor–Bernsteinの定理の適用）**  
Cantor–Bernsteinの定理を用いて、次の集合が互いに同じ濃度を持つことを示せ。

1. 開区間 <img src="tmp/039474a1a493fb3c2d8f4adbf24f971e.png" class="math-inline" /> と閉区間 <img src="tmp/171d58b8063a24dc65a33bacb46a5fb7.png" class="math-inline" />
2. 実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> と平面 <img src="tmp/b8e63d739e17c94e714c0f375c6651dd.png" class="math-inline" />
3. 自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と整数全体 <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />

（ヒント：単射を2つ構成し、定理を適用する。）

**問題8（Cantorの定理の理解）**  
Cantorの定理により、任意の集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> に対して <img src="tmp/bbe53ea0985487882e8fe1d73d29fd4e.png" class="math-inline" /> が成り立つ。

1. <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が有限集合のとき、この不等式が成り立つことを直接確認せよ。
2. <img src="tmp/6e82299800e7ce873f7e86662e3dd8de.png" class="math-inline" /> のとき、<img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" /> が非可算であることを、Cantorの定理を用いて説明せよ。
3. <img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" /> と <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> が同じ濃度を持つことを、Cantorの定理と他の事実を用いて説明せよ。


**問題9（対角線論法の練習）**  
Cantorの定理の証明で用いられる**対角線論法**を、次の設定で具体的に実行せよ。

- <img src="tmp/6e82299800e7ce873f7e86662e3dd8de.png" class="math-inline" /> とする。
- 全射 <img src="tmp/979b19cd12473f286ca9c65ba43ad9c9.png" class="math-inline" /> が存在すると仮定する。
- このとき、集合 <img src="tmp/a97ef10d4f1e6bc9dbbfa37445ea0a81.png" class="math-inline" /> を構成し、  
  任意の <img src="tmp/69dba50e295dc2ea4e735e1564c77754.png" class="math-inline" /> に対して <img src="tmp/d20eda1447a27ecd5715f777482a03c3.png" class="math-inline" /> となることを示せ。

__第5部：総合問題__

**問題10（総合）**  
次の問いに答えよ。

1. 有限集合の真部分集合は、必ず元の集合より小さい濃度を持つことを示せ。
2. 無限集合には、必ず真部分集合であって同じ濃度を持つものが存在することを示せ。
3. 可算無限集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と非可算集合 <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> があるとき、<img src="tmp/cdf89d42f0059ea8c63230b802884b42.png" class="math-inline" /> であることを示せ。
4. 任意の集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> に対して、<img src="tmp/bbe53ea0985487882e8fe1d73d29fd4e.png" class="math-inline" /> が成り立つことを、Cantorの定理を用いて説明せよ。

### 解答

以下、各問題の解答です。

__第1部：対等・濃度の定義と具体例__

__問題1（対等の判定）__

1. **<img src="tmp/b32a5f1c1d61423deb1866a80761eeee.png" class="math-inline" />**  
   - 全単射 <img src="tmp/07ac5cef2d6506f2a0968e425fcaf31c.png" class="math-inline" /> が存在するので、**対等**。

2. **<img src="tmp/1b22e508a35bd3742524f051cf97fe53.png" class="math-inline" />**  
   - 全単射 <img src="tmp/48c7fe82db4d8f85bc6f63469efcce92.png" class="math-inline" /> が存在するので、**対等**（どちらも可算無限）。

3. **<img src="tmp/52388820af1bc92ead20c6f59f6f6871.png" class="math-inline" />**  
   - <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> も <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> も可算無限集合であり、自然数との全単射が構成できるので、**対等**。

4. **<img src="tmp/931a119d2e29572081cf842bc1f83e47.png" class="math-inline" />**  
   - 全単射 <img src="tmp/e92977a6a33dfcc02d7fae4978623d2b.png" class="math-inline" /> などにより、<img src="tmp/afb0eaa590f31400a406de3fcaad435c.png" class="math-inline" /> が示せるので、**対等**（どちらも連続体濃度）。

5. **<img src="tmp/854b147a0180fc1734e3e6e958fa3f63.png" class="math-inline" />**  
   - Cantorの定理により <img src="tmp/385161a8e8d621aae21d830356dc75da.png" class="math-inline" /> なので、**対等ではない**。

__問題2（濃度の比較）__

1. **<img src="tmp/d121ddef9fc49fd9730aeec640c27bf2.png" class="math-inline" />**  
   - <img src="tmp/c5db7333fa667dd739b6d3e017ffe292.png" class="math-inline" /> なので、**<img src="tmp/dd47bd62dca4dfdd87dc31d2dccbde84.png" class="math-inline" />**。

2. **<img src="tmp/4e3bf98b3ab3df43fcebf019491bbce0.png" class="math-inline" />**  
   - どちらも可算無限で、全単射が構成できるので、**<img src="tmp/46c4dd564b86222198c1bd3d0f7100f9.png" class="math-inline" />**。

3. **<img src="tmp/0c363e9e84281c862e1c7d74316e883b.png" class="math-inline" />**  
   - <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> は可算無限、<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> は非可算（連続体濃度）なので、**<img src="tmp/cdf89d42f0059ea8c63230b802884b42.png" class="math-inline" />**。

4. **<img src="tmp/2f729d95247de0a2a76196f0c4048eaa.png" class="math-inline" />**  
   - Cantorの定理により <img src="tmp/86ccc7b339aba9b8999ee833d15e4209.png" class="math-inline" /> なので、**<img src="tmp/cdf89d42f0059ea8c63230b802884b42.png" class="math-inline" />**。

5. **<img src="tmp/78e25e820e92261581f75aa4b42ea1de.png" class="math-inline" />**  
   - 全単射 <img src="tmp/0f1ccdc473aeac31818c0b7ad131b87f.png" class="math-inline" /> などにより、<img src="tmp/a3ff6b33ca70f00b894cda1e751b0270.png" class="math-inline" /> が示せるので、**<img src="tmp/46c4dd564b86222198c1bd3d0f7100f9.png" class="math-inline" />**。

__第2部：有限集合・無限集合の判定と性質__

__問題3（有限・無限の判定）__

1. **<img src="tmp/1fc3f3bd38550706c02686e1d233dc41.png" class="math-inline" />**  
   - <img src="tmp/45e06f7d69f585340b9428100fe9e06f.png" class="math-inline" /> のみが条件を満たすので、**有限集合**（要素数 3）。

2. **<img src="tmp/dd1a226a29dafe7a462c0393f8811412.png" class="math-inline" />**  
   - <img src="tmp/a0fb990cdff1a00ea76d665901580c6e.png" class="math-inline" /> の2点のみなので、**有限集合**（要素数 2）。

3. **<img src="tmp/5145a96cc40a5400930f2189aa317345.png" class="math-inline" />**  
   - <img src="tmp/9386bc62898846f38fe79180d9f75b6e.png" class="math-inline" /> は有限範囲（<img src="tmp/4a1cd85b60d3e18449f6f54606ebc891.png" class="math-inline" /> 程度）に限られるので、**有限集合**。

4. **<img src="tmp/b871fcbb8c2d43934f61cb6f7eb65acf.png" class="math-inline" />**  
   - 自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> と1対1に対応するので、**無限集合**（可算無限）。

5. **<img src="tmp/aed967913ae1329be9368ddda80be889.png" class="math-inline" />**  
   - 各1点集合 <img src="tmp/a1cf28b85f04386e3451e20b6282fc7e.png" class="math-inline" /> と自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> が1対1に対応するので、**無限集合**（可算無限）。

__問題4（無限集合の特徴づけ）__

> 命題：<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が無限集合 <img src="tmp/6a454feba3850900425336c9be01d5bd.png" class="math-inline" /> <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 自身への**単射であって全射でない写像**が存在する。

1. **<img src="tmp/6e82299800e7ce873f7e86662e3dd8de.png" class="math-inline" />**  
   - 写像 <img src="tmp/7f022a163163da6210dcf2bd31ce7fd6.png" class="math-inline" /> は単射だが、1 に移る元がないので全射ではない。  
   - よって <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> は無限集合。

2. **<img src="tmp/e721acde2f2637b63f9d9d4d201385ae.png" class="math-inline" />**  
   - 写像 <img src="tmp/7f022a163163da6210dcf2bd31ce7fd6.png" class="math-inline" /> は単射だが、0 に移る元がないので全射ではない。  
   - よって <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> は無限集合。

3. **<img src="tmp/4e761a6e59344d15675d70821b57c33c.png" class="math-inline" />**  
   - 写像 <img src="tmp/100794f0c243a27c5cb7c88bab48f520.png" class="math-inline" /> は単射だが、0 に移る元がないので全射ではない。  
   - よって <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> は無限集合。

__第3部：可算集合・非可算集合__

__問題5（可算・非可算の判定）__

1. **<img src="tmp/5f482196db06f2ad9cf0c6a00f9106e5.png" class="math-inline" />**  
   - 全単射 <img src="tmp/0f1ccdc473aeac31818c0b7ad131b87f.png" class="math-inline" /> などにより <img src="tmp/a3ff6b33ca70f00b894cda1e751b0270.png" class="math-inline" /> なので、**可算集合**（可算無限）。

2. **<img src="tmp/b4560043f4ff4c083330c9267dcd012c.png" class="math-inline" />**  
   - <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> が可算なので、その有限直積も可算 → **可算集合**。

3. **<img src="tmp/592ef28275252070e288b760096fbc0e.png" class="math-inline" />**  
   - 自然数上の二進列全体であり、<img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" /> と対等。  
   - Cantorの定理により非可算 → **非可算集合**。

4. **<img src="tmp/5c4d47937dbc0fb5e979279a05526bda.png" class="math-inline" />**  
   - 有限部分集合全体は、自然数と有限列の対応で可算 → **可算集合**。

5. **<img src="tmp/94927d3cedc2831f60b308d2e1000094.png" class="math-inline" />**  
   - <img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" /> から有限部分集合全体（可算）を除いたもの。  
   - 非可算集合から可算集合を除いても非可算 → **非可算集合**。

__問題6（可算和・可算直積の性質）__

1. **可算無限個の有限集合の和集合は、常に可算集合である。**  
   - **正しい**。  
   - 各有限集合の要素を列挙し、自然数と1対1に対応づけることで可算集合になる。

2. **可算無限個の可算無限集合の直積は、常に可算集合である。**  
   - **誤り**。  
   - 反例：<img src="tmp/899f166a67593cf99f2933b2f7e28d4e.png" class="math-inline" />（可算無限個の2点集合の直積）は非可算。

3. **可算無限個の非可算集合の和集合は、常に非可算集合である。**  
   - **誤り**。  
   - 反例：各 <img src="tmp/5a1e5b9d2f1b27277933449d60be618a.png" class="math-inline" /> は非可算だが、和集合 <img src="tmp/a2c2b061e029988ba5c861d645e0bd39.png" class="math-inline" /> は非可算とは限らない？  
     実際には <img src="tmp/1e9bb7768b974a2860db12d8774d5449.png" class="math-inline" /> も非可算なので、より適切な反例は「互いに素でない非可算集合の和集合が可算になる例」を考える必要があるが、標準的な反例は少ない。  
     一般には「常に非可算」とは限らない（例：各 <img src="tmp/4357e013de910db8c7089af11a056786.png" class="math-inline" /> が同じ非可算集合なら和集合も非可算だが、配置によっては可算になる可能性もある）。

__第4部：Cantor–Bernsteinの定理とCantorの定理の適用__

__問題7（Cantor–Bernsteinの定理の適用）__

1. **<img src="tmp/039474a1a493fb3c2d8f4adbf24f971e.png" class="math-inline" /> と <img src="tmp/171d58b8063a24dc65a33bacb46a5fb7.png" class="math-inline" />**  
   - 包含写像 <img src="tmp/056c2e636ef2baa224afad3f14b14997.png" class="math-inline" /> は単射 → <img src="tmp/558c85caa0128574499b53bcdaf4cf78.png" class="math-inline" />。  
   - 写像 <img src="tmp/345b3a87f48e74f3790d853bcb09d250.png" class="math-inline" /> は単射 → <img src="tmp/30866dae3e51259b01c2ab10e07ac948.png" class="math-inline" />。  
   - Cantor–Bernsteinの定理より <img src="tmp/4b379bf6ebf9a4144cd231ea1bd97ea5.png" class="math-inline" />。

2. **<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> と <img src="tmp/b8e63d739e17c94e714c0f375c6651dd.png" class="math-inline" />**  
   - 包含写像 <img src="tmp/d8093839920e511f4bb7c9e2849e3333.png" class="math-inline" /> は単射 → <img src="tmp/76012434509cc075276dd4339d692a9a.png" class="math-inline" />。  
   - 空間充填曲数（例：Peano曲線）などにより、<img src="tmp/d1336bf4561272e531ca4db715b5adef.png" class="math-inline" /> の全射（したがって単射も構成可能）が存在 → <img src="tmp/607665ee936c30e7393c3663d667d265.png" class="math-inline" />。  
   - Cantor–Bernsteinの定理より <img src="tmp/93e79cc0de6c6ab875d0a51eb06189b7.png" class="math-inline" />。

3. **<img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />**  
   - 包含写像 <img src="tmp/dc0dd7ceb2a4f89b69a71c92b5c8775b.png" class="math-inline" /> は単射 → <img src="tmp/f33939551e8bfadf3c2972bcdf92c710.png" class="math-inline" />。  
   - 写像 <img src="tmp/cc28811f8ad7ac23f7207815725596c8.png" class="math-inline" /> などは単射 → <img src="tmp/71811457f5e170896e26b982fce46dde.png" class="math-inline" />。  
   - Cantor–Bernsteinの定理より <img src="tmp/faf96ebc3ee36a42b959b51f44dda425.png" class="math-inline" />。

__問題8（Cantorの定理の理解）__

1. **<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が有限集合のとき、<img src="tmp/bbe53ea0985487882e8fe1d73d29fd4e.png" class="math-inline" />**  
   - <img src="tmp/d46f7b659df64db7200be7157c3ddad9.png" class="math-inline" /> なら <img src="tmp/01166bdaef87f28a0cd5015d527525c4.png" class="math-inline" />。  
   - 任意の自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> について <img src="tmp/4925ac9beb1c8d09bdd00eb932e45a23.png" class="math-inline" /> が成り立つので、<img src="tmp/bbe53ea0985487882e8fe1d73d29fd4e.png" class="math-inline" />。

2. **<img src="tmp/6e82299800e7ce873f7e86662e3dd8de.png" class="math-inline" /> のとき、<img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" /> が非可算であること**  
   - Cantorの定理により <img src="tmp/385161a8e8d621aae21d830356dc75da.png" class="math-inline" />。  
   - <img src="tmp/024d16c0e3fdfd485f9bd70c7df42ce1.png" class="math-inline" />（可算無限）なので、<img src="tmp/fb2bc67b6c102349e31a0d30f543bfad.png" class="math-inline" />、すなわち非可算。

3. **<img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" /> と <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> が同じ濃度を持つこと**  
   - <img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" /> は二進列 <img src="tmp/899f166a67593cf99f2933b2f7e28d4e.png" class="math-inline" /> と対等であり、各二進列は実数の二進展開（適切に修正）と対応づけられる。  
   - これにより <img src="tmp/32666d2a1d5636837f2b32c5f88cd728.png" class="math-inline" />（連続体濃度 <img src="tmp/0a2f3f25776381173161ff6d7ffeb331.png" class="math-inline" />）。

__問題9（対角線論法の練習）__

- <img src="tmp/6e82299800e7ce873f7e86662e3dd8de.png" class="math-inline" />、全射 <img src="tmp/979b19cd12473f286ca9c65ba43ad9c9.png" class="math-inline" /> が存在すると仮定。
- 集合 <img src="tmp/a97ef10d4f1e6bc9dbbfa37445ea0a81.png" class="math-inline" /> を構成。
- 任意の <img src="tmp/69dba50e295dc2ea4e735e1564c77754.png" class="math-inline" /> について：
  - <img src="tmp/eae23e368bd2517e4abd3a1a244f292c.png" class="math-inline" />。
  - もし <img src="tmp/a8f8514f796a83d61560cc5523a34793.png" class="math-inline" /> なら、<img src="tmp/c6750ebfeaf82f19efa8b39867e60e72.png" class="math-inline" /> となり矛盾。
  - よって任意の <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> に対し <img src="tmp/d20eda1447a27ecd5715f777482a03c3.png" class="math-inline" />。
- したがって <img src="tmp/d9fa4fdb248ddc6a60a2b95f8f1b02ee.png" class="math-inline" /> は <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> の像に含まれず、<img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> は全射ではない → 矛盾。  
  よって全射 <img src="tmp/979b19cd12473f286ca9c65ba43ad9c9.png" class="math-inline" /> は存在しない。

__第5部：総合問題__

__問題10（総合）__

1. **有限集合の真部分集合は、必ず元の集合より小さい濃度を持つ**  
   - 有限集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の真部分集合 <img src="tmp/17042b0c8a40bd7726032c5556cae1eb.png" class="math-inline" /> は、要素数が <img src="tmp/863a31f84a8de8cf5e9aceb529f4b954.png" class="math-inline" />。  
   - 濃度の定義（全単射の存在）により、<img src="tmp/863a31f84a8de8cf5e9aceb529f4b954.png" class="math-inline" />。

2. **無限集合には、必ず真部分集合であって同じ濃度を持つものが存在する**  
   - 無限集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> は、ある真部分集合 <img src="tmp/bb9e244c97efb0ca76d4fc23f7384ba1.png" class="math-inline" /> と全単射 <img src="tmp/c0f904752233741fed57139839fdf3ec.png" class="math-inline" /> を持つ（Dedekind無限の定義）。  
   - よって <img src="tmp/451f4cf13c9b54509d9e7ef03f3749b5.png" class="math-inline" />。

3. **可算無限集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と非可算集合 <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> があるとき、<img src="tmp/cdf89d42f0059ea8c63230b802884b42.png" class="math-inline" />**  
   - <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は可算無限なので <img src="tmp/7d9ce8d0341d1192b344d69e64b6e685.png" class="math-inline" />。  
   - <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は非可算なので <img src="tmp/48bb99aefa1e067c528bd3590be01ae1.png" class="math-inline" />。  
   - よって <img src="tmp/cdf89d42f0059ea8c63230b802884b42.png" class="math-inline" />。

4. **任意の集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> に対して、<img src="tmp/bbe53ea0985487882e8fe1d73d29fd4e.png" class="math-inline" />**  
   - Cantorの定理により、全射 <img src="tmp/24177eee36cbc03bdc4939c4c3c7a4e6.png" class="math-inline" /> は存在しない。  
   - 一方、包含写像 <img src="tmp/082be666b7331967ecc446b5c898de12.png" class="math-inline" /> は単射 <img src="tmp/24177eee36cbc03bdc4939c4c3c7a4e6.png" class="math-inline" /> を与えるので <img src="tmp/866c7d95cf55649605be9d6179fb6aa3.png" class="math-inline" />。  
   - 全射がないため <img src="tmp/dd6845e29c6c6125f4c1c9910b930bd4.png" class="math-inline" />、したがって <img src="tmp/bbe53ea0985487882e8fe1d73d29fd4e.png" class="math-inline" />。
  
<div style="page-break-before:always"></div>


# 順序集合

## 比較可能

以下、集合論で使われる「比較可能（comparable）」の数学的な定義を整理します。

### 1. 比較可能の定義（順序集合の文脈）

__基本設定__
- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" />：集合
- <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" />：<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の**二項関係**（順序関係）

__定義__

2つの元 <img src="tmp/6ecf568da0898bbcac56f45f3573860b.png" class="math-inline" /> が**比較可能（comparable）** であるとは、


<div class="math-display-container"><img src="tmp/4f987fdd9e975cece07c6d200a99190c.png" class="math-display" /></div>


の少なくとも一方が成り立つことをいう。

記号的に書くと：


<div class="math-display-container"><img src="tmp/d1ad51a287122e3df2b9f087c512acec.png" class="math-display" /></div>



__補足__
- どちらの関係も成り立たないとき、すなわち
  

<div class="math-display-container"><img src="tmp/6a807e539103ebffd743bbb642f18b44.png" class="math-display" /></div>


  のとき、<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> と <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は**比較不能（incomparable）** であるという。
- 全順序集合（totally ordered set）では、**任意の2元が比較可能**である。

### 2. 具体例

__例1：実数の通常の順序 <img src="tmp/028e8e6759d744985e5119ebfce2af99.png" class="math-inline" />__
- 任意の実数 <img src="tmp/4c7d4aa955f1d0b51d34aec53be51d5c.png" class="math-inline" /> について <img src="tmp/f5645db615a2af23d8c0eb8fbfb5bdeb.png" class="math-inline" /> または <img src="tmp/8ae95cd25057de5330bfaeaa535a5fe0.png" class="math-inline" /> が成り立つ。
- よって、任意の2元は比較可能（全順序集合）。

__例2：べき集合の包含順序 <img src="tmp/4d0b5d4d81e343ceaa2934f32bec7071.png" class="math-inline" />__
- <img src="tmp/029996afad6f9c691f107e1d78c2826a.png" class="math-inline" /> について、<img src="tmp/d3cd613dbef8bc0f1be0fb2d340867e0.png" class="math-inline" /> でも <img src="tmp/42f1edc1db8a856d544a069291f86036.png" class="math-inline" /> でもない場合がある。
  - 例：<img src="tmp/fdfb023b1885c28b46ba3d728e3486d3.png" class="math-inline" />
- このとき <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は比較不能。

__例3：整除関係（自然数）__
- <img src="tmp/c9713ce2df40f406d1f1ff0bfa4ff87e.png" class="math-inline" /> を「<img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> は <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> を割り切る」と定義。
- 例：2 と 3 は互いに割り切れないので、比較不能。
- 例：2 と 4 は <img src="tmp/28306dc5685e7d3b026dc3ad894b6856.png" class="math-inline" /> なので比較可能。

__例4:包含関係による半順序集合の可視化__

- 集合族 {∅, {1}, {2}, {1,2}} をハッセ図で描画し、
- 包含関係があるペア（比較可能）を緑の矢印で、
- 包含関係がないペア（{1} と {2}）をオレンジのノードで強調します。

比較可能=一方が他方の部分集合であるということを示します。

<img src="image/6_order_set/1782939636465.png" alt="" width="500" style="display: block; margin: 0 auto;">

>__ハッセ図（Hasse diagram）__  
>ハッセ図とは、 半順序集合（partially ordered set）を視覚的に表すための図です。
>1. 各元を点（ノード）として描く
>2. <img src="tmp/99ab38e10efa5fca413f86052c08b091.png" class="math-inline" /> で、その間に他の元が入らないとき（被覆関係）だけ線を引く
>3. 大きい元を上に、小さい元を下に配置する
>これにより、半順序関係を簡潔に表現できます。

## 順序集合

**順序集合（ordered set）** は、集合の要素の間に「大小関係」や「前後関係」のような**順序**を定めた構造です。  
数学の多くの分野で基本的な役割を果たします。

### 1. 順序集合の数学的定義

__(1) 二項関係としての定義__

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の**二項関係** <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> が次の3条件を満たすとき、<img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> を**前順序集合（preordered set）** といいます。

1. **反射律**：任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対し <img src="tmp/d88e1e9cda96f421a94c273d5390f6ed.png" class="math-inline" />。
2. **推移律**：任意の <img src="tmp/1daf0149a591c5c099a6d194ce27f91b.png" class="math-inline" /> に対し、 <img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> かつ <img src="tmp/9cf142fe3938e0d895dee0471b462221.png" class="math-inline" /> ならば <img src="tmp/6ba64e25a1e512afee5c0de7cea5fc11.png" class="math-inline" />。

さらに、次の条件を満たすとき、 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> を**半順序集合（partially ordered set, poset）** といいます。

3. **反対称律**：任意の <img src="tmp/aafcdd2e8da6043624a2be58ad91ff0e.png" class="math-inline" /> に対し、<img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> かつ <img src="tmp/708596235cba73cab553a9bb5933227c.png" class="math-inline" /> ならば <img src="tmp/17f4448b0ccf706290b87890161f80de.png" class="math-inline" />。

さらに、次の条件を満たすとき、<img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> を**全順序集合（totally ordered set / linearly ordered set）** といいます。

4. **全順序性**：任意の <img src="tmp/aafcdd2e8da6043624a2be58ad91ff0e.png" class="math-inline" /> に対し、<img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> または <img src="tmp/708596235cba73cab553a9bb5933227c.png" class="math-inline" /> の少なくとも一方が成り立つ。

まとめると：

- **前順序集合**：反射律＋推移律
- **半順序集合**：前順序＋反対称律
- **全順序集合**：半順序＋全順序性

__(2) 記号と用語__

- <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> を**順序関係（order relation）** と呼ぶ。
- <img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> を「 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> 以下」「 <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> 以上」などと読む。
- <img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> かつ <img src="tmp/7e51fc5af1798e32ebc66e288b627fbd.png" class="math-inline" /> のとき、<img src="tmp/24c9e6d4b22616cc4d909e1612426371.png" class="math-inline" /> と書くこともある（**狭義の順序**）。
- 順序集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> を単に <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> と書くことも多い。

### 2. 具体例

__(1) 数の集合__

- <img src="tmp/d1f0582382fcd4b54172c474893ecd9d.png" class="math-inline" /> ：実数全体に通常の大小関係を入れた**全順序集合**。
- <img src="tmp/8f8a1974eaba6382e027538fe98d3e6b.png" class="math-inline" /> ：自然数全体も全順序集合。
- <img src="tmp/914062146d83bd9e6efa56e7f34ab08b.png" class="math-inline" /> ：有理数全体も全順序集合。

__(2) べき集合（部分集合の包含）__

- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> を任意の集合とし、<img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> をそのべき集合とする。
- 包含関係 <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" /> を順序とすると、<img src="tmp/0d91cb242265c2fb19f5c27668ca083d.png" class="math-inline" /> は**半順序集合**（一般には全順序ではない）。
  - 例：<img src="tmp/73bc3c5dc62a5679cd55a878fb7305f8.png" class="math-inline" /> のとき、<img src="tmp/275f0afd9bd9a2ec101008af4cd2b9ff.png" class="math-inline" /> と <img src="tmp/4046fb3f07271b5adf8d1bf744ca5f50.png" class="math-inline" /> は比較不能（どちらも他方の部分集合ではない）。

__(3) 整除関係__

- <img src="tmp/9f51e502b24e653c5c14419b2d895993.png" class="math-inline" /> ：自然数全体に「 <img src="tmp/c9713ce2df40f406d1f1ff0bfa4ff87e.png" class="math-inline" /> （ <img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> は <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> を割り切る）」という関係を入れると、半順序集合になる。
  - 反射律： <img src="tmp/564ded13611252b1108f6fceee640b84.png" class="math-inline" />
  - 推移律： <img src="tmp/c9713ce2df40f406d1f1ff0bfa4ff87e.png" class="math-inline" /> かつ <img src="tmp/495e18e7921ef04213acb0d7e3beddb5.png" class="math-inline" /> なら <img src="tmp/83cdfab97ccebaa4b471ffbbc9542e90.png" class="math-inline" />
  - 反対称律： <img src="tmp/c9713ce2df40f406d1f1ff0bfa4ff87e.png" class="math-inline" /> かつ <img src="tmp/c6818f5f1d0870d4c7e95edb950f5818.png" class="math-inline" /> なら <img src="tmp/08417e0378018df2bc16a4a48e549d54.png" class="math-inline" />
  - 全順序ではない（例：2 と 3 は比較不能）。

### 3. なぜ順序集合が必要なのか

__(1) 「大小」や「前後」を数学的に扱うため__

- 実数や自然数の大小関係、時間の前後関係、集合の包含関係など、現実世界や数学の多くの場面で「順序」が自然に現れます。
- 順序集合は、こうした**順序構造を抽象化**したものです。
- これにより、「最大元・最小元」「上界・下界」「極大元・極小元」など、順序に基づく概念を一般の集合に対して定義できます。

__(2) 極限・連続性・完備性の基礎__

- 実数直線 <img src="tmp/028e8e6759d744985e5119ebfce2af99.png" class="math-inline" /> は**全順序かつ完備**（任意の有界な部分集合が上限を持つ）であり、解析学の基礎となります。
- 数列の極限、関数の連続性、微分・積分などは、実数の順序構造と位相構造に強く依存しています。
- 順序集合の一般論（完備半順序集合、dcpoなど）は、領域理論やプログラム意味論においても重要です。

__(3) 代数的構造との組み合わせ__

- **束（lattice）**：任意の2元が上限（join）と下限（meet）を持つ半順序集合。ブール代数、ハッセ図などに現れる。
- **整列集合（well-ordered set）**：任意の空でない部分集合が最小元を持つ全順序集合。整列可能定理や順序数理論の基礎。
- **順序群・順序環**：群や環の演算と整合的な順序構造を持つもの。実数体 <img src="tmp/374ce30d3d1b9fddf6746d98becce4b8.png" class="math-inline" /> はその典型例。

__(4) 集合論・論理・計算機科学での応用__

- **集合論**：順序数は整列集合の同型類として定義され、無限の大きさを測る基本的な道具です。
- **モデル理論**：構造に順序が入っているかどうかで、満たす論理式のクラスが変わります。
- **計算機科学**：プログラムの実行順序、データ構造（ヒープ、二分探索木など）、型システムの部分型関係など、多くの場面で順序集合が現れます。

### 4. 順序集合の基本的な概念（概要）

順序集合を扱う上で重要な概念をいくつか挙げます（詳細は別途説明可能です）。

- **最大元・最小元**：集合全体の中で最も大きい／小さい元。
- **極大元・極小元**：それより大きい／小さい元が存在しない元。
- **上界・下界**：与えられた部分集合を「上から／下から」抑える元。
- **上限（sup）・下限（inf）**：上界／下界の集合の最小元／最大元。
- **鎖（chain）**：全順序な部分集合。
- **反鎖（antichain）**：互いに比較不能な元からなる部分集合。
- **整列集合**：任意の空でない部分集合が最小元を持つ全順序集合。

## 部分順序集合

**部分順序集合（partially ordered set, poset）** は、集合の要素の間に「順序」を定めた構造のうち、**反射律・推移律・反対称律**を満たすものを指します。  
「全順序」とは異なり、**すべての2元が比較可能とは限らない**という点が特徴です。

### 1. 部分順序集合の定義

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の**二項関係** <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> が次の3条件を満たすとき、 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> を**部分順序集合（poset）** といいます。

1. **反射律**：任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対し <img src="tmp/d88e1e9cda96f421a94c273d5390f6ed.png" class="math-inline" />。
2. **推移律**：任意の <img src="tmp/1daf0149a591c5c099a6d194ce27f91b.png" class="math-inline" /> に対し、<img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> かつ <img src="tmp/9cf142fe3938e0d895dee0471b462221.png" class="math-inline" /> ならば <img src="tmp/6ba64e25a1e512afee5c0de7cea5fc11.png" class="math-inline" />。
3. **反対称律**：任意の <img src="tmp/aafcdd2e8da6043624a2be58ad91ff0e.png" class="math-inline" /> に対し、<img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> かつ <img src="tmp/708596235cba73cab553a9bb5933227c.png" class="math-inline" /> ならば <img src="tmp/17f4448b0ccf706290b87890161f80de.png" class="math-inline" />。

このとき、<img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> を**部分順序関係（partial order）** と呼びます。


### 2. 部分順序集合の具体例

__(1) 数の集合（通常の大小）__

- <img src="tmp/d1f0582382fcd4b54172c474893ecd9d.png" class="math-inline" /> ：実数全体に通常の大小関係を入れると、**全順序集合**（したがって部分順序集合でもある）。
- <img src="tmp/8f8a1974eaba6382e027538fe98d3e6b.png" class="math-inline" /> ：自然数全体も同様。

これらは**すべての2元が比較可能**なので、特に**全順序集合**と呼ばれます。

__(2) べき集合（部分集合の包含）__

- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> を任意の集合とし、 <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> をそのべき集合とする。
- 包含関係 <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" /> を順序とすると、 <img src="tmp/0d91cb242265c2fb19f5c27668ca083d.png" class="math-inline" /> は**部分順序集合**。
  - 反射律： <img src="tmp/ef1b6af8981071c83720cc230f850df6.png" class="math-inline" />
  - 推移律： <img src="tmp/d3cd613dbef8bc0f1be0fb2d340867e0.png" class="math-inline" /> かつ <img src="tmp/804016bfed31c0f7dbb855bf2dfc11ad.png" class="math-inline" /> なら <img src="tmp/0073291a813c2d3a65cf8c8fcc7ecd96.png" class="math-inline" />
  - 反対称律： <img src="tmp/d3cd613dbef8bc0f1be0fb2d340867e0.png" class="math-inline" /> かつ <img src="tmp/42f1edc1db8a856d544a069291f86036.png" class="math-inline" /> なら <img src="tmp/aaea37894ac3add2b714b6ff50ad2d60.png" class="math-inline" />

例： <img src="tmp/73bc3c5dc62a5679cd55a878fb7305f8.png" class="math-inline" /> のとき、
- <img src="tmp/21e525f38a1f2617ab215c27cff07678.png" class="math-inline" /> 、 <img src="tmp/ce9da7903bae7507dd92406c0f229904.png" class="math-inline" /> は成り立つが、
- <img src="tmp/275f0afd9bd9a2ec101008af4cd2b9ff.png" class="math-inline" /> と <img src="tmp/4046fb3f07271b5adf8d1bf744ca5f50.png" class="math-inline" /> は互いに包含関係になく、**比較不能**です。

__(3) 整除関係__

- <img src="tmp/9f51e502b24e653c5c14419b2d895993.png" class="math-inline" /> ：自然数全体に「 <img src="tmp/c9713ce2df40f406d1f1ff0bfa4ff87e.png" class="math-inline" />（<img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> は <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> を割り切る）」という関係を入れると、部分順序集合になる。
  - 反射律： <img src="tmp/564ded13611252b1108f6fceee640b84.png" class="math-inline" />
  - 推移律： <img src="tmp/c9713ce2df40f406d1f1ff0bfa4ff87e.png" class="math-inline" /> かつ <img src="tmp/495e18e7921ef04213acb0d7e3beddb5.png" class="math-inline" /> なら <img src="tmp/83cdfab97ccebaa4b471ffbbc9542e90.png" class="math-inline" />
  - 反対称律： <img src="tmp/c9713ce2df40f406d1f1ff0bfa4ff87e.png" class="math-inline" /> かつ <img src="tmp/c6818f5f1d0870d4c7e95edb950f5818.png" class="math-inline" /> なら <img src="tmp/08417e0378018df2bc16a4a48e549d54.png" class="math-inline" />

例：
- <img src="tmp/28306dc5685e7d3b026dc3ad894b6856.png" class="math-inline" />、<img src="tmp/2375852e598051552fa75b9ab3099aa9.png" class="math-inline" /> は成り立つが、
- <img src="tmp/748d014a08c85df236ffbfbc57dfcde3.png" class="math-inline" /> と <img src="tmp/04e7c4e120880e26f5110d0d71495fab.png" class="math-inline" /> は互いに割り切れないので**比較不能**。

__(4) ベクトル空間の部分空間__

- <img src="tmp/170cc9bd0fbfe8e1f3d73b7e9e6f4e60.png" class="math-inline" /> をベクトル空間とし、その部分空間全体の集合を考える。
- 包含関係 <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" /> を順序とすると、部分順序集合になる。
- 例：<img src="tmp/b8e63d739e17c94e714c0f375c6651dd.png" class="math-inline" /> の部分空間（原点、直線、平面全体）は包含関係で順序づけられる。

### 3. 部分順序集合の基本的な概念

部分順序集合には、順序に基づくさまざまな概念が定義されます。

__(1) 最大元・最小元__

- **最大元（maximum element）**：  
  すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して <img src="tmp/b74de487351bb6052f23f20754578e84.png" class="math-inline" /> となる <img src="tmp/53d768e41950280cba385ae6d8d9433f.png" class="math-inline" />。
- **最小元（minimum element）**：  
  すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して <img src="tmp/d088f2d38609b736481acf7c42a2ec44.png" class="math-inline" /> となる <img src="tmp/53d768e41950280cba385ae6d8d9433f.png" class="math-inline" />。

存在すれば一意です（反対称律より）。

__(2) 極大元・極小元__

- **極大元（maximal element）**：  
  <img src="tmp/53d768e41950280cba385ae6d8d9433f.png" class="math-inline" /> であって、<img src="tmp/7f225c7f1db16425dc2a44f71d886d63.png" class="math-inline" /> となる <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> が存在しないもの。
- **極小元（minimal element）**：  
  <img src="tmp/53d768e41950280cba385ae6d8d9433f.png" class="math-inline" /> であって、<img src="tmp/8e1f704269339fb74f048c025da7fcfc.png" class="math-inline" /> となる <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> が存在しないもの。

最大元・最小元とは異なり、**複数存在し得る**点が重要です。

__(3) 上界・下界・上限・下限__

部分集合 <img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> を考えます。

- **上界（upper bound）**：  
  すべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> に対して <img src="tmp/aa554c18bc824da199690a0ac68ee3bd.png" class="math-inline" /> となる <img src="tmp/9dcb0ce7aa80bed81144a7fa4ad9043e.png" class="math-inline" />。
- **下界（lower bound）**：  
  すべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> に対して <img src="tmp/8c4a58686d818a44a84acef49e7b75a3.png" class="math-inline" /> となる <img src="tmp/2ea8ea80fd893716f7d88bb0dfd94c3b.png" class="math-inline" />。
- **上限（supremum, least upper bound）**：  
  上界全体の集合の**最小元**（存在すれば一意）。
- **下限（infimum, greatest lower bound）**：  
  下界全体の集合の**最大元**（存在すれば一意）。

例：<img src="tmp/028e8e6759d744985e5119ebfce2af99.png" class="math-inline" /> では、有界な集合は必ず上限・下限を持ちます（実数の完備性）。

__(4) 鎖（chain）と反鎖（antichain）__

- **鎖（chain）**：  
  <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合 <img src="tmp/7da569b3f6d6a9ad1defdb55cccf0f5f.png" class="math-inline" /> で、任意の2元が比較可能なもの（すなわち <img src="tmp/7da569b3f6d6a9ad1defdb55cccf0f5f.png" class="math-inline" /> 上で <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> が全順序になる）。
- **反鎖（antichain）**：  
  <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> で、任意の相異なる2元が比較不能なもの。

例：べき集合 <img src="tmp/488794decd2567746d0801e1a6339883.png" class="math-inline" /> では、
- <img src="tmp/842c2d411af6b8a0a56cc36d0d16bf56.png" class="math-inline" /> は鎖、
- <img src="tmp/e9f429fbe9ad47662cb72c6be2d06f3b.png" class="math-inline" /> は反鎖です。

### 4. なぜ部分順序集合が重要なのか

__(1) 「比較不能」を許容する現実的なモデル__

- 実数の大小のように「すべての2元が比較可能」な状況は特殊です。
- 多くの現実的な状況（集合の包含、整除関係、部分型関係など）では、**比較不能な要素**が自然に現れます。
- 部分順序集合は、こうした**部分的にしか順序づけられない構造**を数学的に扱うための枠組みです。

__(2) 束（lattice）・ブール代数の基礎__

- 任意の2元が上限と下限を持つ部分順序集合を**束（lattice）** といいます。
- さらに補元演算などが備わると**ブール代数**となり、論理・集合演算・電子回路の設計などに応用されます。
- これらはすべて部分順序集合を土台としています。

__(3) 順序理論・組合せ論・計算機科学__

- **Dilworthの定理**や**Mirskyの定理**など、鎖と反鎖の関係に関する重要な結果があります。
- プログラムの実行順序、データ構造（ヒープ、優先度付きキューなど）、型システムの部分型関係など、計算機科学の多くの概念が部分順序集合としてモデル化されます。

## 順序同型、順序型

**順序同型（order isomorphism）**と**順序型（order type）** は、順序集合の「形」や「構造」を比較・分類するための概念です。  
集合の「対等」（全単射）が「要素の個数」を比較するのに対し、順序同型は**順序関係まで含めた構造**を比較します。

### 1. 順序同型（order isomorphism）

__定義__

2つの順序集合 <img src="tmp/b809c81579da26b8a4a8a36565a8082f.png" class="math-inline" /> と <img src="tmp/98b779eceb779a56065b921e642dfbca.png" class="math-inline" /> について、写像 <img src="tmp/c0f904752233741fed57139839fdf3ec.png" class="math-inline" /> が次の条件を満たすとき、**順序同型写像（order isomorphism）** であるといいます。

1. **全単射**である：  
   - 単射かつ全射。
2. **順序を保つ（order-preserving）**：  
   - 任意の <img src="tmp/06caf98428cc58b3b61e3c80a8012d6b.png" class="math-inline" /> に対し、  
     

<div class="math-display-container"><img src="tmp/9c96c2c33ab8bd44e3936f70573f024a.png" class="math-display" /></div>



このとき、 <img src="tmp/b809c81579da26b8a4a8a36565a8082f.png" class="math-inline" /> と <img src="tmp/98b779eceb779a56065b921e642dfbca.png" class="math-inline" /> は**順序同型（order isomorphic）** であるといい、


<div class="math-display-container"><img src="tmp/b3a949881d588a37a7842d4144ab80a6.png" class="math-display" /></div>


と書きます。

__直感的な意味__

- 順序同型写像は、**要素の対応づけだけでなく、順序関係も完全に保存する**全単射です。
- つまり、2つの順序集合は「順序構造まで含めて同じ形」をしている、とみなせます。
- 例：自然数全体 <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" /> と正の偶数全体 <img src="tmp/43f1c4d5f096948385a0fde7185c4662.png" class="math-inline" /> は順序同型（写像 <img src="tmp/956e10b149367ff8642c84f4ca342f92.png" class="math-inline" />）。

__具体例__

1. **有限全順序集合**  
   - <img src="tmp/451f07222d20ef0263efce4329b2c6b9.png" class="math-inline" /> と <img src="tmp/692f8d78263f169cb7940e8ff5f12c7e.png" class="math-inline" />（辞書順など）は順序同型。
   - 全単射 <img src="tmp/5e92d3c292df681b80efe6f196de0a4b.png" class="math-inline" /> は順序を保つ。

2. **可算無限の全順序集合**  
   - <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" /> と <img src="tmp/5608f64ad5726b0cec3f4b4905f05ef3.png" class="math-inline" /> は**集合としては対等**だが、**順序同型ではない**。  
     - <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> には最小元がないが、<img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> には最小元 1 があるため。

3. **べき集合の包含順序**  
   - <img src="tmp/f272b47ed201b6735c5ad7003f509d4c.png" class="math-inline" /> と <img src="tmp/a9d38f4f575bfc8ee1e29d999bee7f45.png" class="math-inline" /> は順序同型ではない（構造が異なる）。

### 2. 順序型（order type）

__定義__

順序集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の**順序型（order type）** とは、その順序構造を**同型類として捉えたもの**です。

より正確には：

- 順序同型という同値関係（反射律・対称律・推移律を満たす）によって、すべての順序集合を類別します。
- この同値類のそれぞれを**順序型**と呼びます。

記号では、順序型を <img src="tmp/442feb3aa71e0bcc2fff58db5de3e28f.png" class="math-inline" /> や単に <img src="tmp/087d054f71c5c1407b2cc03af3498c4e.png" class="math-inline" /> などと書くことがあります。

__直感的な意味__

- 順序型は、順序集合の「形」や「パターン」を表すラベルのようなものです。
- 順序同型な順序集合は**同じ順序型**を持ち、順序同型でないものは**異なる順序型**を持ちます。

__具体例（全順序集合の場合）__

全順序集合の順序型は特に重要で、**順序数（ordinal number）** と深く関係します。

1. **有限順序型**  
   - 要素数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> の全順序集合の順序型は、自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> で表されることが多い。  
   - 例： <img src="tmp/1d52550eb8e30bbbeb233f8505e520e2.png" class="math-inline" />。

2. **可算無限の順序型**  
   - 自然数全体 <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" /> の順序型は <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />（オメガ）と呼ばれる。  
   - 例： <img src="tmp/091de236faabc63ce33cf29dbf1e7445.png" class="math-inline" /> や <img src="tmp/2a9a74be1b88a507bca22ff741b1ee93.png" class="math-inline" /> はどちらも順序型 <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />。

3. **整数全体の順序型**  
   - <img src="tmp/5608f64ad5726b0cec3f4b4905f05ef3.png" class="math-inline" /> の順序型は <img src="tmp/1892f4c70104ffdfcf0e11ff2097ed4f.png" class="math-inline" />（ゼータ）などと書かれることがある。  
   - 最小元がなく、両方向に無限に伸びる構造。

4. **有理数全体の順序型**  
   - <img src="tmp/b44755cfa1b3f0f2e9af4bc0a10f42d4.png" class="math-inline" /> の順序型は <img src="tmp/d045575f98043b44b3b0b529e350b213.png" class="math-inline" />（エータ）と呼ばれる。  
   - 「稠密（dense）かつ可算」という特徴的な構造を持つ。

5. **実数全体の順序型**  
   - <img src="tmp/028e8e6759d744985e5119ebfce2af99.png" class="math-inline" /> の順序型は <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> などと書かれることもあるが、標準的な記号はない。  
   - 連続体濃度を持ち、完備な全順序集合。

### 3. 順序同型と順序型の性質

__(1) 順序同型は同値関係__

- 反射律：恒等写像は順序同型。
- 対称律： <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> が順序同型なら逆写像 <img src="tmp/626b94f7abf48e332446d92b9a8a1fad.png" class="math-inline" /> も順序同型。
- 推移律： <img src="tmp/c0f904752233741fed57139839fdf3ec.png" class="math-inline" /> と <img src="tmp/3b6447189b0cd21af1df5cfdc0d4b658.png" class="math-inline" /> が順序同型なら、合成 <img src="tmp/636881b0ded84102543487ddb0fe2290.png" class="math-inline" /> も順序同型。

したがって、順序同型は**同値関係**であり、順序型はその同値類です。

__(2) 順序同型なら集合としても対等__

- 順序同型写像は全単射なので、**順序同型な順序集合は必ず対等（同じ濃度）** です。
- 逆は成り立ちません（例：<img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> は対等だが順序同型ではない）。

__(3) 順序型と濃度__

- 順序型は**順序構造**を、濃度は**集合の大きさ**を表します。
- 同じ順序型なら同じ濃度ですが、同じ濃度でも異なる順序型を持つことがあります。
- 例：  
  - <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> はどちらも濃度 <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> だが、順序型は異なる（<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> と <img src="tmp/1892f4c70104ffdfcf0e11ff2097ed4f.png" class="math-inline" />）。

### 4. なぜ順序同型・順序型が必要なのか

__(1) 順序構造の「形」を比較するため__

- 集合の対等（全単射）だけでは、「最小元があるか」「稠密か」「完備か」といった**順序的な性質**は保存されません。
- 順序同型は、これらの性質まで含めて「同じ形か」を判定するための概念です。

__(2) 順序数理論の基礎__

- 整列集合（well-ordered set）の順序型は**順序数（ordinal number）** と呼ばれ、無限の大きさを測る基本的な道具です。
- 順序数は、整列集合の順序同型類として定義されます。

__(3) モデル理論・記述集合論での応用__

- 与えられた順序型を持つ構造がどのような論理式を満たすか、という問題はモデル理論で重要です。
- 実数直線や有理数直線の順序型は、記述集合論や位相群論でも頻繁に現れます。

## 全順序集合

**全順序集合（totally ordered set / linearly ordered set）** は、集合の要素の間に「順序」が定まっており、**任意の2つの要素が必ず比較可能**であるような順序集合です。  
数学的には、部分順序集合の定義に「全順序性」を加えたものとして定義されます。

### 1. 全順序集合の定義

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の**二項関係** <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> が次の4条件を満たすとき、<img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> を**全順序集合**といいます。

1. **反射律**：任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対し <img src="tmp/d88e1e9cda96f421a94c273d5390f6ed.png" class="math-inline" />。
2. **推移律**：任意の <img src="tmp/1daf0149a591c5c099a6d194ce27f91b.png" class="math-inline" /> に対し、<img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> かつ <img src="tmp/9cf142fe3938e0d895dee0471b462221.png" class="math-inline" /> ならば <img src="tmp/6ba64e25a1e512afee5c0de7cea5fc11.png" class="math-inline" />。
3. **反対称律**：任意の <img src="tmp/aafcdd2e8da6043624a2be58ad91ff0e.png" class="math-inline" /> に対し、<img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> かつ <img src="tmp/708596235cba73cab553a9bb5933227c.png" class="math-inline" /> ならば <img src="tmp/17f4448b0ccf706290b87890161f80de.png" class="math-inline" />。
4. **全順序性（totality / linearity）**：  
   任意の <img src="tmp/aafcdd2e8da6043624a2be58ad91ff0e.png" class="math-inline" /> に対し、<img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> または <img src="tmp/708596235cba73cab553a9bb5933227c.png" class="math-inline" /> の少なくとも一方が成り立つ。

このとき、<img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> を**全順序関係（total order / linear order）** と呼びます。

__記号と用語__

- <img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" />：  
  - 「<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> 以下」「<img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> 以上」などと読む。
- <img src="tmp/24c9e6d4b22616cc4d909e1612426371.png" class="math-inline" />：  
  - <img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> かつ <img src="tmp/7e51fc5af1798e32ebc66e288b627fbd.png" class="math-inline" /> のとき（**狭義の順序**）。
- **線形順序（linear order）**：  
  - 全順序と同じ意味で使われることが多い。要素が「一直線上に並ぶ」イメージ。

### 2. 全順序集合の具体例

__(1) 数の集合（通常の大小）__

- <img src="tmp/d1f0582382fcd4b54172c474893ecd9d.png" class="math-inline" /> ：実数全体に通常の大小関係を入れたもの。  
  → 任意の2実数は大小比較可能なので全順序集合。
- <img src="tmp/914062146d83bd9e6efa56e7f34ab08b.png" class="math-inline" /> ：有理数全体も同様。
- <img src="tmp/8f8a1974eaba6382e027538fe98d3e6b.png" class="math-inline" /> ：自然数全体も全順序集合。
- <img src="tmp/bee122edd4ade678e3909141329cc055.png" class="math-inline" /> ：整数全体も全順序集合。

これらはすべて、**数直線**のように要素が一直線上に並ぶ構造を持ちます。

__(2) 辞書式順序（lexicographic order）__

- 有限列や文字列に「辞書順」を入れると全順序集合になります。
- 例：アルファベット順に並べた単語の集合（"apple" < "banana" など）。

__(3) 順序数（ordinal numbers）__

- 順序数は**整列集合**（後述）であり、特に全順序集合です。
- 例： <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />（自然数全体の順序型）、<img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" />、<img src="tmp/838230f39aa4893752cb4d57f76fdc99.png" class="math-inline" /> など。

### 3. 全順序集合と部分順序集合の違い

- **部分順序集合（poset）**：  
  反射律・推移律・反対称律を満たすが、**比較不能な要素**が存在し得る。
- **全順序集合**：  
  部分順序集合の条件に加えて、**任意の2元が比較可能**。

例：
- べき集合 <img src="tmp/f272b47ed201b6735c5ad7003f509d4c.png" class="math-inline" /> は部分順序集合だが、全順序ではない  
  （ <img src="tmp/275f0afd9bd9a2ec101008af4cd2b9ff.png" class="math-inline" /> と <img src="tmp/4046fb3f07271b5adf8d1bf744ca5f50.png" class="math-inline" /> は比較不能）。
- <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" /> は全順序集合（任意の2自然数は大小比較可能）。

### 4. 全順序集合の基本的な概念

全順序集合では、部分順序集合と同じ概念（最大元・最小元、上界・下界など）が定義されますが、**全順序性により性質が強まる**ことが多いです。

__(1) 最大元・最小元__

- 全順序集合では、最大元・最小元が存在すれば一意です。
- 例：<img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" /> の最小元は 1、最大元は存在しない。

__(2) 上界・下界・上限・下限__

- 全順序集合では、上界・下界の構造が比較的単純になります。
- 特に**実数直線** <img src="tmp/028e8e6759d744985e5119ebfce2af99.png" class="math-inline" /> では、有界な部分集合は必ず上限・下限を持ちます（実数の完備性）。

__(3) 鎖（chain）__

- 全順序集合そのものが**鎖**です（任意の2元が比較可能）。
- 部分順序集合の中の「全順序な部分集合」を鎖と呼ぶことがありますが、全順序集合では全体が鎖です。

### 5. 整列集合（well-ordered set）

全順序集合のうち、特に重要なサブクラスが**整列集合**です。

__定義__

全順序集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> が**整列集合**であるとは、  
任意の空でない部分集合 <img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> が**最小元**を持つことをいいます。

- 例：<img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" /> は整列集合（任意の空でない自然数の集合には最小元がある）。
- 例：<img src="tmp/5608f64ad5726b0cec3f4b4905f05ef3.png" class="math-inline" /> は整列集合ではない（負の整数全体には最小元がない）。
- 例：<img src="tmp/b44755cfa1b3f0f2e9af4bc0a10f42d4.png" class="math-inline" /> も整列集合ではない（開区間などに最小元がない）。

__順序数との関係__

- 整列集合の順序同型類を**順序数（ordinal number）** と呼びます。
- 順序数は無限の大きさを測る基本的な道具であり、集合論の基盤となります。

### 6. なぜ全順序集合が必要なのか

__(1) 「一直線に並ぶ」構造をモデル化するため__

- 数直線、時間の流れ、辞書順、優先順位など、現実世界の多くの「順序」は全順序です。
- 全順序集合は、こうした**線形な順序構造**を数学的に扱うための枠組みです。

__(2) 解析学の基礎__

- 実数直線 <img src="tmp/028e8e6759d744985e5119ebfce2af99.png" class="math-inline" /> は全順序かつ完備であり、数列の極限、関数の連続性、微分・積分など、解析学のほぼすべての概念がこの構造に依存しています。

__(3) 整列可能定理と選択公理__

- **整列可能定理**（任意の集合に整列順序が存在する）は選択公理と同値であり、集合論の重要な結果です。
- これにより、任意の集合を「一直線に並べる」ことができることが保証されます（ただし具体的な順序は明示されない）。

__(4) 順序数理論・超限帰納法__

- 順序数は整列集合の順序型として定義され、超限帰納法・超限再帰の基礎となります。
- これは数学の証明手法や再帰的定義を無限に拡張するための強力な道具です。


## 最大元、最小元、極大元、極小元、上限、下限

**最大元・最小元・極大元・極小元・上限・下限**は、順序集合（特に部分順序集合）において、部分集合の「端」や「境界」を表す概念です。  
以下、順序集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> とその部分集合 <img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> に対して定義します。

### 1. 最大元（maximum element）と最小元（minimum element）

__最大元の定義__

元 <img src="tmp/9507e0235f1b54262f27fa314e741f93.png" class="math-inline" /> が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**最大元**であるとは、  
すべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> に対して <img src="tmp/f295d319f16567bf5ce1423a190d13f0.png" class="math-inline" /> が成り立つことをいいます。

- 記号：<img src="tmp/2424f8ac2bfca3e79399e8daf2a806a4.png" class="math-inline" /> などと書く。
- 意味：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の要素の中で「最も大きい」元。

__最小元の定義__

元 <img src="tmp/9507e0235f1b54262f27fa314e741f93.png" class="math-inline" /> が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**最小元**であるとは、  
すべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> に対して <img src="tmp/24b42b7ecec14bb0186ff6f006cb1b20.png" class="math-inline" /> が成り立つことをいいます。

- 記号：<img src="tmp/3d0360549d605677037e82842522bdfc.png" class="math-inline" /> などと書く。
- 意味：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の要素の中で「最も小さい」元。

__性質__

- 存在すれば**一意**です（反対称律より）。
- 例：
  - <img src="tmp/07cc1897e2aa76f97cf2f059400351ff.png" class="math-inline" />（通常の大小）なら、<img src="tmp/bab9936c5119e5b800a5d0fdb0f32b9b.png" class="math-inline" />。
  - <img src="tmp/91495507fc607b535d5db82ba4e8ae4b.png" class="math-inline" /> なら、最小元も最大元も存在しない。

### 2. 極大元（maximal element）と極小元（minimal element）

__極大元の定義__

元 <img src="tmp/9507e0235f1b54262f27fa314e741f93.png" class="math-inline" /> が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**極大元**であるとは、  
「<img src="tmp/3304866da369e7deb1f91d3aef5053d6.png" class="math-inline" /> となる <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> が存在しない」ことをいいます。

- 言い換え：<img src="tmp/24b42b7ecec14bb0186ff6f006cb1b20.png" class="math-inline" /> となる <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> は <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> 自身に限られる。
- 意味：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の中で「それより大きい元がない」元。

__極小元の定義__

元 <img src="tmp/9507e0235f1b54262f27fa314e741f93.png" class="math-inline" /> が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**極小元**であるとは、  
「<img src="tmp/676d4f1e821d92ab48f2e61b6a7dcc15.png" class="math-inline" /> となる <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> が存在しない」ことをいいます。

- 言い換え：<img src="tmp/f295d319f16567bf5ce1423a190d13f0.png" class="math-inline" /> となる <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> は <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> 自身に限られる。
- 意味：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の中で「それより小さい元がない」元。

__最大元・最小元との違い__

- **最大元**は「すべての元より大きい」、**極大元**は「それより大きい元がない」。
- 最大元は極大元ですが、逆は必ずしも成り立ちません。
- 例：<img src="tmp/163b408cf9150320b45a522714569d49.png" class="math-inline" /> で、順序を「整除関係 <img src="tmp/c9713ce2df40f406d1f1ff0bfa4ff87e.png" class="math-inline" />」とする。
  - 極大元：3, 4（これらより大きい元はない）。
  - 最大元：存在しない（3 と 4 は比較不能）。

### 3. 上界（upper bound）と下界（lower bound）

__上界の定義__

元 <img src="tmp/9dcb0ce7aa80bed81144a7fa4ad9043e.png" class="math-inline" /> が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**上界**であるとは、  
すべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> に対して <img src="tmp/aa554c18bc824da199690a0ac68ee3bd.png" class="math-inline" /> が成り立つことをいいます。

- 意味：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の要素を「上から抑える」元。
- 例：<img src="tmp/91495507fc607b535d5db82ba4e8ae4b.png" class="math-inline" /> の上界は、1 以上のすべての実数。

__下界の定義__

元 <img src="tmp/2ea8ea80fd893716f7d88bb0dfd94c3b.png" class="math-inline" /> が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**下界**であるとは、  
すべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> に対して <img src="tmp/8c4a58686d818a44a84acef49e7b75a3.png" class="math-inline" /> が成り立つことをいいます。

- 意味：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の要素を「下から支える」元。
- 例：<img src="tmp/91495507fc607b535d5db82ba4e8ae4b.png" class="math-inline" /> の下界は、0 以下のすべての実数。

__性質__

- 上界・下界は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の外の元でも構いません。
- 上界の集合を <img src="tmp/891cefa5bec891cbec2017bf8beca57e.png" class="math-inline" />、下界の集合を <img src="tmp/600eda729f996ba8cd609c7eed2cacc0.png" class="math-inline" /> と書くことがあります。

### 4. 上限（supremum）と下限（infimum）

__上限の定義__

<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の上界全体の集合 <img src="tmp/891cefa5bec891cbec2017bf8beca57e.png" class="math-inline" /> が**最小元**を持つとき、その元を <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**上限**といいます。

- 記号：<img src="tmp/f179083ab7e8aba75cff85adb7173f92.png" class="math-inline" />（supremum）。
- 意味：上界の中で「最も小さい」元。
- 存在すれば**一意**です。

__下限の定義__

<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の下界全体の集合 <img src="tmp/600eda729f996ba8cd609c7eed2cacc0.png" class="math-inline" /> が**最大元**を持つとき、その元を <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**下限**といいます。

- 記号：<img src="tmp/33e75204617311a197191c1ceead5b4c.png" class="math-inline" />（infimum）。
- 意味：下界の中で「最も大きい」元。
- 存在すれば**一意**です。

__性質__

- <img src="tmp/f179083ab7e8aba75cff85adb7173f92.png" class="math-inline" /> は「最小上界（least upper bound）」、<img src="tmp/33e75204617311a197191c1ceead5b4c.png" class="math-inline" /> は「最大下界（greatest lower bound）」とも呼ばれます。
- 例：
  - <img src="tmp/91495507fc607b535d5db82ba4e8ae4b.png" class="math-inline" /> なら、<img src="tmp/55f7da9ac2937c24476ce9c6e212ee30.png" class="math-inline" />。
  - <img src="tmp/450ae02fcab3f7939a0f4712fcfd3338.png" class="math-inline" /> なら、<img src="tmp/55f7da9ac2937c24476ce9c6e212ee30.png" class="math-inline" />（最大元は 1）。

### 5. 最大元・最小元と上限・下限の関係

- 最大元が存在すれば、それは上限と一致します（<img src="tmp/c119854056005bc7ea017ba4b1049454.png" class="math-inline" />）。
- 最小元が存在すれば、それは下限と一致します（<img src="tmp/a70c1f7ef7d5b8e7804a625e31bb5567.png" class="math-inline" />）。
- 逆は成り立ちません：
  - 例：<img src="tmp/61ecaab281656ad1743eabc4c95f9822.png" class="math-inline" /> では <img src="tmp/b2d22991c6d0733067067215e3bac979.png" class="math-inline" /> だが、最大元は存在しない。

### 6. 具体例での確認

__例1：実数の区間__

- <img src="tmp/cff6cffa62ac5d7a1478923e191429c1.png" class="math-inline" />：  
  - <img src="tmp/bd9a156b2ec1c46132a2eb4ec0a1c215.png" class="math-inline" />。  
  - <img src="tmp/55f7da9ac2937c24476ce9c6e212ee30.png" class="math-inline" />。
- <img src="tmp/61ecaab281656ad1743eabc4c95f9822.png" class="math-inline" />：  
  - 最小元・最大元は存在しない。  
  - <img src="tmp/55f7da9ac2937c24476ce9c6e212ee30.png" class="math-inline" />。

__例2：べき集合の包含順序__

<img src="tmp/370d0b7a1f97fce98a2a6d59bc55410f.png" class="math-inline" />、<img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> に包含関係 <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" /> を入れる。

- <img src="tmp/cafcb6933f1dc23e9fd25237017dd291.png" class="math-inline" />：  
  - 極大元：<img src="tmp/bd66678e1e29ad068753c86bba6f7186.png" class="math-inline" />（いずれも包含関係で比較不能）。  
  - 最大元：存在しない。  
  - 上界：<img src="tmp/7952a039a96398415c8e6161bcd97f66.png" class="math-inline" /> のみ。  
  - <img src="tmp/3fa62ac53b3c9f1bda26831bf0789e73.png" class="math-inline" />。

__例3：整除関係__

<img src="tmp/7ec1af771c1c3ea8605dd60ffdf2a59b.png" class="math-inline" />：<img src="tmp/c9713ce2df40f406d1f1ff0bfa4ff87e.png" class="math-inline" /> を順序とする。

- <img src="tmp/c8bb67ef9129585efe8156f3e1e65f8d.png" class="math-inline" />：  
  - 極大元：3, 4, 6（これらより大きい元はない）。  
  - 最大元：存在しない（3 と 4 は比較不能）。  
  - 上界：12, 24, …（公倍数）。  
  - <img src="tmp/ae68e1fd34bc0149f52576f5abb0203f.png" class="math-inline" />。

## 実数の連続性

**実数の連続性（continuity of the real numbers）** は、実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> が「数直線として隙間なくつながっている」ことを保証する性質です。  
数学的には、**実数の完備性（completeness）** とも呼ばれ、いくつか同値な形で表現されます（上限公理・区間縮小法・有界単調数列の収束など）。

以下、主な定式化とその意味を説明します。

### 1. 上限公理（least upper bound property）

__定義__

実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> は次の性質を持ちます：

> 任意の空でない上に有界な部分集合 <img src="tmp/a67c02c1b28e67f45ee41270f914a445.png" class="math-inline" /> は、**上限（最小上界）** を持つ。

すなわち、<img src="tmp/70ae9ae581001c1f1f062998fd9a6d2a.png" class="math-inline" /> かつある <img src="tmp/87d8e88a7522d5a8ac0637ad4c8ba4ae.png" class="math-inline" /> が存在してすべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> について <img src="tmp/aef40d7ab39eba472cd75560b58e841e.png" class="math-inline" /> ならば、  
ある実数 <img src="tmp/6f0b859c87d79180d255b6d7a792f89b.png" class="math-inline" /> が存在して：
- すべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> について <img src="tmp/dc56b99893f75761ddc30fb04d64206e.png" class="math-inline" />（<img src="tmp/add5bb4fc3bfe26587c101125d12ae33.png" class="math-inline" /> は上界）
- 任意の上界 <img src="tmp/0816f9a6f46a1c29fbc180f1c8726250.png" class="math-inline" /> について <img src="tmp/5cb639191570d5270893f7bc7ae844c6.png" class="math-inline" />（<img src="tmp/add5bb4fc3bfe26587c101125d12ae33.png" class="math-inline" /> は最小の上界）

この <img src="tmp/add5bb4fc3bfe26587c101125d12ae33.png" class="math-inline" /> を <img src="tmp/f179083ab7e8aba75cff85adb7173f92.png" class="math-inline" /> と書きます。

__具体例__

- <img src="tmp/61ecaab281656ad1743eabc4c95f9822.png" class="math-inline" />：上界は 1 以上の実数。最小上界は 1 → <img src="tmp/b2d22991c6d0733067067215e3bac979.png" class="math-inline" />。
- <img src="tmp/be66fe79aa53fbc268d52c9d51151d4c.png" class="math-inline" />：上界は 1 以上の実数。最小上界は 1 → <img src="tmp/b2d22991c6d0733067067215e3bac979.png" class="math-inline" />。
- <img src="tmp/0df1aedc6854a8ecfd550aebaeb575bd.png" class="math-inline" />：有理数の中では最小上界が存在しないが、実数では <img src="tmp/0d73e76addc2ebfd68ff96c58348ecb3.png" class="math-inline" />。

この最後の例が、**有理数 <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> が連続でない（完備でない）** ことの典型です。

### 2. 下限公理（greatest lower bound property）

上限公理と双対的に、次も成り立ちます：

> 任意の空でない下に有界な部分集合 <img src="tmp/a67c02c1b28e67f45ee41270f914a445.png" class="math-inline" /> は、**下限（最大下界）** を持つ。

記号では <img src="tmp/33e75204617311a197191c1ceead5b4c.png" class="math-inline" /> と書きます。

### 3. 有界単調数列の収束（monotone convergence theorem）

__定理__

実数列 <img src="tmp/f93c7049f53ad6f0b91f244d5381a57e.png" class="math-inline" /> が次の条件を満たすとする：

- **単調増加**：<img src="tmp/2b56df7749a18544049f6471d456e865.png" class="math-inline" />
- **上に有界**：ある <img src="tmp/87d8e88a7522d5a8ac0637ad4c8ba4ae.png" class="math-inline" /> が存在してすべての <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> について <img src="tmp/746c4148a7cb7fe2e305f7a737a5debf.png" class="math-inline" />

このとき、<img src="tmp/f93c7049f53ad6f0b91f244d5381a57e.png" class="math-inline" /> は実数に収束する。  
より正確には、極限 <img src="tmp/0433f9321b50f48888da9c479bebbe04.png" class="math-inline" /> が存在し、それは <img src="tmp/03ea906baadb74a81706a3d43a191b3a.png" class="math-inline" /> に等しい。

同様に、単調減少で下に有界な数列も収束します。

__例__

- <img src="tmp/f302414acd95a0cdf2a7120e7f81cca5.png" class="math-inline" />：単調増加、上に有界（1 以下）→ <img src="tmp/e22545fc516c8be231bf8cbb8c9356d6.png" class="math-inline" />。
- <img src="tmp/83b90d06af5581d0706953118c65c61a.png" class="math-inline" />：単調増加、上に有界（<img src="tmp/83e18979b3abea680d64fc6e530bb32c.png" class="math-inline" /> 以下）→ 収束。

### 4. 区間縮小法（nested interval property）

__定理__

閉区間の列 <img src="tmp/afd7e807198d837b8bf675b7b2e283dd.png" class="math-inline" /> が次の条件を満たすとする：

- **縮小**：<img src="tmp/0f5df6713c14d310a22f8e324f9ef802.png" class="math-inline" />
- **長さが 0 に収束**：<img src="tmp/5197b6241f8a1fb6098dd261abd0a105.png" class="math-inline" />

このとき、すべての区間 <img src="tmp/10c8583ee377dd98911963403764c0a7.png" class="math-inline" /> に属する実数が**ちょうど1つ**存在する。  
すなわち、<img src="tmp/0ec9953a7a6b64fa7ec656daba62e91f.png" class="math-inline" /> は1点からなる。

__例__

- <img src="tmp/2e096086268120205eb6f4da0adfa0cf.png" class="math-inline" />：<img src="tmp/fdb66fee1f3c95082415c19084adc69e.png" class="math-inline" />。
- 有理数では成り立たない（例：<img src="tmp/4a8492ec3d7ed021636a7f5cccd72391.png" class="math-inline" /> の共通部分は空）。

### 5. Cauchy列の収束（Cauchy completeness）

__定義__

実数列 <img src="tmp/f93c7049f53ad6f0b91f244d5381a57e.png" class="math-inline" /> が**Cauchy列**であるとは：

> 任意の <img src="tmp/bfe8bbbead05469f40e4beff00daa290.png" class="math-inline" /> に対し、ある <img src="tmp/3b9eb0c5462a710f5a43cfbdce30b469.png" class="math-inline" /> が存在して、  
> すべての <img src="tmp/fd65fecd7d41663384bcde5a99b444fd.png" class="math-inline" /> について <img src="tmp/5c7a7fcdd4b9923a3919e569054954b6.png" class="math-inline" /> が成り立つ。

すなわち、「十分先では項同士が互いに近づく」列です。

__定理（実数の完備性）__

実数においては、**Cauchy列は必ず収束する**。  
すなわち、<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> は**完備距離空間**です。

__例__

- <img src="tmp/f6c7d01a9e128ff8b566ec7358a61073.png" class="math-inline" />：Cauchy列 → 収束（2）。
- 有理数では成り立たない：  
  <img src="tmp/1d11dc50d25d98c4f5cbd7bec398c1ed.png" class="math-inline" /> は有理数のCauchy列だが、極限 <img src="tmp/9d52504917c386d86d0c2e44005258b5.png" class="math-inline" /> は無理数。

### 6. なぜ実数の連続性が重要なのか

__(1) 解析学の基礎__

- 微分・積分、関数の連続性、一様収束など、解析学のほぼすべての定理は実数の連続性に依存しています。
- 例：中間値の定理・最大値定理・一様連続性の定理などは、実数の完備性から導かれます。

__(2) 有理数との違い__

- 有理数 <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> は**稠密**（任意の2有理数の間に別の有理数が存在）ですが、**完備ではない**（Cauchy列が収束しないことがある）。
- 実数は、有理数の「隙間」をすべて埋めたものとして構成できます（Dedekind切断やCauchy列の同値類による構成）。

__(3) 他の完備な空間のモデル__

- 実数の完備性は、<img src="tmp/1cbc190537dcb7b338fa4f5eeb77919b.png" class="math-inline" /> や関数空間（<img src="tmp/8b8c5d1e4a3b51e3b44dadf3435e269e.png" class="math-inline" /> など）の完備性のモデルとなります。
- バナッハ空間・ヒルベルト空間といった無限次元空間でも、「Cauchy列が収束する」という完備性が重要な性質です。

## 擬順序

**擬順序（preorder / quasiorder）** は、集合の要素の間に「順序」を定めた構造のうち、**反射律と推移律**のみを満たすものです。  
部分順序や全順序と異なり、**反対称律を要求しない**ため、「同じでないのに互いに大きいとみなせる」ような関係も許容します。

### 1. 擬順序の定義

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の**二項関係** <img src="tmp/b2ebf7379839345afa8b29aeff5dd3cd.png" class="math-inline" /> が次の2条件を満たすとき、<img src="tmp/97c52d28d71c6a46b6e846733d1d408f.png" class="math-inline" /> を**擬順序集合（preordered set）** といいます。

1. **反射律**：任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対し <img src="tmp/a7a8cfa74f58920203688f6cf39894da.png" class="math-inline" />。
2. **推移律**：任意の <img src="tmp/1daf0149a591c5c099a6d194ce27f91b.png" class="math-inline" /> に対し、<img src="tmp/bc02eaf4de7b723abf3228f45efaa1c1.png" class="math-inline" /> かつ <img src="tmp/dd5b9920c9b23d4481cde80fbd0f4dc0.png" class="math-inline" /> ならば <img src="tmp/a4051adbc2ac4d7becf43c2705dea85c.png" class="math-inline" />。

このとき、<img src="tmp/b2ebf7379839345afa8b29aeff5dd3cd.png" class="math-inline" /> を**擬順序関係（preorder）** と呼びます。

__記号と用語__

- <img src="tmp/bc02eaf4de7b723abf3228f45efaa1c1.png" class="math-inline" />：  
  - 「<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> より小さい（または同等）」「<img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> より大きい（または同等）」などと読む。
- <img src="tmp/fd43c40ff8a2aa5089e6f675c4358eff.png" class="math-inline" />：  
  - <img src="tmp/bc02eaf4de7b723abf3228f45efaa1c1.png" class="math-inline" /> かつ <img src="tmp/0f9cba5a0f672a757f8758c88863f9f7.png" class="math-inline" /> のとき、<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> と <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は**同値（equivalent）** であるという。
- 擬順序集合は、**前順序集合**とも呼ばれます。

### 2. 擬順序と部分順序・同値関係の関係

__(1) 部分順序との違い__

- **部分順序（partial order）** は、反射律・推移律に加えて**反対称律**を満たします：
  - <img src="tmp/bc02eaf4de7b723abf3228f45efaa1c1.png" class="math-inline" /> かつ <img src="tmp/0f9cba5a0f672a757f8758c88863f9f7.png" class="math-inline" /> ならば <img src="tmp/17f4448b0ccf706290b87890161f80de.png" class="math-inline" />。
- 擬順序ではこの反対称律を要求しないため、  
  <img src="tmp/fd43c40ff8a2aa5089e6f675c4358eff.png" class="math-inline" /> であっても <img src="tmp/17f4448b0ccf706290b87890161f80de.png" class="math-inline" /> とは限りません。

__(2) 同値関係との関係__

- 擬順序 <img src="tmp/b2ebf7379839345afa8b29aeff5dd3cd.png" class="math-inline" /> から自然に**同値関係** <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> が誘導されます：
  - <img src="tmp/7a477e8ece87a9e412fd7143b0b21a04.png" class="math-inline" />。
- この同値関係で割った商集合 <img src="tmp/7957ba082625c33993d7806cf8a180aa.png" class="math-inline" /> には、自然に**部分順序**が入ります（後述）。

### 3. 擬順序の具体例

__(1) 実数上の「≤」関係__

- <img src="tmp/028e8e6759d744985e5119ebfce2af99.png" class="math-inline" /> は反射律・推移律・反対称律をすべて満たすので、**部分順序（かつ全順序）** です。
- したがって擬順序でもありますが、通常はより強い「順序」として扱われます。

__(2) ベクトルの成分ごとの比較__

- <img src="tmp/3a03c7f8694b4cd988eedcb32a3b0f52.png" class="math-inline" /> とし、  
  <img src="tmp/e8a46e39a7c7c62a4c8025bb8a5bd1e5.png" class="math-inline" /> と定義する。
- 反射律・推移律は満たすが、反対称律は成り立たない（例：<img src="tmp/9e907f4f972da1c05d15440f620f77b2.png" class="math-inline" /> かつ <img src="tmp/333d0d7974f1c1f0c00b89f49fe66c20.png" class="math-inline" /> は偽）。  
  実はこれは部分順序ですが、一般には擬順序の例として挙げられることがあります。

より典型的な擬順序の例は次のようなものです。

__(3) 整除関係（倍数関係）__

- <img src="tmp/6e82299800e7ce873f7e86662e3dd8de.png" class="math-inline" /> とし、<img src="tmp/d55e19be9120f9c3083289352216f091.png" class="math-inline" />（<img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> は <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> を割り切る）と定義する。
- 反射律：<img src="tmp/564ded13611252b1108f6fceee640b84.png" class="math-inline" />。
- 推移律：<img src="tmp/c9713ce2df40f406d1f1ff0bfa4ff87e.png" class="math-inline" /> かつ <img src="tmp/495e18e7921ef04213acb0d7e3beddb5.png" class="math-inline" /> なら <img src="tmp/83cdfab97ccebaa4b471ffbbc9542e90.png" class="math-inline" />。
- 反対称律は成り立つ（<img src="tmp/c9713ce2df40f406d1f1ff0bfa4ff87e.png" class="math-inline" /> かつ <img src="tmp/c6818f5f1d0870d4c7e95edb950f5818.png" class="math-inline" /> なら <img src="tmp/286578ad1051d203e1342cae25965ff5.png" class="math-inline" />）ので、実はこれは部分順序です。  
  より「擬順序らしい」例は次のようなものです。

__(4) 漸近的な大小関係（ランダウの記号）__

- 関数の集合 <img src="tmp/65480d9b9b29b6185e8b058d314761fb.png" class="math-inline" /> に対し、  
  <img src="tmp/e3956976788b1d7f47a6b4575062cf73.png" class="math-inline" />（ある定数 <img src="tmp/f435ec3a88362561127ca7edeeb111c6.png" class="math-inline" /> と <img src="tmp/451dde9af0823a245e3fd8c015c5437f.png" class="math-inline" /> が存在して <img src="tmp/c3a1dedb12f77d834cb6cc226cdd7f81.png" class="math-inline" />）と定義する。
- 反射律：<img src="tmp/7dea025ad449d35692bad608ff89713f.png" class="math-inline" />。
- 推移律：<img src="tmp/ce8e9480bdf012baef0ef2998f008099.png" class="math-inline" /> かつ <img src="tmp/0a08eaab13b8ebba4c9c5ee45be5a003.png" class="math-inline" /> なら <img src="tmp/fc315e92da026df266726d38616e3296.png" class="math-inline" />。
- 反対称律は成り立たない：  
  例：<img src="tmp/e8912394afd318807a1604f5955d8066.png" class="math-inline" /> とすると、<img src="tmp/43b8449fce49c05662dd9d60fb136d68.png" class="math-inline" /> かつ <img src="tmp/55f466c8bbb314d6e516c0e33081788f.png" class="math-inline" /> だが <img src="tmp/153aa91dc69a64186de4aacab98432dc.png" class="math-inline" />。
- したがってこれは**擬順序**であり、同値関係 <img src="tmp/40159b884e9f87de07de26b14baa717e.png" class="math-inline" />（同程度のオーダー）を誘導します。

__(5) 優先順位や好みの関係__

- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> を選択肢の集合とし、<img src="tmp/bc02eaf4de7b723abf3228f45efaa1c1.png" class="math-inline" /> を「<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> より好ましくない（または同等）」と解釈する。
- 反射律（自分自身とは同等）と推移律（一貫性）は自然だが、  
  <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> と <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> が「同等に好ましい」からといって同じ選択肢とは限らない → 反対称律は不要。
- これは**擬順序**の典型例です。

### 4. 擬順序から部分順序への商構成

擬順序 <img src="tmp/b2ebf7379839345afa8b29aeff5dd3cd.png" class="math-inline" /> が与えられたとき、同値関係


<div class="math-display-container"><img src="tmp/2b57baf05bb7d83c8cc1884618636771.png" class="math-display" /></div>


で割った商集合 <img src="tmp/7957ba082625c33993d7806cf8a180aa.png" class="math-inline" /> を考えます。

この商集合上に、自然な順序


<div class="math-display-container"><img src="tmp/177518e453f31273a5d1c1ed7551b161.png" class="math-display" /></div>


を入れると、<img src="tmp/ab4f299945a8982d579b4fabc1660913.png" class="math-inline" /> は**部分順序集合**になります。

- 反射律・推移律は <img src="tmp/b2ebf7379839345afa8b29aeff5dd3cd.png" class="math-inline" /> から継承されます。
- 反対称律：<img src="tmp/03a4962dcd4c21d86ec23a8bf8d6d3b0.png" class="math-inline" /> かつ <img src="tmp/667d776df6cccf3ea986772d97045984.png" class="math-inline" /> なら、定義より <img src="tmp/fd43c40ff8a2aa5089e6f675c4358eff.png" class="math-inline" /> なので <img src="tmp/71141234599704237d80cfeaa2d1935c.png" class="math-inline" />。

このように、擬順序は **「同値類のレベルでは部分順序」** を与える構造とみなせます。

### 5. なぜ擬順序が必要なのか

__(1) 反対称律が自然でない場面のモデル化__

- 漸近的評価、優先順位、同値類の比較など、  
  「同じでないが同等とみなしたい」状況は多くあります。
- 擬順序は、こうした**緩い順序関係**を数学的に扱うための枠組みです。

__(2) 圏論における前順序圏__

- 圏論では、対象の間に射が高々1つしかない圏を**前順序圏（preorder category）** と呼びます。
- これはまさに擬順序集合に対応し、圏論的な視点から順序構造を統一的に扱うことができます。

__(3) 理論計算機科学・形式手法__

- プログラムの精緻化関係、型の部分型関係、プロセスの模倣関係など、  
  多くの「順序的な関係」は擬順序として定式化されます。
- 特に、反対称律を課さないことで、より柔軟なモデルが可能になります。


## 演習

### 問題


__1. 順序集合の分類__

集合 <img src="tmp/85bcfdc85cb185d1adb8653d097a4429.png" class="math-inline" /> 上の二項関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> を次のように定義する。



<div class="math-display-container"><img src="tmp/190b67e9ed5445b9b0f874550ba265c1.png" class="math-display" /></div>



(1) 反射律・推移律・反対称律・全順序性のそれぞれについて、成り立つかどうかを判定せよ。  
(2) <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は前順序集合・部分順序集合・全順序集合のどれか、理由とともに答えよ。  
(3) 最大元・最小元・極大元・極小元をすべて求めよ。  
(4) 部分集合 <img src="tmp/121d4c699ddede0e4cf1ca1c5eb256ad.png" class="math-inline" /> の上界・下界・上限・下限を求めよ（存在しない場合はその旨を述べよ）。

__2. 順序同型と順序型__

次の順序集合について、順序同型かどうかを判定し、順序型（可能なら記号で）を答えよ。

(1) <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" /> と <img src="tmp/43f1c4d5f096948385a0fde7185c4662.png" class="math-inline" />  
(2) <img src="tmp/5608f64ad5726b0cec3f4b4905f05ef3.png" class="math-inline" /> と <img src="tmp/b44755cfa1b3f0f2e9af4bc0a10f42d4.png" class="math-inline" />  
(3) <img src="tmp/451f07222d20ef0263efce4329b2c6b9.png" class="math-inline" /> と <img src="tmp/692f8d78263f169cb7940e8ff5f12c7e.png" class="math-inline" />（辞書順）  
(4) <img src="tmp/f272b47ed201b6735c5ad7003f509d4c.png" class="math-inline" /> と <img src="tmp/a9d38f4f575bfc8ee1e29d999bee7f45.png" class="math-inline" />

__3. 実数の連続性（上限公理）__

次の実数の部分集合 <img src="tmp/a67c02c1b28e67f45ee41270f914a445.png" class="math-inline" /> について、上限 <img src="tmp/f179083ab7e8aba75cff85adb7173f92.png" class="math-inline" /> と下限 <img src="tmp/33e75204617311a197191c1ceead5b4c.png" class="math-inline" /> を求めよ。  
また、最大元・最小元が存在するかどうかも答えよ。

(1) <img src="tmp/cff6cffa62ac5d7a1478923e191429c1.png" class="math-inline" />  
(2) <img src="tmp/61ecaab281656ad1743eabc4c95f9822.png" class="math-inline" />  
(3) <img src="tmp/be66fe79aa53fbc268d52c9d51151d4c.png" class="math-inline" />  
(4) <img src="tmp/0df1aedc6854a8ecfd550aebaeb575bd.png" class="math-inline" />

__4. 有界単調数列の収束__

次の実数列 <img src="tmp/f93c7049f53ad6f0b91f244d5381a57e.png" class="math-inline" /> が収束するかどうかを判定し、収束するなら極限値を求めよ。  
また、その収束が「有界単調数列の収束定理」によって保証されることを説明せよ。

(1) <img src="tmp/c3819ae5814d38c18687ba2eac37f026.png" class="math-inline" />  
(2) <img src="tmp/82bf0e688ab80743555f0d93757938f5.png" class="math-inline" />  
(3) <img src="tmp/7419e43ef2ea5b5fa6d4928b4c9fcdb6.png" class="math-inline" />  
(4) <img src="tmp/005165defcf15d1b8e5df5a380fb7ea4.png" class="math-inline" />

__5. Cauchy列と完備性__

次の実数列 <img src="tmp/f93c7049f53ad6f0b91f244d5381a57e.png" class="math-inline" /> がCauchy列かどうかを判定せよ。  
また、Cauchy列である場合はその収束先を求めよ。

(1) <img src="tmp/ee4d8c572decf71a85f60479e87b523f.png" class="math-inline" />  
(2) <img src="tmp/165253952aa313093a186404ce9f9920.png" class="math-inline" />  
(3) <img src="tmp/82bf0e688ab80743555f0d93757938f5.png" class="math-inline" />  
(4) <img src="tmp/005165defcf15d1b8e5df5a380fb7ea4.png" class="math-inline" />

__6. 擬順序と商構成__

関数の集合 <img src="tmp/65480d9b9b29b6185e8b058d314761fb.png" class="math-inline" /> 上に、次の関係を定義する。



<div class="math-display-container"><img src="tmp/5b2ede3cc660d77a4fbe15974d2316f6.png" class="math-display" /></div>



(1) <img src="tmp/97c52d28d71c6a46b6e846733d1d408f.png" class="math-inline" /> が擬順序集合であることを示せ（反射律・推移律を確認せよ）。  
(2) 反対称律が成り立たないことを、具体例を用いて示せ。  
(3) 同値関係 <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> を <img src="tmp/53f35bc07bdfaa99d44f961843f2e9ae.png" class="math-inline" /> で定義するとき、  
　 <img src="tmp/b37fed6113ee92ab0b392f7c3636fb88.png" class="math-inline" /> はどのような関係か（ランダウの記号で表せ）。  
(4) 商集合 <img src="tmp/7957ba082625c33993d7806cf8a180aa.png" class="math-inline" /> 上に自然な順序 <img src="tmp/fe069b7cd84f9650dabae8c044e15898.png" class="math-inline" /> を入れると、  
　 <img src="tmp/ab4f299945a8982d579b4fabc1660913.png" class="math-inline" /> は部分順序集合になることを説明せよ。

__7. 総合問題（順序集合と実数の連続性）__

実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> に通常の大小関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> を入れた全順序集合 <img src="tmp/028e8e6759d744985e5119ebfce2af99.png" class="math-inline" /> を考える。

(1) 部分集合 <img src="tmp/085c3e1411ceeb09763df76ed9f7e4c7.png" class="math-inline" /> について、<img src="tmp/f179083ab7e8aba75cff85adb7173f92.png" class="math-inline" /> と <img src="tmp/33e75204617311a197191c1ceead5b4c.png" class="math-inline" /> を求めよ。  
(2) 有理数全体 <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> に制限した順序集合 <img src="tmp/b44755cfa1b3f0f2e9af4bc0a10f42d4.png" class="math-inline" /> では、<img src="tmp/4bea7101cbb8efae50cb963ee6bd224a.png" class="math-inline" /> の上限が存在しないことを説明せよ。  
(3) このことから、有理数 <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> が「連続でない（完備でない）」ことをどのように理解できるか、簡潔に述べよ。  
(4) 実数の連続性（完備性）が、解析学においてなぜ重要であるかを、1〜2文でまとめよ。


### 解答

以下、各問題の解答を示します。

__1. 順序集合の分類（整除関係）__

<img src="tmp/85bcfdc85cb185d1adb8653d097a4429.png" class="math-inline" />、<img src="tmp/bd7ee4b4ce045e9acb8bb3dc51b9f83a.png" class="math-inline" />。

__(1) 各法則の判定__

- **反射律**：任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について <img src="tmp/ff4d798c3cfdf36dada98eac157c1ca3.png" class="math-inline" /> は真（1,2,3,4 は自分自身を割り切る）。→ **成立**
- **推移律**：<img src="tmp/644da28a9530fde13c2d3868808d15cb.png" class="math-inline" /> かつ <img src="tmp/1b20f06a04d931a9a921ef32a42e8438.png" class="math-inline" /> なら <img src="tmp/a11dc20550a4ca83485199c27d1a1c3e.png" class="math-inline" /> は整数の整除の性質より真。→ **成立**
- **反対称律**：<img src="tmp/644da28a9530fde13c2d3868808d15cb.png" class="math-inline" /> かつ <img src="tmp/e661fc5595a0ca54656cf6142c476254.png" class="math-inline" /> なら <img src="tmp/a8d00a85f130e3342a5e1403e04434ae.png" class="math-inline" /> は整数の整除の性質より真。→ **成立**
- **全順序性**：任意の <img src="tmp/76a339370d44c380e6fa6526a6a20485.png" class="math-inline" /> について <img src="tmp/644da28a9530fde13c2d3868808d15cb.png" class="math-inline" /> または <img src="tmp/e661fc5595a0ca54656cf6142c476254.png" class="math-inline" /> か？  
  例：2 と 3 は互いに割り切れないので比較不能。→ **不成立**

__(2) 分類__

- 反射律・推移律・反対称律を満たすので、**部分順序集合（poset）**。
- 全順序性は成り立たないので、全順序集合ではない。
- 前順序集合でもあるが、通常は最も強い「部分順序集合」と答える。

**答**：部分順序集合（全順序ではない）。

__(3) 最大元・最小元・極大元・極小元__

整除関係を書き下す：
- 1 は 1,2,3,4 を割り切る（1 ≤ 1,2,3,4）。
- 2 は 2,4 を割り切る。
- 3 は 3 のみ。
- 4 は 4 のみ。

- **最小元**：1（すべての元を割り切る）。
- **最大元**：存在しない（例：3 と 4 は比較不能で、どちらも他より大きくはない）。
- **極小元**：2,3（これらより小さい元はない：1 は 2,3 を割り切るが 1<2,3 なので「より小さい」とは言えない。狭義の順序では 1<2,3 だが、極小の定義は「それより小さい元がない」なので、2,3 は極小）。
  - 厳密には：2 に対して <img src="tmp/79b2b9dff279043f0e0847ec24a630ce.png" class="math-inline" /> となる <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は 1 のみだが、1 は 2 を割り切るので 1 ≤ 2。しかし 1=2 ではないので 1<2。よって 2 は極小ではない？  
    → ここは定義の解釈によるが、通常「整除関係での極小元」は 2,3,4 とされることが多い（1 はすべてを割り切るので「最小元」であり極小でもあるが、最大・最小と極大・極小は別概念）。  
    より安全な答え：**極小元は 2,3,4**（これらより真に小さい元はない）、**極大元は 3,4**（これらより真に大きい元はない）。
- **極大元**：3,4（これらより大きい元はない）。

**答**：
- 最小元：1
- 最大元：なし
- 極小元：2,3,4
- 極大元：3,4

__(4) <img src="tmp/121d4c699ddede0e4cf1ca1c5eb256ad.png" class="math-inline" /> の上界・下界・上限・下限__

- **上界**：2 と 3 をともに割り切る元は 1 のみ？ → 1 は 2,3 を割り切るので 2 ≤ 1, 3 ≤ 1？  
  整除関係では「<img src="tmp/bd7ee4b4ce045e9acb8bb3dc51b9f83a.png" class="math-inline" />」なので、上界 <img src="tmp/0816f9a6f46a1c29fbc180f1c8726250.png" class="math-inline" /> は「すべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> について <img src="tmp/1de0ad71c16f3ebb3c5c02f5769b547e.png" class="math-inline" />」、すなわち <img src="tmp/0816f9a6f46a1c29fbc180f1c8726250.png" class="math-inline" /> は 2 と 3 の公倍数。  
  <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 内の公倍数は なし（最小公倍数 6 は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> にない）。よって **上界は存在しない**。
- **下界**：2 と 3 をともに割り切る元は 1 のみ。よって下界は 1。
- **上限**：上界が存在しないので上限も存在しない。
- **下限**：下界の集合 <img src="tmp/275f0afd9bd9a2ec101008af4cd2b9ff.png" class="math-inline" /> の最大元は 1。よって <img src="tmp/eec4d3542604b1b802b7ed926eb3bbdf.png" class="math-inline" />。

**答**：
- 上界：なし
- 下界：1
- 上限：なし
- 下限：1

__2. 順序同型と順序型__

__(1) <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" /> と <img src="tmp/43f1c4d5f096948385a0fde7185c4662.png" class="math-inline" />__

- 写像 <img src="tmp/0e319c5b6138c35cae66b37c0ea34b92.png" class="math-inline" /> は全単射で、順序を保つ（<img src="tmp/e84d929140614f47007ecacdbc9c4c3b.png" class="math-inline" />）。
- よって**順序同型**。
- 順序型：どちらも最小元を持つ可算無限の全順序集合なので、順序型は <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />（自然数型）。

**答**：順序同型、順序型 <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />。

__(2) <img src="tmp/5608f64ad5726b0cec3f4b4905f05ef3.png" class="math-inline" /> と <img src="tmp/b44755cfa1b3f0f2e9af4bc0a10f42d4.png" class="math-inline" />__

- <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> は「両方向に無限だが離散的」、<img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> は「稠密」。
- 順序同型なら順序構造（最小元の有無、稠密性など）が一致する必要があるが、<img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> は稠密でないので同型ではない。
- 順序型：<img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> は <img src="tmp/1892f4c70104ffdfcf0e11ff2097ed4f.png" class="math-inline" />、<img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> は <img src="tmp/d045575f98043b44b3b0b529e350b213.png" class="math-inline" />。

**答**：順序同型ではない（構造が異なる）。

__(3) <img src="tmp/451f07222d20ef0263efce4329b2c6b9.png" class="math-inline" /> と <img src="tmp/692f8d78263f169cb7940e8ff5f12c7e.png" class="math-inline" />（辞書順）__

- 3元の全順序集合はすべて順序同型（全単射で順序を保つ写像が存在）。
- 順序型：有限の全順序集合の順序型は要素数 3 で表すことが多い。

**答**：順序同型、順序型 3。

__(4) <img src="tmp/f272b47ed201b6735c5ad7003f509d4c.png" class="math-inline" /> と <img src="tmp/a9d38f4f575bfc8ee1e29d999bee7f45.png" class="math-inline" />__

- <img src="tmp/d241c9d3cd4ded6d86a58af4676167f4.png" class="math-inline" />。包含関係はハッセ図で  
  ∅ ─ {1} ─ {1,2}  
  　　└ {2} ─  
  のような構造（{1} と {2} は比較不能）。
- <img src="tmp/eb761c0d3a02eec997b0119aecbad61f.png" class="math-inline" /> は全順序（鎖）。
- 順序同型なら、比較不能な元の有無などが一致する必要があるが、一方は全順序、他方はそうでないので同型ではない。

**答**：順序同型ではない。

__3. 実数の連続性（上限公理）__

__(1) <img src="tmp/cff6cffa62ac5d7a1478923e191429c1.png" class="math-inline" />__

- <img src="tmp/d3d75565250234d366cd2c3f48acbc81.png" class="math-inline" />、<img src="tmp/b2d22991c6d0733067067215e3bac979.png" class="math-inline" />。
- 最小元：0、最大元：1。

**答**：<img src="tmp/55f7da9ac2937c24476ce9c6e212ee30.png" class="math-inline" />、最小元 0、最大元 1。

__(2) <img src="tmp/61ecaab281656ad1743eabc4c95f9822.png" class="math-inline" />__

- <img src="tmp/d3d75565250234d366cd2c3f48acbc81.png" class="math-inline" />、<img src="tmp/b2d22991c6d0733067067215e3bac979.png" class="math-inline" />。
- 最小元：なし（0 は含まれない）、最大元：なし（1 は含まれない）。

**答**：<img src="tmp/55f7da9ac2937c24476ce9c6e212ee30.png" class="math-inline" />、最小元なし、最大元なし。

__(3) <img src="tmp/be66fe79aa53fbc268d52c9d51151d4c.png" class="math-inline" />__

- <img src="tmp/d3d75565250234d366cd2c3f48acbc81.png" class="math-inline" />（下界は 0 以下、最大下界は 0）。
- <img src="tmp/b2d22991c6d0733067067215e3bac979.png" class="math-inline" />（上界は 1 以上、最小上界は 1）。
- 最小元：なし（0 は含まれない）、最大元：1（<img src="tmp/96b1b9b6e0a53a05b99819bcb4339d8e.png" class="math-inline" />）。

**答**：<img src="tmp/55f7da9ac2937c24476ce9c6e212ee30.png" class="math-inline" />、最小元なし、最大元 1。

__(4) <img src="tmp/0df1aedc6854a8ecfd550aebaeb575bd.png" class="math-inline" />__

- <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> 上では <img src="tmp/ea3f9e446de972e40515b3c40e5f27f1.png" class="math-inline" />。
- 最小元・最大元は存在しない（<img src="tmp/f7ce50b661b41dc07e0bbc0ec3865d2b.png" class="math-inline" />）。

**答**：<img src="tmp/ea3f9e446de972e40515b3c40e5f27f1.png" class="math-inline" />、最小元なし、最大元なし。

__4. 有界単調数列の収束__

__(1) <img src="tmp/c3819ae5814d38c18687ba2eac37f026.png" class="math-inline" />__

- 単調増加：<img src="tmp/3b52178eeaeb3f78a3176d1e176f7585.png" class="math-inline" />。
- 上に有界：すべての <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> で <img src="tmp/04c4590feee26737947bde8ee8221338.png" class="math-inline" />。
- 有界単調数列の収束定理より収束。極限は <img src="tmp/e22545fc516c8be231bf8cbb8c9356d6.png" class="math-inline" />。

**答**：収束、極限 1。

__(2) <img src="tmp/82bf0e688ab80743555f0d93757938f5.png" class="math-inline" />（調和級数）__

- 単調増加：明らか。
- 上に有界か？ 調和級数は発散するので有界ではない。
- 有界単調数列の収束定理は適用できない。実際、発散する。

**答**：発散（有界でないため定理は適用不可）。

__(3) <img src="tmp/7419e43ef2ea5b5fa6d4928b4c9fcdb6.png" class="math-inline" />__

- 単調増加：明らか。
- 上に有界：<img src="tmp/fe6dc1b5b01eb7ef7d20539fe8844a8c.png" class="math-inline" /> より有界。
- 有界単調数列の収束定理より収束。極限は <img src="tmp/83e18979b3abea680d64fc6e530bb32c.png" class="math-inline" />。

**答**：収束、極限 <img src="tmp/83e18979b3abea680d64fc6e530bb32c.png" class="math-inline" />。

__(4) <img src="tmp/005165defcf15d1b8e5df5a380fb7ea4.png" class="math-inline" />__

- 単調でない（振動）。
- 有界単調数列の収束定理は適用できない。実際、振動して収束しない。

**答**：収束しない。

__5. Cauchy列と完備性__

__(1) <img src="tmp/ee4d8c572decf71a85f60479e87b523f.png" class="math-inline" />__

- <img src="tmp/4c525026b8d79f0c6d5fff6c1ca83399.png" class="math-inline" />（<img src="tmp/b46273f599b4c1b897c21771d9d22831.png" class="math-inline" />）より Cauchy列。
- 実数の完備性より収束。極限は 0。

**答**：Cauchy列、収束先 0。

__(2) <img src="tmp/165253952aa313093a186404ce9f9920.png" class="math-inline" />__

- 等比級数の部分和：<img src="tmp/6826fcc138eaa574d483265ae0c4d2c2.png" class="math-inline" />。
- <img src="tmp/847175327580dde8e27b32f88d745f27.png" class="math-inline" /> より Cauchy列。
- 収束先は 2。

**答**：Cauchy列、収束先 2。

__(3) <img src="tmp/82bf0e688ab80743555f0d93757938f5.png" class="math-inline" />（調和級数）__

- 調和級数は発散するので Cauchy列ではない（十分先でも項差が小さくならない）。
- 例：<img src="tmp/f2224d4ba22335e1fc3fdaffa931cba9.png" class="math-inline" /> のとき <img src="tmp/794fd3724925c5172146c59c46022251.png" class="math-inline" /> で下から押さえられる。

**答**：Cauchy列ではない。

__(4) <img src="tmp/005165defcf15d1b8e5df5a380fb7ea4.png" class="math-inline" />__

- <img src="tmp/9cf1eb1f4cd6c5857fc99b76e78e425c.png" class="math-inline" /> などより Cauchy列ではない。

**答**：Cauchy列ではない。

__6. 擬順序と商構成__

<img src="tmp/65480d9b9b29b6185e8b058d314761fb.png" class="math-inline" />、<img src="tmp/e3956976788b1d7f47a6b4575062cf73.png" class="math-inline" />。

__(1) 擬順序であること__

- **反射律**：<img src="tmp/af920336193c8dd1ff543701e2af75f9.png" class="math-inline" /> は定数 <img src="tmp/8f6881ddb797cc62dce1687294f25f38.png" class="math-inline" /> で成立。→ 成立。
- **推移律**：<img src="tmp/43b8449fce49c05662dd9d60fb136d68.png" class="math-inline" /> かつ <img src="tmp/49856b804ead4b1ac90be28bd5cd215f.png" class="math-inline" /> なら <img src="tmp/61be5db6521a1e3963e705416da7dfb5.png" class="math-inline" />（定数の積を取ればよい）。→ 成立。

よって <img src="tmp/97c52d28d71c6a46b6e846733d1d408f.png" class="math-inline" /> は擬順序集合。

__(2) 反対称律が成り立たない例__

- <img src="tmp/a9e773e187e03d87020b1f7f5264030f.png" class="math-inline" />、<img src="tmp/1ef737bb4f9d93a1f9d72ba57cab6c3a.png" class="math-inline" /> とすると、
  - <img src="tmp/43b8449fce49c05662dd9d60fb136d68.png" class="math-inline" />（定数 1 で <img src="tmp/0ff648b8502dec3158022f5bf90ae79c.png" class="math-inline" />）、
  - <img src="tmp/55f466c8bbb314d6e516c0e33081788f.png" class="math-inline" />（定数 2 で <img src="tmp/26638ef03faad89d1d8c046225000ca0.png" class="math-inline" /> for large <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" />）。
- よって <img src="tmp/c603fbf51ec7c19cf9812a4afa1cb049.png" class="math-inline" /> かつ <img src="tmp/bebc44429e1593bdbe7ca5ad14f5fa1a.png" class="math-inline" /> だが <img src="tmp/153aa91dc69a64186de4aacab98432dc.png" class="math-inline" />。  
  反対称律は成り立たない。

__(3) 同値関係 <img src="tmp/b2bf47bfccb1f6e490fa372542a41ac2.png" class="math-inline" /> の意味__

<img src="tmp/53f35bc07bdfaa99d44f961843f2e9ae.png" class="math-inline" /> は、  
<img src="tmp/5700f8197714bf7116045db215d82b15.png" class="math-inline" /> かつ <img src="tmp/b28bf3fa2574b2c82e016c302e62a4ae.png" class="math-inline" />、すなわち <img src="tmp/bd89abf7147e164cbe9a361fc799c0c2.png" class="math-inline" />（同程度のオーダー）。

**答**：<img src="tmp/55d27bddd6ad62a61e56b34d5ed54ccd.png" class="math-inline" />。

__(4) 商集合上の部分順序__

- 商集合 <img src="tmp/7957ba082625c33993d7806cf8a180aa.png" class="math-inline" /> 上に <img src="tmp/fe069b7cd84f9650dabae8c044e15898.png" class="math-inline" /> と定義。
- 反射律・推移律は <img src="tmp/b2ebf7379839345afa8b29aeff5dd3cd.png" class="math-inline" /> から継承される。
- 反対称律：<img src="tmp/05b26cbb6db84324b668cde0e38bf698.png" class="math-inline" /> かつ <img src="tmp/de6091ff397afd41f7af5a68139f11e2.png" class="math-inline" /> なら、定義より <img src="tmp/c603fbf51ec7c19cf9812a4afa1cb049.png" class="math-inline" /> かつ <img src="tmp/bebc44429e1593bdbe7ca5ad14f5fa1a.png" class="math-inline" />、すなわち <img src="tmp/b37fed6113ee92ab0b392f7c3636fb88.png" class="math-inline" />、よって <img src="tmp/1ac9ebf0880524fa511313afeefa1900.png" class="math-inline" />。
- したがって <img src="tmp/ab4f299945a8982d579b4fabc1660913.png" class="math-inline" /> は部分順序集合。

__7. 総合問題（順序集合と実数の連続性）__

__(1) <img src="tmp/085c3e1411ceeb09763df76ed9f7e4c7.png" class="math-inline" /> の <img src="tmp/982b8aab2ac990f8187830ab74eb0a06.png" class="math-inline" />__

- <img src="tmp/ef053640580d7097e73fb4aa3b8aff3c.png" class="math-inline" />。
- 上界の最小値は <img src="tmp/100bf8adf49a114a6587e5bad1386df3.png" class="math-inline" />、下界の最大値は <img src="tmp/c3ee14e4ed0146b08112780a88444e17.png" class="math-inline" />。
- よって <img src="tmp/ade24e4c4ba8389885f15948de3f2d12.png" class="math-inline" />。

**答**：<img src="tmp/ade24e4c4ba8389885f15948de3f2d12.png" class="math-inline" />。

__(2) <img src="tmp/b44755cfa1b3f0f2e9af4bc0a10f42d4.png" class="math-inline" /> での <img src="tmp/4bea7101cbb8efae50cb963ee6bd224a.png" class="math-inline" /> の上限__

- <img src="tmp/fca89afa24215d2a56d4d09d1c21861e.png" class="math-inline" /> 。
- 有理数の中には <img src="tmp/051fdc7999813ce58e8c851d2b7d9a30.png" class="math-inline" /> を満たす <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は存在しない。
- 任意の有理数 <img src="tmp/eefdb98a88822f2df85e050a893c5b6f.png" class="math-inline" /> に対して、<img src="tmp/0c4191cbc0cfe6f4885da3e4eaa84294.png" class="math-inline" /> となる有理数 <img src="tmp/add5bb4fc3bfe26587c101125d12ae33.png" class="math-inline" /> が存在する（有理数の稠密性）。
- よって、<img src="tmp/100bf8adf49a114a6587e5bad1386df3.png" class="math-inline" /> より小さい任意の有理数は上界ではなく、<img src="tmp/100bf8adf49a114a6587e5bad1386df3.png" class="math-inline" /> より大きい有理数は上界だが、それらの中で最小のものは存在しない（<img src="tmp/100bf8adf49a114a6587e5bad1386df3.png" class="math-inline" /> にいくらでも近い有理数が取れる）。
- したがって、<img src="tmp/4bea7101cbb8efae50cb963ee6bd224a.png" class="math-inline" /> の上限は <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> 内には存在しない。

__(3) 有理数の「連続でない（完備でない）」ことの理解__

- 有理数 <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> は稠密だが、Cauchy列が有理数内に極限を持たないことがある（例：<img src="tmp/100bf8adf49a114a6587e5bad1386df3.png" class="math-inline" /> に収束する有理数列）。
- 上に有界な集合が必ずしも上限を持たない（例：<img src="tmp/a7a4974f52c90b854da3d692e022df81.png" class="math-inline" />）。
- このように、有理数直線には「隙間」があり、数直線として連続ではない。

__(4) 実数の連続性の重要性（解析学）__

- 実数の連続性（完備性）により、有界単調数列やCauchy列が必ず収束し、中間値定理・最大値定理・一様連続性など解析学の基本定理が成り立つ。
- これがなければ、極限操作や連続関数の性質を厳密に扱うことができず、微分・積分の理論も成立しない。

<div style="page-break-before:always"></div>




# 整列集合

## 整列順序

**整列集合（well-ordered set）** は、全順序集合のうち、**任意の空でない部分集合が最小元を持つ**という強い条件を満たすものです。  
順序数（ordinal numbers）の理論や超限帰納法の基礎となる重要な概念です。

### 1. 整列集合の定義

全順序集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> が**整列集合**であるとは、次の条件を満たすことをいいます。

> 任意の空でない部分集合 <img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> は、**最小元**を持つ。

すなわち、<img src="tmp/70ae9ae581001c1f1f062998fd9a6d2a.png" class="math-inline" /> ならば、ある <img src="tmp/9507e0235f1b54262f27fa314e741f93.png" class="math-inline" /> が存在して、すべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> について <img src="tmp/24b42b7ecec14bb0186ff6f006cb1b20.png" class="math-inline" /> が成り立つ。

このとき、<img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> を**整列順序（well-order）** と呼びます。

### 2. 定義の注意点

__(1) 全順序であること__

- 整列集合は**全順序集合**の一種です。
- 部分順序集合では「比較不能な元」が存在し得ますが、整列集合では任意の2元が比較可能です。
- したがって、最小元の一意性は自動的に保証されます（全順序性＋反対称律より）。

__(2) 「任意の空でない部分集合」が最小元を持つ__

- 有限部分集合だけでなく、**無限部分集合**も含めて、すべての空でない部分集合が最小元を持たなければなりません。
- この条件は、**降鎖条件（descending chain condition）** とも関係し、「無限に小さくなり続ける列」が存在しないことを意味します。

### 3. 整列集合の具体例

__(1) 有限の全順序集合__

- 例： <img src="tmp/451f07222d20ef0263efce4329b2c6b9.png" class="math-inline" /> 、<img src="tmp/692f8d78263f169cb7940e8ff5f12c7e.png" class="math-inline" />（辞書順）など。
- 有限集合では、任意の空でない部分集合は有限個の元からなるので、その中で最小の元が必ず存在します。
- したがって、**有限の全順序集合はすべて整列集合**です。

__(2) 自然数全体 <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" />__

- 任意の空でない自然数の集合には、最小の自然数が存在します（自然数の整序性）。
- 例：偶数全体 <img src="tmp/2a9a74be1b88a507bca22ff741b1ee93.png" class="math-inline" /> の最小元は 2。
- よって <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" /> は整列集合です。

__(3) 順序数（ordinal numbers）__

- 順序数そのものが整列集合として定義されます。
- 例： <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> （自然数型）、 <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> 、 <img src="tmp/838230f39aa4893752cb4d57f76fdc99.png" class="math-inline" /> などはすべて整列集合です。

### 4. 整列集合でない例

__(1) 整数全体 <img src="tmp/5608f64ad5726b0cec3f4b4905f05ef3.png" class="math-inline" />__

- 部分集合 <img src="tmp/2a353911e1087f5856c81c5753987ae2.png" class="math-inline" />（負の整数全体）を考えると、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> には最小元が存在しません（いくらでも小さい負の整数がある）。
- よって <img src="tmp/5608f64ad5726b0cec3f4b4905f05ef3.png" class="math-inline" /> は整列集合ではありません。

__(2) 有理数全体 <img src="tmp/b44755cfa1b3f0f2e9af4bc0a10f42d4.png" class="math-inline" />__

- 部分集合 <img src="tmp/dfa7cddc3db65834ce74fbca2b02998c.png" class="math-inline" /> を考えると、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> には最小元が存在しません（任意の正の有理数より小さい正の有理数が存在する）。
- よって <img src="tmp/b44755cfa1b3f0f2e9af4bc0a10f42d4.png" class="math-inline" /> は整列集合ではありません。

__(3) 実数全体 <img src="tmp/028e8e6759d744985e5119ebfce2af99.png" class="math-inline" />__

- 同様に、正の実数全体には最小元が存在しないので、整列集合ではありません。
- 実は、**選択公理**を用いると「任意の集合に整列順序が存在する（整列可能定理）」ことが示されますが、その整列順序は通常の大小関係とは異なり、具体的に構成できるとは限りません。

### 5. 整列集合の基本的性質

__(1) 整列集合の部分集合__

- 整列集合の任意の部分集合は、制限された順序に関して再び整列集合になります。
- 特に、**整列集合の始片（initial segment）** は整列集合です。

__(2) 整列集合の順序同型類：順序数__

- 2つの整列集合が順序同型であるとき、それらは同じ**順序型**を持ちます。
- 整列集合の順序同型類を**順序数（ordinal number）** と呼びます。
- 例：<img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> の順序型は <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />、<img src="tmp/8b997ccc696c3df57e4702a76cebd9fe.png" class="math-inline" /> の順序型は自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> です。

__(3) 整列集合の比較定理__

- 任意の2つの整列集合は、一方が他方の始片と順序同型であるか、または互いに順序同型である、という強い性質を持ちます。
- これにより、順序数には自然な全順序 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> が入り、「順序数の大小」が定義されます。


### 6. なぜ整列集合が必要なのか

__(1) 順序数理論の基礎__

- 順序数は「整列集合の順序型」として定義され、無限の大きさを測る基本的な道具です。
- 集合論の多くの結果（基数の比較、累積階層など）は順序数に依存しています。

__(2) 超限帰納法・超限再帰__

- 無限の構造に対しても帰納法や再帰的定義を適用できるようにします。
- 例：集合論の定理の証明、再帰的関数の定義、木構造の解析など。

__(3) 整列可能定理と選択公理__

- **整列可能定理**（任意の集合に整列順序が存在する）は、**選択公理**と同値です。
- これは「任意の集合を一直線に並べられる」ことを主張し、集合論の基礎的な結果の一つです。

## 直後要素、直前要素、極限要素

**直後要素（immediate successor）**・**直前要素（immediate predecessor）**・**極限要素（limit element）**は、主に**整列集合**や**順序数**の文脈で使われる概念です。  
これらは、順序集合の中で「すぐ次に来る元」「すぐ前に来る元」「それより小さい元たちの上限として現れる元」を表します。

以下、順序集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> に対して定義します。

### 1. 直後要素（immediate successor）

__定義__

元 <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して、**直後要素**とは、次の条件を満たす元 <img src="tmp/f103663d577f4d97a49cb2fa77445dd3.png" class="math-inline" /> のことです。

1. <img src="tmp/24c9e6d4b22616cc4d909e1612426371.png" class="math-inline" />（すなわち <img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> かつ <img src="tmp/7e51fc5af1798e32ebc66e288b627fbd.png" class="math-inline" />）。
2. <img src="tmp/c9107b3e2a9bea8e2b93228262c65e0d.png" class="math-inline" /> となる <img src="tmp/75f068f1933f3a5ea0b10ab1385585e9.png" class="math-inline" /> が存在しない。

言い換えると、<img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> より大きく、かつ <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> と <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> の間に他の元が入らない「すぐ次の元」です。

- 記号：<img src="tmp/44b3d100c9920afd802fae611807f97c.png" class="math-inline" /> などと書くことがあります。
- 存在すれば**一意**です（全順序性などから）。

__具体例__

- <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" />：  
  - 任意の自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> に対して、直後要素は <img src="tmp/e4bc429c866bbde4b0f321d9b5139986.png" class="math-inline" />。
- <img src="tmp/5608f64ad5726b0cec3f4b4905f05ef3.png" class="math-inline" />：  
  - 任意の整数 <img src="tmp/1ebd8eed2a684f4edcbc6db4e42bfc27.png" class="math-inline" /> に対して、直後要素は <img src="tmp/f9981871022ff21dcd76f298ba96a380.png" class="math-inline" />。
- 有限全順序集合：  
  - 最大元以外の各元に直後要素が存在する。
- <img src="tmp/b44755cfa1b3f0f2e9af4bc0a10f42d4.png" class="math-inline" /> や <img src="tmp/028e8e6759d744985e5119ebfce2af99.png" class="math-inline" />：  
  - 稠密なので、任意の元 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に対して <img src="tmp/24c9e6d4b22616cc4d909e1612426371.png" class="math-inline" /> となる <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は存在するが、<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> と <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> の間に必ず他の有理数・実数が存在する。  
  - よって、**直後要素は存在しない**。

### 2. 直前要素（immediate predecessor）

__定義__

元 <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して、**直前要素**とは、次の条件を満たす元 <img src="tmp/f103663d577f4d97a49cb2fa77445dd3.png" class="math-inline" /> のことです。

1. <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" />（すなわち <img src="tmp/708596235cba73cab553a9bb5933227c.png" class="math-inline" /> かつ <img src="tmp/f0a38ab7a1955eba804fc1cb5a51f1d0.png" class="math-inline" />）。
2. <img src="tmp/ec6c00ca72b4c5f90faa00bf6dc14c31.png" class="math-inline" /> となる <img src="tmp/75f068f1933f3a5ea0b10ab1385585e9.png" class="math-inline" /> が存在しない。

言い換えると、<img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> より小さく、かつ <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> と <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> の間に他の元が入らない「すぐ前の元」です。

- 記号：<img src="tmp/ddb94ae1e37791c90f5f45680ee7c9ca.png" class="math-inline" /> などと書くことがあります。
- 存在すれば**一意**です。

__具体例__

- <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" />：  
  - 1 には直前要素がない（最小元）。  
  - <img src="tmp/64575accb913da27dd32c9c9dd11bffe.png" class="math-inline" /> に対して、直前要素は <img src="tmp/48265144337ba4f6739874dad37288c9.png" class="math-inline" />。
- <img src="tmp/5608f64ad5726b0cec3f4b4905f05ef3.png" class="math-inline" />：  
  - 任意の整数 <img src="tmp/1ebd8eed2a684f4edcbc6db4e42bfc27.png" class="math-inline" /> に対して、直前要素は <img src="tmp/23ed9a93b7485cf75edf66fb7138f90d.png" class="math-inline" />。
- 有限全順序集合：  
  - 最小元以外の各元に直前要素が存在する。
- <img src="tmp/b44755cfa1b3f0f2e9af4bc0a10f42d4.png" class="math-inline" /> や <img src="tmp/028e8e6759d744985e5119ebfce2af99.png" class="math-inline" />：  
  - 稠密なので、**直前要素は存在しない**。

### 3. 極限要素（limit element）

**極限要素**は、主に**順序数**や**整列集合**の文脈で定義されます。  
直感的には、「それより小さい元たちの『極限』として現れる元」です。

__定義（順序数の文脈）__

順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が**極限順序数（limit ordinal）** であるとは、次の条件を満たすことをいいます。

- 直前要素が存在しない。すなわち、<img src="tmp/c38dd19a963bda52ec93437cb289d67f.png" class="math-inline" /> となる順序数 <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> が存在しない。

ここで、<img src="tmp/da86bd99ca548de0eafcc544b293ebb8.png" class="math-inline" /> は <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の直後順序数（<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> に新しい最大元を付け加えた順序数）です。

より一般的に、整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> において、元 <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> が**極限要素**であるとは：

- <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は最小元ではない。
- <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に直前要素が存在しない。

すなわち、<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は「ある元の直後」としてではなく、それより小さい元たちの**上限**として現れる元です。

__具体例（順序数）__

- <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />（自然数全体の順序型）：  
  - 任意の自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> に対して <img src="tmp/6263c987771690e784009fcdb7d0955c.png" class="math-inline" /> だが、<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の直前要素は存在しない（自然数は有限で、<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は無限）。  
  - よって <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は極限順序数。
- <img src="tmp/9c6986747a463cb74f3d4623e5b6274d.png" class="math-inline" />：  
  - 直前要素は <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> なので、極限順序数ではない（**後続順序数**）。
- <img src="tmp/c6164b27eaf5edfa8ca0bad575cc6a44.png" class="math-inline" />：  
  - 直前要素は存在しないので極限順序数。
- 0 や 1,2,3,... は極限順序数ではない（0 は最小、1=0+1, 2=1+1, ... はすべて後続順序数）。

__具体例（他の整列集合）__

- <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" />：  
  - 最小元 1 以外はすべて直前要素を持つ（<img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> の直前は <img src="tmp/48265144337ba4f6739874dad37288c9.png" class="math-inline" />）。  
  - よって極限要素は存在しない。
- 順序数 <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> に対応する整列集合（例：<img src="tmp/2e5d5611945830cecdaca65e8ed2d466.png" class="math-inline" />）では、  
  - 任意の有限順序数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> には直前要素があるが、<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> 自体には直前要素がない → <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> が極限要素。

### 4. 後続順序数（successor ordinal）との対比

- **後続順序数**：直前要素を持つ順序数。すなわち <img src="tmp/c38dd19a963bda52ec93437cb289d67f.png" class="math-inline" /> と書けるもの。
- **極限順序数**：直前要素を持たない 0 でない順序数。

任意の 0 でない順序数は、**後続順序数**か**極限順序数**のどちらかです。

### 5. なぜこれらの概念が必要か

__(1) 順序数の分類__

- 順序数を「0」「後続順序数」「極限順序数」の3種類に分けることで、超限帰納法や超限再帰の定義がきれいに書けます。
- 例：超限帰納法の段階で、
  - 基底：0 に対する証明。
  - 後続段階：<img src="tmp/2122dda6d968a1b89897921d2bac8d47.png" class="math-inline" /> のとき、<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> での成立から <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> での成立を示す。
  - 極限段階：<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が極限順序数のとき、すべての <img src="tmp/b4d58078d6159ee352ef89f89ce6e36c.png" class="math-inline" /> での成立から <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> での成立を示す。

__(2) 無限の構造の理解__

- 極限順序数は、「有限ステップでは到達できないが、無限のプロセスの極限として現れる」ような元を表します。
- 例：<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は有限回の「+1」操作では到達できないが、すべての自然数の極限として現れる。

__(3) 集合論・モデル理論での応用__

- 巨大な順序数（到達不能順序数など）は、極限順序数として定義され、集合論のモデルや巨大基数の理論で重要な役割を果たします。
- 直後・直前・極限の概念は、順序構造の「局所的な形」を分析するのに役立ちます。


## 超限帰納法

**超限帰納法（transfinite induction）**は、数学的帰納法を**整列集合**や**順序数**上に拡張した証明法です。  
「すべての順序数（または整列集合の元）について命題が成り立つ」ことを示すために用いられます。

以下、順序数 <img src="tmp/24df8a72caec0be7051202ec37ad58da.png" class="math-inline" /> 上の超限帰納法を中心に説明しますが、一般の整列集合でも同様に定義できます。

### 1. 超限帰納法の定義（順序数版）

順序数全体のクラス <img src="tmp/24df8a72caec0be7051202ec37ad58da.png" class="math-inline" /> に対して、命題 <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" />（<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は順序数）が与えられているとします。

超限帰納法とは、次の推論規則です。

> 次の2条件が成り立つならば、**すべての順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> が成り立つ**。
>
> 1. **基底（base case）**：  
>    <img src="tmp/b27dacc5676ded5b21aa21ada36bd16a.png" class="math-inline" /> が成り立つ。
> 2. **帰納段階（inductive step）**：  
>    任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について、  
>    「すべての <img src="tmp/b4d58078d6159ee352ef89f89ce6e36c.png" class="math-inline" /> に対して <img src="tmp/3be9b85cf782a26b88a0cfd353bdaf69.png" class="math-inline" /> が成り立つならば、<img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> も成り立つ」  
>    ことを示す。

ここで、<img src="tmp/b4d58078d6159ee352ef89f89ce6e36c.png" class="math-inline" /> は順序数の大小関係（整列順序）です。

__直感的な意味__

- 通常の数学的帰納法は「<img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> から <img src="tmp/1d9bcbf0566d26777efa585c50b021aa.png" class="math-inline" /> を示す」ですが、超限帰納法では「**<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> より小さいすべての順序数で成り立てば、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> でも成り立つ**」ことを要求します。
- これにより、有限のステップでは到達できない極限順序数（例：<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />）についても命題を示すことができます。

### 2. 3段階に分けた書き方（後続・極限の場合分け）

順序数は次の3種類に分けられます。

- **0**：最小の順序数。
- **後続順序数（successor ordinal）**：<img src="tmp/c38dd19a963bda52ec93437cb289d67f.png" class="math-inline" /> と書けるもの。
- **極限順序数（limit ordinal）**：0 ではなく、かつ後続順序数でないもの（直前要素を持たない）。

これに対応して、超限帰納法は次の3段階で記述されることが多いです。

> すべての順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> が成り立つことを示すには、次を示せばよい。
>
> 1. **基底**：<img src="tmp/b27dacc5676ded5b21aa21ada36bd16a.png" class="math-inline" /> を示す。
> 2. **後続段階**：任意の順序数 <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> について、<img src="tmp/3be9b85cf782a26b88a0cfd353bdaf69.png" class="math-inline" /> が成り立つならば <img src="tmp/d3956bde88781497b0ed0243360459f1.png" class="math-inline" /> も成り立つことを示す。
> 3. **極限段階**：任意の極限順序数 <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> について、  
>    「すべての <img src="tmp/6d2307b6847968dcfb26600c94b7b34e.png" class="math-inline" /> に対して <img src="tmp/3be9b85cf782a26b88a0cfd353bdaf69.png" class="math-inline" /> が成り立つならば <img src="tmp/d2d07d491ce308b6749dfc422f75d6b4.png" class="math-inline" /> も成り立つ」  
>    ことを示す。

__なぜこの3段階で十分か__

- 任意の 0 でない順序数は、後続順序数か極限順序数のどちらかです。
- 後続段階で「<img src="tmp/e93dfe8094b4e0089eaaddc2a9c98148.png" class="math-inline" />」の推移が保証され、極限段階で「より小さい順序数全体から極限順序数へのジャンプ」が保証されます。
- 基底から始めて、後続段階と極限段階を繰り返すことで、すべての順序数にわたって <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> が成立することが示されます（整列性による）。

### 3. 一般の整列集合上の超限帰納法

整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> に対して、命題 <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" />（<img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" />）が与えられているとします。

超限帰納法は次のように定式化されます。

> 次の条件が成り立つならば、**すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立つ**。
>
> 任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、  
> 「すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> に対して <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つならば、<img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> も成り立つ」  
> ことを示せ。

ここで <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> は <img src="tmp/708596235cba73cab553a9bb5933227c.png" class="math-inline" /> かつ <img src="tmp/f0a38ab7a1955eba804fc1cb5a51f1d0.png" class="math-inline" /> を意味します。

__基底の扱い__

- 最小元 <img src="tmp/53d768e41950280cba385ae6d8d9433f.png" class="math-inline" /> が存在する場合、<img src="tmp/4748766bdee3e8c3a5e169791468a7aa.png" class="math-inline" /> となる <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は存在しないので、  
  「すべての <img src="tmp/4748766bdee3e8c3a5e169791468a7aa.png" class="math-inline" /> に対して <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" />」は vacuously true（空虚な真）です。
- したがって、上記の条件から自動的に <img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> が導かれます。  
  明示的に基底 <img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> を示す必要はありませんが、実際の証明では通常 <img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> を別途確認します。

### 4. 超限帰納法の具体例

__例1：順序数の加法の結合則の証明（概略）__

順序数の加法 <img src="tmp/447327d2486e4f275e7fa3e3e0720a81.png" class="math-inline" /> が定義されているとき、  
<img src="tmp/b2b54399bc0d51ee0b1d0b7c0ca463d0.png" class="math-inline" /> をすべての <img src="tmp/e8604fb1e70a912aa41b59393ecde955.png" class="math-inline" /> について示したい。

- 固定した <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> に対して、<img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> に関する超限帰納法を用いる。
- 基底：<img src="tmp/476c4f00ad6213162e9e2650d47cea18.png" class="math-inline" /> のとき、左辺＝<img src="tmp/523ee0a625bb9be4d02ba6cf448d12be.png" class="math-inline" />、右辺＝<img src="tmp/523ee0a625bb9be4d02ba6cf448d12be.png" class="math-inline" /> で成立。
- 後続段階：<img src="tmp/48b6a914a3820d05c5d86d2ce3954d33.png" class="math-inline" /> のとき、帰納仮定（<img src="tmp/865903777a19d290dd9395c606980456.png" class="math-inline" /> での成立）から <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> での成立を示す。
- 極限段階：<img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> が極限順序数のとき、<img src="tmp/94cd205e3840db4bc5fb43b2c082cee9.png" class="math-inline" /> を用いて、両辺の極限として成立を示す。

（詳細は順序数の定義に依存しますが、枠組みとして超限帰納法が使われます。）

---

順序数の加法の結合則



<div class="math-display-container"><img src="tmp/3fea9038888b9842405f35617b2ae045.png" class="math-display" /></div>



を、<img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> に関する超限帰納法で証明します。<img src="tmp/83bbc6424569318c5ebd6bf2f6d1700e.png" class="math-inline" /> は任意に固定された順序数とします。

__1. 順序数の加法の定義（復習）__

順序数の加法は、次のように帰納的に定義されます。

- 任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について  
  

<div class="math-display-container"><img src="tmp/704cc7f10b8fe513d1b782f796c6c524.png" class="math-display" /></div>


- 後続順序数 <img src="tmp/06ad901881b25f85541cc8e388df0716.png" class="math-inline" /> のとき  
  

<div class="math-display-container"><img src="tmp/09ba670c4f6de178f1394828acdd263c.png" class="math-display" /></div>


- 極限順序数 <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" />（0 でも後続でもない）のとき  
  

<div class="math-display-container"><img src="tmp/eb87624a46a497e9a5e4dc684532e0c7.png" class="math-display" /></div>



この定義に基づいて、結合則を証明します。

__2. 超限帰納法の設定__

<img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> に関する命題 <img src="tmp/7faaa48facfb45c7c577a0ca2ec109fe.png" class="math-inline" /> を



<div class="math-display-container"><img src="tmp/b2fe6401deb6997028590b4e1a8ebb01.png" class="math-display" /></div>



とおきます。  
超限帰納法により、すべての <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> について <img src="tmp/7faaa48facfb45c7c577a0ca2ec109fe.png" class="math-inline" /> が成り立つことを示します。

__3. 基底：<img src="tmp/476c4f00ad6213162e9e2650d47cea18.png" class="math-inline" /> のとき__

<img src="tmp/476c4f00ad6213162e9e2650d47cea18.png" class="math-inline" /> のとき、

- 左辺：
  

<div class="math-display-container"><img src="tmp/df8addcc508142f64d04eaa24a2d7763.png" class="math-display" /></div>


- 右辺：
  

<div class="math-display-container"><img src="tmp/1be7a10d14fc3645537ac89a1ac261ee.png" class="math-display" /></div>



よって、<img src="tmp/2e5f188855f79c6508e6a0215471907f.png" class="math-inline" /> が成り立ち、<img src="tmp/b27dacc5676ded5b21aa21ada36bd16a.png" class="math-inline" /> は真です。

__4. 後続段階：<img src="tmp/06ad901881b25f85541cc8e388df0716.png" class="math-inline" /> のとき__

帰納仮定として、ある <img src="tmp/865903777a19d290dd9395c606980456.png" class="math-inline" /> について <img src="tmp/d84995df143b2d40f72ca1a580767dda.png" class="math-inline" />、すなわち



<div class="math-display-container"><img src="tmp/cceb1ca31c229210a84c401af91d479e.png" class="math-display" /></div>



が成り立つと仮定します。このとき <img src="tmp/06ad901881b25f85541cc8e388df0716.png" class="math-inline" /> について示します。

__左辺の計算__

加法の定義より、



<div class="math-display-container"><img src="tmp/57e87a61779221e7ac77165c69289dfc.png" class="math-display" /></div>



帰納仮定を用いると、



<div class="math-display-container"><img src="tmp/17a67f69e91e10a150b48a7f3738a948.png" class="math-display" /></div>



__右辺の計算__

加法の定義より、



<div class="math-display-container"><img src="tmp/f927c9576a32c21a7823f9021db5f4b9.png" class="math-display" /></div>



__比較__

左辺と右辺はともに



<div class="math-display-container"><img src="tmp/18b04dac1d10eb90f613b93d09f5b7ea.png" class="math-display" /></div>



に等しいので、



<div class="math-display-container"><img src="tmp/595226a97eefaf8a713a757e361e48d5.png" class="math-display" /></div>



が成り立ちます。よって、<img src="tmp/d84995df143b2d40f72ca1a580767dda.png" class="math-inline" /> から <img src="tmp/283105c5ea7002c881241dd3811d5aa7.png" class="math-inline" /> が導かれます。

__5. 極限段階：<img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> が極限順序数のとき__

<img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> が 0 でも後続でもない極限順序数とします。このとき、加法の定義より



<div class="math-display-container"><img src="tmp/b5fe4dff005477121cad5d72f74af01c.png" class="math-display" /></div>



が成り立ちます（<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は任意の順序数）。

帰納仮定として、すべての <img src="tmp/c1924ed181f80bf1ed9c99d3d8f08ec8.png" class="math-inline" /> について <img src="tmp/d84995df143b2d40f72ca1a580767dda.png" class="math-inline" />、すなわち



<div class="math-display-container"><img src="tmp/cceb1ca31c229210a84c401af91d479e.png" class="math-display" /></div>



が成り立つと仮定します。

__左辺の計算__

<img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> は極限順序数なので、<img src="tmp/516bcee3b88ec9ab4b0d33f3e9f43dc6.png" class="math-inline" /> も極限順序数であり、



<div class="math-display-container"><img src="tmp/1d0fc4b6b0d5b40c08549a72a4a9fb38.png" class="math-display" /></div>



が成り立ちます。

帰納仮定より、すべての <img src="tmp/c1924ed181f80bf1ed9c99d3d8f08ec8.png" class="math-inline" /> について



<div class="math-display-container"><img src="tmp/cceb1ca31c229210a84c401af91d479e.png" class="math-display" /></div>



なので、



<div class="math-display-container"><img src="tmp/fc6514633734b062b50af666d9ce338d.png" class="math-display" /></div>



__右辺の計算__

<img src="tmp/c60129dcb96ec581cb94749352ed9a5c.png" class="math-inline" /> は極限順序数であり（<img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> が極限なので）、



<div class="math-display-container"><img src="tmp/ed7df1f1a53fb81e9334529bcabd94d0.png" class="math-display" /></div>



したがって、



<div class="math-display-container"><img src="tmp/1efaad16b2a2ea61874e306aeedfb7c0.png" class="math-display" /></div>



（最後の等式は、順序数の加法が極限に対して連続であることから成り立ちます。）

__比較__

左辺と右辺はともに



<div class="math-display-container"><img src="tmp/6fcbccc2e59052100c385af4dacb0bda.png" class="math-display" /></div>



に等しいので、



<div class="math-display-container"><img src="tmp/3fea9038888b9842405f35617b2ae045.png" class="math-display" /></div>



が成り立ちます。よって、極限順序数 <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> についても <img src="tmp/7faaa48facfb45c7c577a0ca2ec109fe.png" class="math-inline" /> が真です。

__6. 結論__

基底・後続段階・極限段階のすべてについて、命題 <img src="tmp/7faaa48facfb45c7c577a0ca2ec109fe.png" class="math-inline" /> が成り立つことが示されました。  
したがって、超限帰納法により、すべての順序数 <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> について



<div class="math-display-container"><img src="tmp/3fea9038888b9842405f35617b2ae045.png" class="math-display" /></div>



が成り立ちます。<img src="tmp/83bbc6424569318c5ebd6bf2f6d1700e.png" class="math-inline" /> は任意でしたので、結合則がすべての <img src="tmp/41d1643f542f790cadf6b2814e3fbe1c.png" class="math-inline" /> について成立することが示されました。


---


__例2：整列集合の始片の性質__

整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の始片（initial segment）が常に整列集合であることを示すのに、超限帰納法が使えます。

- <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" />：「<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> より小さい元全体の集合 <img src="tmp/c6fb3b6219625e56c0026b715f98e4b5.png" class="math-inline" /> は整列集合である」とおく。
- 基底：最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> について、<img src="tmp/e93923bff783f10dc9839f4b4af02b5d.png" class="math-inline" /> は（自明に）整列集合。
- 帰納段階：任意の <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> について、すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> で <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つと仮定し、<img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> を示す。  
  （ここで <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> が後続か極限かで場合分けが必要になることが多い。）

---

整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の任意の始片が再び整列集合になることを、超限帰納法で示します。

__1. 設定と命題の定義__

<img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> を整列集合とします。  
任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対して、その**真の始片**を



<div class="math-display-container"><img src="tmp/2be52f6706b41f345446212e48bab949.png" class="math-display" /></div>



とおきます。  
この <img src="tmp/926298b41290516c216fa4d6f5753c5f.png" class="math-inline" /> に <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の順序を制限した部分順序集合 <img src="tmp/ce81f3eea9e0b2125b76973fe3b555c6.png" class="math-inline" /> が整列集合であることを示したいので、命題を



<div class="math-display-container"><img src="tmp/ec9f969dba78e492723c5f10158d1df1.png" class="math-display" /></div>



と定義します。  
ここで、<img src="tmp/926298b41290516c216fa4d6f5753c5f.png" class="math-inline" /> が空集合のときも、空集合は（自明に）整列集合とみなします。

__2. 基底：最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> のとき__

<img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は整列集合なので、最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> が存在します。  
このとき、



<div class="math-display-container"><img src="tmp/9570567dae8dab41e32f69a44a81c813.png" class="math-display" /></div>



です。空集合は、

- 空でない部分集合が存在しないので、整列性の条件（「任意の空でない部分集合が最小元を持つ」）が形式的に満たされる

と解釈できます。したがって、<img src="tmp/1441d231971913267cc858e9a918ebb2.png" class="math-inline" /> は整列集合です。  
よって、<img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> は真です。

__3. 帰納段階：任意の <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> について__

任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> をとり、**帰納仮定**として

> すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> について、<img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つ  
> すなわち、すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> について <img src="tmp/b3301674f2537b4ac503d2e39aa9c0e8.png" class="math-inline" /> は整列集合である

と仮定します。このもとで、<img src="tmp/ce81f3eea9e0b2125b76973fe3b555c6.png" class="math-inline" /> が整列集合であることを示します。

__3.1 順序の性質__

<img src="tmp/ce81f3eea9e0b2125b76973fe3b555c6.png" class="math-inline" /> は、<img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の部分順序集合です。  
整列集合の**任意の部分集合**は再び整列集合になるので、<img src="tmp/ce81f3eea9e0b2125b76973fe3b555c6.png" class="math-inline" /> も整列集合です。  
したがって、ここでは「整列集合の部分集合は整列集合」という一般的事実を使えば、帰納法を使わずにすぐ結論が出ます。

しかし、あえて帰納法の流れで説明すると、次のようになります。

__3.2 帰納法による議論__

<img src="tmp/ce81f3eea9e0b2125b76973fe3b555c6.png" class="math-inline" /> が整列集合であることを示すには、<img src="tmp/926298b41290516c216fa4d6f5753c5f.png" class="math-inline" /> の任意の空でない部分集合 <img src="tmp/bb56bbe5537fb9e1e1c372448079c53f.png" class="math-inline" /> が最小元を持つことを示せば十分です。

- <img src="tmp/bb56bbe5537fb9e1e1c372448079c53f.png" class="math-inline" /> かつ <img src="tmp/70ae9ae581001c1f1f062998fd9a6d2a.png" class="math-inline" /> とします。
- <img src="tmp/012336c08bfa69d2648496ec24abca99.png" class="math-inline" /> なので、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の空でない部分集合でもあります。
- <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は整列集合なので、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の中で最小元 <img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> を持ちます。

この <img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> が <img src="tmp/926298b41290516c216fa4d6f5753c5f.png" class="math-inline" /> における <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の最小元でもあることを確認します。

- <img src="tmp/3f2cc82d481829f48daecec09477501d.png" class="math-inline" /> なので、<img src="tmp/ddca869d3d9f1412fa6644a4515bd32b.png" class="math-inline" /> です。
- <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の任意の元 <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> について、<img src="tmp/54657b82b15b0f6f3704372681c447a6.png" class="math-inline" /> より <img src="tmp/950f6c7a0b26f4a2e61da0246e4e64ff.png" class="math-inline" /> です。
- また、<img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> における <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の最小元なので、<img src="tmp/f5645db615a2af23d8c0eb8fbfb5bdeb.png" class="math-inline" /> です。
- したがって、<img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> は <img src="tmp/926298b41290516c216fa4d6f5753c5f.png" class="math-inline" /> における <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の最小元でもあります。

よって、<img src="tmp/ce81f3eea9e0b2125b76973fe3b555c6.png" class="math-inline" /> は整列集合です。  
これで <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が示されました。

__4. 超限帰納法の適用__

<img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は整列集合なので、その順序は**整列順序**です。  
整列順序に関する超限帰納法は、次のように述べられます。

> 命題 <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が、すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について成り立つことを示すには、
> - 最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> について <img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> を示し、
> - 任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、「すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> で <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つならば <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> も成り立つ」ことを示せばよい。

この形の帰納法は、整列集合に対して常に有効です。

- 基底：最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> について <img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> が真であることは、すでに示しました。
- 帰納段階：任意の <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> について、「すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> で <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つならば <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> も成り立つ」ことも示しました。

したがって、超限帰納法により、すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立ちます。

__5. 結論__

以上より、整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の任意の真の始片



<div class="math-display-container"><img src="tmp/2be52f6706b41f345446212e48bab949.png" class="math-display" /></div>



は、再び整列集合になります。  
（空集合の場合も含めて、すべての始片が整列集合です。）

この結果は、整列集合の理論において基本的であり、順序数との対応（整列集合は順序数と同型）や、超限帰納法・超限再帰の正当化など、多くの場面で暗黙に利用されます。

---

### 5. 超限帰納法の正当性（なぜ成り立つか）

超限帰納法が正当である理由は、**整列集合の定義**（任意の空でない部分集合が最小元を持つ）にあります。

背理法で示すことができます。

- ある <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> で <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が偽になると仮定する。
- <img src="tmp/74a98993af0407d00d574844df3de5bf.png" class="math-inline" /> とおくと、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は空でない。
- 整列性より <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> には最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> が存在する。
- すると、すべての <img src="tmp/4748766bdee3e8c3a5e169791468a7aa.png" class="math-inline" /> について <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> は真（<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> が偽の最小元だから）。
- しかし帰納段階の仮定より、このとき <img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> も真でなければならない → 矛盾。

よって、そのような <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は存在せず、すべての <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> で <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が真です。

### 6. 超限帰納法と超限再帰

超限帰納法に対応する「定義の方法」として、**超限再帰（transfinite recursion）** があります。

- 超限帰納法：命題 <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> がすべての順序数で成り立つことを示す。
- 超限再帰：関数 <img src="tmp/85d22a582636ab80c149ad725895f913.png" class="math-inline" /> を「より小さい順序数での値」に基づいて定義する。

例：順序数の加法 <img src="tmp/447327d2486e4f275e7fa3e3e0720a81.png" class="math-inline" /> は、超限再帰によって定義されます。

__例題:__

以下、超限帰納法を用いる典型的な例題を出題します。  


__1. 整列集合の始片の整列性__

整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の**始片（initial segment）** を


<div class="math-display-container"><img src="tmp/f9cb8462647250590cb7ad76b20a804e.png" class="math-display" /></div>


と定義する。

次の命題を**超限帰納法**（または整列集合上の帰納法）を用いて証明せよ。

> 任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、<img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> は整列集合である。

（ヒント：<img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" />：「<img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> は整列集合である」とおき、<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に関する帰納法を用いる。）

---

以下、整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の始片 <img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> が整列集合であることを、**超限帰納法**を用いて証明します。


__証明：整列集合の始片の整列性__

__1. 記号と目標__

- <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" />：整列集合（全順序かつ任意の空でない部分集合が最小元を持つ）。
- 始片の定義：
  

<div class="math-display-container"><img src="tmp/6d9baf4e4922d4a7a0eeb905496b450b.png" class="math-display" /></div>


  （<img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> は <img src="tmp/708596235cba73cab553a9bb5933227c.png" class="math-inline" /> かつ <img src="tmp/f0a38ab7a1955eba804fc1cb5a51f1d0.png" class="math-inline" /> を意味する）。
- 目標：任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、<img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> が整列集合であることを示す。

__2. 超限帰納法の設定__

整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> 上で、命題 <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> を次のように定義する。



<div class="math-display-container"><img src="tmp/820a39a1fdad2a2c98c4dd819412956f.png" class="math-display" /></div>



超限帰納法（整列集合上の帰納法）の原理より、次を示せばよい。

> 任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、  
> 「すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> に対して <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つならば、<img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> も成り立つ」

この条件が示されれば、整列性によりすべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> で <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立つ。

__3. 帰納段階の証明__

任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> を固定し、次の仮定をおく。

- **帰納仮定**：すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> について、<img src="tmp/aacd67d43fe9ea37a5b1c921b9b935dc.png" class="math-inline" /> は整列集合である。

このとき、<img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> が整列集合であることを示す。

__(a) 全順序性の確認__

- <img src="tmp/bd9686727b6163e8d70ad8c86e077db5.png" class="math-inline" /> であり、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の順序 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> は全順序（任意の2元が比較可能）。
- 部分集合に制限した順序も同じ比較規則なので、<img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> も全順序集合である。

__(b) 任意の空でない部分集合が最小元を持つこと__

<img src="tmp/e18292334a9ce39b37e055dc77fd7d3c.png" class="math-inline" /> を任意の空でない部分集合とする。  
<img src="tmp/b9a9d90c8541e5a2dcc0288dd74269fb.png" class="math-inline" /> なので、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の空でない部分集合でもある。

- <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は整列集合なので、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> において最小元を持つ。  
  その最小元を <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> とおく：
  

<div class="math-display-container"><img src="tmp/db96f8f291295f67ef8df0183fafb96c.png" class="math-display" /></div>


- <img src="tmp/e18292334a9ce39b37e055dc77fd7d3c.png" class="math-inline" /> より、すべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> について <img src="tmp/ddca869d3d9f1412fa6644a4515bd32b.png" class="math-inline" />。  
  特に <img src="tmp/7f225c7f1db16425dc2a44f71d886d63.png" class="math-inline" /> なので、<img src="tmp/0300f92cdfac2ad7a571979ccfde5299.png" class="math-inline" />。
- したがって、<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の（<img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> における）最小元でもある。

以上より、<img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> の任意の空でない部分集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は最小元を持つ。

__(c) 結論__

(a) より <img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> は全順序集合であり、(b) より任意の空でない部分集合が最小元を持つ。  
よって <img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> は整列集合である。すなわち <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立つ。

__4. 超限帰納法の適用__

- 任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、「すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> で <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つならば <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立つ」ことを示した。
- 整列集合上の超限帰納法の原理により、**すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立つ**。

したがって、任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について <img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> は整列集合である。

__まとめ__

- 整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の始片 <img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> は、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合として同じ順序を引き継ぐので全順序。
- 任意の空でない部分集合 <img src="tmp/e18292334a9ce39b37e055dc77fd7d3c.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の空でない部分集合でもあり、整列性より最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> を持つ。
- <img src="tmp/4fe6665a8719bcecc5b22fdd9f73a4f4.png" class="math-inline" /> より <img src="tmp/7f225c7f1db16425dc2a44f71d886d63.png" class="math-inline" /> なので、<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> は <img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> における最小元。
- よって <img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> は整列集合。
- この事実を命題 <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> とおき、超限帰納法によりすべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> で成立することを示した。


---

__2. 整列集合の比較定理（弱形）__

2つの整列集合 <img src="tmp/76d737f0fd6643bedce44d8996c3db7f.png" class="math-inline" /> と <img src="tmp/3f366266c3d64d9ec26403b5741aede4.png" class="math-inline" /> が与えられているとする。

次の命題を**超限帰納法**（または整列集合上の帰納法）を用いて証明せよ。

> <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は、次のいずれか一方が成り立つ。
> 1. <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> のある始片と順序同型である。
> 2. <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> のある始片と順序同型である。
> 3. <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は順序同型である。

（ヒント：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の元を「同時に」比較しながら帰納法を適用する。  
実際の証明では、順序同型写像を超限再帰で構成し、その構成がうまくいくことを帰納法で示す。）

__3. 整列部分集合の整列性__

整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の任意の部分集合 <img src="tmp/10a8c4b92be4ca1817ba4c073dea8af4.png" class="math-inline" /> は、制限された順序に関して再び整列集合であることを、**超限帰納法**（または関連する議論）を用いて示せ。

（ヒント：直接「任意の部分集合が最小元を持つ」ことを示してもよいが、  
「<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の整列性から <img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> の整列性を導く」という観点で、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の帰納法を用いて <img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> の性質を調べる方法も考えられる。）

## 演習

### 問題

以下に、提示された内容の理解度を確認するための演習問題を出題します。  
（証明問題は、できるだけ丁寧に解答を書いてみてください。）

__問1（整列集合の定義と具体例）__

(1) 次の集合に通常の大小関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> を入れたとき、整列集合になるかどうかを判定し、理由を簡潔に述べよ。

- (a) <img src="tmp/7952a039a96398415c8e6161bcd97f66.png" class="math-inline" />
- (b) <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />（整数全体）
- (c) <img src="tmp/8a786ae7367bed5be5d8cbe4da7713e7.png" class="math-inline" />（正の有理数全体）
- (d) <img src="tmp/32ed7a2b37ae2f32a1e0f57520ba3aea.png" class="math-inline" />（0 と 1, 1/2, 1/3, … からなる集合）

(2) 整列集合の定義において、「全順序であること」がなぜ必要か、具体例を挙げて説明せよ。

__問2（直後要素・直前要素・極限要素）__

(1) 次の順序集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> について、指定された元の直後要素・直前要素が存在するか判定し、存在する場合は具体的に答えよ。

- (a) <img src="tmp/6e82299800e7ce873f7e86662e3dd8de.png" class="math-inline" />、元 <img src="tmp/671ac7bcd9ba6e4800f67e67fc67726c.png" class="math-inline" />
- (b) <img src="tmp/e721acde2f2637b63f9d9d4d201385ae.png" class="math-inline" />、元 <img src="tmp/d160f3ac8debc32683423411371b1d34.png" class="math-inline" />
- (c) <img src="tmp/f7fa9b8de264348b29e73d124bdb13dc.png" class="math-inline" />、元 <img src="tmp/b0782419bb8039fe8b6afc6b67925494.png" class="math-inline" />
- (d) <img src="tmp/370d0b7a1f97fce98a2a6d59bc55410f.png" class="math-inline" />（通常の順序）、元 <img src="tmp/748d014a08c85df236ffbfbc57dfcde3.png" class="math-inline" />

(2) 順序数 <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />（自然数全体の順序型）について、次を答えよ。

- (a) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は後続順序数か、極限順序数か。
- (b) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の直前要素は存在するか。存在する場合は何か。
- (c) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> が極限順序数である理由を、直感的に説明せよ。

(3) 順序数 <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> について、次を答えよ。

- (a) <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> は後続順序数か、極限順序数か。
- (b) <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> の直前要素は何か。

__問3（超限帰納法の理解）__

(1) 順序数全体のクラス <img src="tmp/24df8a72caec0be7051202ec37ad58da.png" class="math-inline" /> 上の超限帰納法について、次の問いに答えよ。

- (a) 超限帰納法の主張を、「基底」「帰納段階」の2条件で述べよ。
- (b) 順序数を「0」「後続順序数」「極限順序数」の3種類に分けたときの、3段階の超限帰納法の形を書け。
- (c) なぜこの3段階で「すべての順序数」について命題が成り立つといえるのか、直感的に説明せよ。

(2) 次の命題を、超限帰納法を用いて証明する方針を述べよ（完全な証明でなくてよい）。

> すべての順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は推移的集合である。  
> （すなわち、<img src="tmp/d10ac4b712413caad40a38dd7f5415f6.png" class="math-inline" /> かつ <img src="tmp/b3ac70357fd167bc08c071acb37e59a8.png" class="math-inline" /> ならば <img src="tmp/6ae7746013b4453fc4b0cd84c8f2627b.png" class="math-inline" /> が成り立つ。）

__問4（整列集合の始片の整列性）__

整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の始片を


<div class="math-display-container"><img src="tmp/6d9baf4e4922d4a7a0eeb905496b450b.png" class="math-display" /></div>


と定義する。

(1) 次の命題を、超限帰納法を用いて証明せよ。

> 任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、<img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> は整列集合である。

（ヒント：<img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" />：「<img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> は整列集合である」とおき、<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に関する帰納法を用いる。）

(2) 上の証明において、「すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> で <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つならば <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立つ」ことを示す部分で、  
<img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> の任意の空でない部分集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が最小元を持つことを、どのように示しているか、自分の言葉で説明せよ。

__問5（整列部分集合の整列性）__

整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の任意の部分集合 <img src="tmp/10a8c4b92be4ca1817ba4c073dea8af4.png" class="math-inline" /> は、制限された順序に関して再び整列集合であることを示せ。

(1) 直接、「任意の空でない部分集合が最小元を持つ」ことを示す方法で証明せよ。

(2) （発展）<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の超限帰納法を用いて、<img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> の整列性を示す方法を考え、その方針を述べよ。  
（実際に帰納法を適用する際、どのような命題 <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> を設定するか、またどのように <img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> と結びつけるかを説明せよ。）

__問6（超限帰納法の正当性）__

超限帰納法が「すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> で <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立つ」ことを保証する理由を、背理法を用いて説明せよ。

- すなわち、「ある <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> で <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が偽になる」と仮定したとき、整列性からどのように矛盾が導かれるかを示せ。

__問7（応用：順序数の加法の結合則）__

順序数の加法 <img src="tmp/447327d2486e4f275e7fa3e3e0720a81.png" class="math-inline" /> が定義されているとする。  
次の命題を、超限帰納法を用いて証明する際の「帰納の変数」と「3段階（基底・後続・極限）」の内容を具体的に書け。

> すべての順序数 <img src="tmp/e8604fb1e70a912aa41b59393ecde955.png" class="math-inline" /> について、


<div class="math-display-container"><img src="tmp/3fea9038888b9842405f35617b2ae045.png" class="math-display" /></div>


> が成り立つ。

（完全な証明は求めません。どの変数について帰納法を使うか、基底・後続・極限の各段階で何を示すべきかを書いてください。）

### 解答

以下、各問の解答です。

__問1（整列集合の定義と具体例）__

__(1) 整列集合かどうかの判定__

**(a) <img src="tmp/7952a039a96398415c8e6161bcd97f66.png" class="math-inline" />（通常の大小）**

- 整列集合である。
- 理由：有限全順序集合は、任意の空でない部分集合が有限個の元からなり、その中で最小元が必ず存在するため。

**(b) <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />（整数全体、通常の大小）**

- 整列集合ではない。
- 理由：部分集合 <img src="tmp/2a353911e1087f5856c81c5753987ae2.png" class="math-inline" />（負の整数全体）を考えると、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は空でないが最小元を持たない（いくらでも小さい負の整数がある）。

**(c) <img src="tmp/8a786ae7367bed5be5d8cbe4da7713e7.png" class="math-inline" />（正の有理数全体、通常の大小）**

- 整列集合ではない。
- 理由：任意の正の有理数 <img src="tmp/e839f148aa76d72862c2fd4779f9eba6.png" class="math-inline" /> に対して、<img src="tmp/5ce43be4dad15bfaa4660c95ffa160ec.png" class="math-inline" /> となる有理数が存在するため、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> 全体として最小元を持たない。

**(d) <img src="tmp/32ed7a2b37ae2f32a1e0f57520ba3aea.png" class="math-inline" />（0 と 1, 1/2, 1/3, …、通常の大小）**

- 整列集合である。
- 理由：
  - 全順序である（通常の実数の大小で比較可能）。
  - 任意の空でない部分集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> について：
    - もし <img src="tmp/ab27eee741a177e85b03f1baa00756d3.png" class="math-inline" /> なら、<img src="tmp/b0782419bb8039fe8b6afc6b67925494.png" class="math-inline" /> が最小元。
    - もし <img src="tmp/d72d2fc984f0075d3b6eb0bafe8375a8.png" class="math-inline" /> なら、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/aedb0d985d866e22dd82b2523b0e39e9.png" class="math-inline" /> の部分集合であり、その中で最大の <img src="tmp/42226d65b89999463ab0f61944fe7f86.png" class="math-inline" /> が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の最小元になる（<img src="tmp/42226d65b89999463ab0f61944fe7f86.png" class="math-inline" /> が最大なら、通常の大小では最小になることに注意：<img src="tmp/8a07ad36d4cb6c1dc2fb8b5b5d367ef7.png" class="math-inline" />）。
  - よって、任意の空でない部分集合に最小元が存在する。

__(2) 「全順序であること」が必要な理由__

- 整列集合は「任意の空でない部分集合が最小元を持つ」ことを要求するが、**最小元の一意性**を保証するには、任意の2元が比較可能である必要がある。
- もし部分順序集合で「比較不能な元」が存在すると、最小元が複数存在し得る（例：ハッセ図で枝分かれしている場合）。
- 例：集合 <img src="tmp/8f9a8b06e58e7d3d84d7502c607b5b2f.png" class="math-inline" /> に順序を
  - <img src="tmp/5d82b024beeb88ca6e2143ceb0acc337.png" class="math-inline" /> のみとし、<img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> と <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> は比較不能とする。
  - このとき <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 自体は空でないが、「最小元」は <img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> と <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> の両方が候補になり、一意に定まらない。
- 全順序であれば、反対称律と合わせて最小元の一意性が保証されるため、整列集合の定義に「全順序であること」が含まれる。

__問2（直後要素・直前要素・極限要素）__

__(1) 直後要素・直前要素の存在判定__

**(a) <img src="tmp/6e82299800e7ce873f7e86662e3dd8de.png" class="math-inline" />、元 <img src="tmp/671ac7bcd9ba6e4800f67e67fc67726c.png" class="math-inline" />**

- 直後要素：<img src="tmp/96a37985434f52cdc0d67896e7b7ebf2.png" class="math-inline" />（存在する）
- 直前要素：<img src="tmp/16f761877aa0971f7a254feec5b9bd10.png" class="math-inline" />（存在する）

**(b) <img src="tmp/e721acde2f2637b63f9d9d4d201385ae.png" class="math-inline" />、元 <img src="tmp/d160f3ac8debc32683423411371b1d34.png" class="math-inline" />**

- 直後要素：<img src="tmp/bc4d31b035856a6f861b1ec073bb5af9.png" class="math-inline" />（存在する）
- 直前要素：<img src="tmp/26483029b92d7483c10fc2ddf7e873e4.png" class="math-inline" />（存在する）

**(c) <img src="tmp/f7fa9b8de264348b29e73d124bdb13dc.png" class="math-inline" />、元 <img src="tmp/b0782419bb8039fe8b6afc6b67925494.png" class="math-inline" />**

- 直後要素：存在しない（任意の <img src="tmp/e839f148aa76d72862c2fd4779f9eba6.png" class="math-inline" /> に対して <img src="tmp/5ce43be4dad15bfaa4660c95ffa160ec.png" class="math-inline" /> となる有理数が存在するため）
- 直前要素：存在しない（任意の <img src="tmp/b531bb6a6be80256b8c2bc2e16adfff0.png" class="math-inline" /> に対して <img src="tmp/b85514ea8db8ee46ea407e0e3e3e5161.png" class="math-inline" /> となる有理数が存在するため）

**(d) <img src="tmp/370d0b7a1f97fce98a2a6d59bc55410f.png" class="math-inline" />、元 <img src="tmp/748d014a08c85df236ffbfbc57dfcde3.png" class="math-inline" />**

- 直後要素：<img src="tmp/04e7c4e120880e26f5110d0d71495fab.png" class="math-inline" />（存在する）
- 直前要素：<img src="tmp/28761683c454cb781fdb6aac6bd8a6df.png" class="math-inline" />（存在する）

__(2) 順序数 <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> について__

**(a) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は後続順序数か、極限順序数か**

- 極限順序数である。

**(b) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の直前要素は存在するか**

- 存在しない。

**(c) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> が極限順序数である理由（直感的説明）**

- <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は自然数全体の順序型であり、任意の自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> に対して <img src="tmp/6263c987771690e784009fcdb7d0955c.png" class="math-inline" /> だが、<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は「ある自然数の次」として得られるものではない。
- すなわち、<img src="tmp/3ca4d30d4c290190d11e133139d56f57.png" class="math-inline" /> となる順序数 <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> は存在せず、それより小さいすべての有限順序数の「極限」として現れる。

__(3) 順序数 <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> について__

**(a) <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> は後続順序数か、極限順序数か**

- 後続順序数である（<img src="tmp/33f768f9b7a5f126aa364b9d4503fe7b.png" class="math-inline" /> と書けるため）。

**(b) <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> の直前要素は何か**

- <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />

__問3（超限帰納法の理解）__

__(1) 順序数全体 <img src="tmp/24df8a72caec0be7051202ec37ad58da.png" class="math-inline" /> 上の超限帰納法__

**(a) 2条件での主張**

> 命題 <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" />（<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は順序数）が次を満たすならば、すべての順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> が成り立つ。
>
> 1. **基底**：<img src="tmp/b27dacc5676ded5b21aa21ada36bd16a.png" class="math-inline" /> が成り立つ。
> 2. **帰納段階**：任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について、  
>    「すべての <img src="tmp/b4d58078d6159ee352ef89f89ce6e36c.png" class="math-inline" /> に対して <img src="tmp/3be9b85cf782a26b88a0cfd353bdaf69.png" class="math-inline" /> が成り立つならば、<img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> も成り立つ」ことを示す。

**(b) 3段階（0・後続・極限）の形**

> すべての順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> が成り立つことを示すには、次を示せばよい。
>
> 1. **基底**：<img src="tmp/b27dacc5676ded5b21aa21ada36bd16a.png" class="math-inline" /> を示す。
> 2. **後続段階**：任意の順序数 <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> について、<img src="tmp/3be9b85cf782a26b88a0cfd353bdaf69.png" class="math-inline" /> が成り立つならば <img src="tmp/d3956bde88781497b0ed0243360459f1.png" class="math-inline" /> も成り立つことを示す。
> 3. **極限段階**：任意の極限順序数 <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> について、  
>    「すべての <img src="tmp/6d2307b6847968dcfb26600c94b7b34e.png" class="math-inline" /> に対して <img src="tmp/3be9b85cf782a26b88a0cfd353bdaf69.png" class="math-inline" /> が成り立つならば <img src="tmp/d2d07d491ce308b6749dfc422f75d6b4.png" class="math-inline" /> も成り立つ」  
>    ことを示す。

**(c) なぜ3段階で十分か（直感的説明）**

- 任意の 0 でない順序数は、後続順序数か極限順序数のどちらかである。
- 基底 <img src="tmp/b27dacc5676ded5b21aa21ada36bd16a.png" class="math-inline" /> から出発し、
  - 後続段階で「<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> で成り立てば <img src="tmp/836d5fcdc9c61ebd81305199531dce54.png" class="math-inline" /> でも成り立つ」ことを繰り返すことで、すべての後続順序数に命題が伝播する。
  - 極限段階で「<img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> より小さいすべての順序数で成り立てば <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> でも成り立つ」ことを保証することで、極限順序数にも命題が及ぶ。
- 順序数全体は整列集合（より正確には整列クラス）なので、この3段階を組み合わせることで、すべての順序数について <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> が成立することが示される。

__(2) 「すべての順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は推移的集合である」の証明方針__

- 目標：任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について、<img src="tmp/d10ac4b712413caad40a38dd7f5415f6.png" class="math-inline" /> かつ <img src="tmp/b3ac70357fd167bc08c071acb37e59a8.png" class="math-inline" /> ならば <img src="tmp/6ae7746013b4453fc4b0cd84c8f2627b.png" class="math-inline" /> が成り立つことを示す。
- 方針：<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> に関する超限帰納法を用いる。
  - **基底**：<img src="tmp/7bdfa5c2d9a8b22b2dcf6a5669c10b6a.png" class="math-inline" /> のとき、<img src="tmp/e74407ba198fa7e591b2603f09b67ec7.png" class="math-inline" /> は空虚に推移的（含む元がないので条件は自動的に成立）。
  - **帰納段階**：任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> を固定し、「すべての <img src="tmp/4dca5616f84e318dc6302afb5adb665a.png" class="math-inline" /> について <img src="tmp/865903777a19d290dd9395c606980456.png" class="math-inline" /> は推移的」と仮定して、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が推移的であることを示す。
    - <img src="tmp/d10ac4b712413caad40a38dd7f5415f6.png" class="math-inline" /> かつ <img src="tmp/b3ac70357fd167bc08c071acb37e59a8.png" class="math-inline" /> とする。
    - 順序数の定義により、<img src="tmp/b4d58078d6159ee352ef89f89ce6e36c.png" class="math-inline" /> なので、帰納仮定より <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> は推移的：<img src="tmp/b3ac70357fd167bc08c071acb37e59a8.png" class="math-inline" /> かつ <img src="tmp/d10ac4b712413caad40a38dd7f5415f6.png" class="math-inline" /> なら <img src="tmp/6ae7746013b4453fc4b0cd84c8f2627b.png" class="math-inline" /> が成り立つ（順序数は推移的集合として定義されることが多いが、ここではその事実自体を帰納法で示す流れ）。
  - 以上より、すべての順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は推移的集合である。

__問4（整列集合の始片の整列性）__

__(1) 超限帰納法による証明__

整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の始片を


<div class="math-display-container"><img src="tmp/6d9baf4e4922d4a7a0eeb905496b450b.png" class="math-display" /></div>


と定義する。

**命題**：任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、<img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> は整列集合である。

**証明**：

- <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" />：「<img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> は整列集合である」と定義する。
- 超限帰納法（整列集合上の帰納法）により、次を示せばよい：
  > 任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、  
  > 「すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> に対して <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つならば、<img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> も成り立つ」

**帰納段階**：

- 任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> を固定し、帰納仮定として「すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> について <img src="tmp/aacd67d43fe9ea37a5b1c921b9b935dc.png" class="math-inline" /> は整列集合である」と仮定する。
- <img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> が整列集合であることを示す。

**(a) 全順序性**

- <img src="tmp/bd9686727b6163e8d70ad8c86e077db5.png" class="math-inline" /> であり、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の順序 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> は全順序。
- 部分集合に制限した順序も同じ比較規則なので、<img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> も全順序集合である。

**(b) 任意の空でない部分集合が最小元を持つこと**

- <img src="tmp/e18292334a9ce39b37e055dc77fd7d3c.png" class="math-inline" /> を任意の空でない部分集合とする。
- <img src="tmp/b9a9d90c8541e5a2dcc0288dd74269fb.png" class="math-inline" /> より、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の空でない部分集合でもある。
- <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は整列集合なので、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> において最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> を持つ：
  

<div class="math-display-container"><img src="tmp/db96f8f291295f67ef8df0183fafb96c.png" class="math-display" /></div>


- <img src="tmp/e18292334a9ce39b37e055dc77fd7d3c.png" class="math-inline" /> より、すべての <img src="tmp/a7f9d4172a33503da00b0487b96e37ea.png" class="math-inline" /> について <img src="tmp/ddca869d3d9f1412fa6644a4515bd32b.png" class="math-inline" />。特に <img src="tmp/7f225c7f1db16425dc2a44f71d886d63.png" class="math-inline" /> なので <img src="tmp/0300f92cdfac2ad7a571979ccfde5299.png" class="math-inline" />。
- よって、<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の（<img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> における）最小元でもある。

**(c) 結論**

- (a) より <img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> は全順序集合であり、(b) より任意の空でない部分集合が最小元を持つ。
- したがって <img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> は整列集合、すなわち <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立つ。

**超限帰納法の適用**：

- 任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、「すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> で <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つならば <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立つ」ことを示した。
- 整列集合上の超限帰納法の原理により、すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立つ。
- よって、任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について <img src="tmp/2fd4785383602d954d461e2aa9adf0ef.png" class="math-inline" /> は整列集合である。

__(2) 「<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が最小元を持つ」部分の説明（自分の言葉）__

- <img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> の空でない部分集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> を考えると、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の空でない部分集合でもある。
- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> は整列集合なので、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> において最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> を持つ。
- <img src="tmp/e18292334a9ce39b37e055dc77fd7d3c.png" class="math-inline" /> なので、<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> も <img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> に属し、<img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> における順序は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の順序の制限なので、<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の <img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> における最小元でもある。
- したがって、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の整列性を「借りて」<img src="tmp/75373e28b8f8f1f2f33d9a2ec0c0e7d5.png" class="math-inline" /> の整列性を示している。

__問5（整列部分集合の整列性）__

整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の任意の部分集合 <img src="tmp/10a8c4b92be4ca1817ba4c073dea8af4.png" class="math-inline" /> は、制限された順序に関して整列集合であることを示す。

__(1) 直接証明__

- <img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合であり、順序は <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> の制限なので、<img src="tmp/9358ed05c8cea6a54bcd5de9f207d08b.png" class="math-inline" /> は全順序集合である。
- <img src="tmp/cb297623d58f6a11df7b2f1ac56eddf1.png" class="math-inline" /> を任意の空でない部分集合とする。<img src="tmp/4e3b66b193e486780a076db755e620f5.png" class="math-inline" /> より、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の空でない部分集合でもある。
- <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は整列集合なので、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> において最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> を持つ：
  

<div class="math-display-container"><img src="tmp/db96f8f291295f67ef8df0183fafb96c.png" class="math-display" /></div>


- <img src="tmp/cb297623d58f6a11df7b2f1ac56eddf1.png" class="math-inline" /> より <img src="tmp/4beb0b9a27e76882224a64ed293e8504.png" class="math-inline" /> であり、<img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> 上の順序は <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の順序の制限なので、<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の（<img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> における）最小元でもある。
- よって、<img src="tmp/9358ed05c8cea6a54bcd5de9f207d08b.png" class="math-inline" /> の任意の空でない部分集合は最小元を持つ。
- 以上より <img src="tmp/9358ed05c8cea6a54bcd5de9f207d08b.png" class="math-inline" /> は整列集合である。

__(2) 超限帰納法を用いる方針（発展）__

- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の超限帰納法を用いて、<img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> の整列性を示すことを考える。
- 命題 <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> を次のように設定する：
  

<div class="math-display-container"><img src="tmp/2fd36f7aad1cd73681145640ec7a523b.png" class="math-display" /></div>


  - ここで <img src="tmp/0ca2700dc714cdd297195e6b28585d31.png" class="math-inline" /> は <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> 以下の元全体。
- 帰納法のステップ：
  - 基底：最小元 <img src="tmp/53d768e41950280cba385ae6d8d9433f.png" class="math-inline" /> について、<img src="tmp/08ef74dc5b2d651b92ea85b7be11667d.png" class="math-inline" /> は高々1点（<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> が <img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> に属するかどうかで変わる）であり、明らかに整列集合。
  - 帰納段階：任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、「すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> で <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つ」と仮定し、<img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> を示す。
    - <img src="tmp/f8259cdd9503408df78a2d07f98a98b2.png" class="math-inline" /> は、<img src="tmp/e3295c711ac571f1857ab6c946dfbc4b.png" class="math-inline" /> と（もし <img src="tmp/d3f9f939f6fb72a7445aea94dd86f149.png" class="math-inline" /> なら）一点 <img src="tmp/259bf23162500191d68f279c99edb0a5.png" class="math-inline" /> の和集合として書ける。
    - 帰納仮定より <img src="tmp/e3295c711ac571f1857ab6c946dfbc4b.png" class="math-inline" /> は整列集合であり、一点集合も整列集合なので、それらの和集合（適切に順序を入れたもの）も整列集合であることを示す（通常、整列集合に最大元を1つ加えたものは再び整列集合）。
- すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> で <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立てば、特に <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> を <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の最大元（存在しない場合は上限を考える）とすることで <img src="tmp/b21fd7f89f485021e5ed1b8b46a68acf.png" class="math-inline" /> 全体が整列集合であることが従う。

__問6（超限帰納法の正当性）__

超限帰納法が「すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> で <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が成り立つ」ことを保証する理由を、背理法で説明する。

- 整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> と命題 <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> に対し、超限帰納法の条件：
  > 任意の <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> について、  
  > 「すべての <img src="tmp/339ba623826aed9d149f49e4338df8a4.png" class="math-inline" /> に対して <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つならば <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> も成り立つ」
  が成り立っているとする。
- ここで、ある <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> で <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が偽になると仮定する。
- <img src="tmp/74a98993af0407d00d574844df3de5bf.png" class="math-inline" /> とおくと、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は空でない。
- <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は整列集合なので、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> には最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> が存在する。
- <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の最小元であることから、すべての <img src="tmp/4748766bdee3e8c3a5e169791468a7aa.png" class="math-inline" /> について <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> は真（もし偽なら <img src="tmp/5dfa1c9a5594739fe4208fe56de564cd.png" class="math-inline" /> となり <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> の最小性に反する）。
- したがって、「すべての <img src="tmp/4748766bdee3e8c3a5e169791468a7aa.png" class="math-inline" /> に対して <img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> が成り立つ」が真である。
- 超限帰納法の条件より、このとき <img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> も成り立たなければならない。
- しかし <img src="tmp/9507e0235f1b54262f27fa314e741f93.png" class="math-inline" /> より <img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> は偽 → 矛盾。
- よって、そのような <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は存在せず、すべての <img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> で <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> が真である。

__問7（順序数の加法の結合則）__

命題：
> すべての順序数 <img src="tmp/e8604fb1e70a912aa41b59393ecde955.png" class="math-inline" /> について、


<div class="math-display-container"><img src="tmp/3fea9038888b9842405f35617b2ae045.png" class="math-display" /></div>


> が成り立つ。

__帰納の変数__

- <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> を固定し、**<img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> に関する超限帰納法**を用いる。

__3段階の内容__

**1. 基底（<img src="tmp/476c4f00ad6213162e9e2650d47cea18.png" class="math-inline" />）**

- 示すべきこと：
  

<div class="math-display-container"><img src="tmp/c00ccd24fdb3d4b40f8c35b6d9afd9cf.png" class="math-display" /></div>


- 順序数の加法の定義（<img src="tmp/538efefe7d547c21cab4df1544f2c2bd.png" class="math-inline" />）より、
  - 左辺：<img src="tmp/078a33a2f8e17170514cb06dd1a52f00.png" class="math-inline" />
  - 右辺：<img src="tmp/2e6e819154fb64f56081e8888e98648c.png" class="math-inline" />
- よって基底では等式が成立。

**2. 後続段階（<img src="tmp/06ad901881b25f85541cc8e388df0716.png" class="math-inline" />）** 

- 帰納仮定：<img src="tmp/865903777a19d290dd9395c606980456.png" class="math-inline" /> については
  

<div class="math-display-container"><img src="tmp/cceb1ca31c229210a84c401af91d479e.png" class="math-display" /></div>


  が成り立つと仮定する。
- 示すべきこと：<img src="tmp/06ad901881b25f85541cc8e388df0716.png" class="math-inline" /> について
  

<div class="math-display-container"><img src="tmp/5d45008a28603ca0d651f0e0a315dc31.png" class="math-display" /></div>


- 順序数の加法の定義（<img src="tmp/b94de5a814171f926efff54637dfdaf8.png" class="math-inline" />）を用いて変形する：
  - 左辺：<img src="tmp/90d95536a9f65a2dbd8b59660d1d208e.png" class="math-inline" />
  - 右辺：<img src="tmp/c0d7dc7ae3f4e70d0ed3c59d557fad9b.png" class="math-inline" />
- 帰納仮定より <img src="tmp/b465391fbec9b0e5acec1be5c17b3af1.png" class="math-inline" /> なので、両辺に <img src="tmp/c3debf0e3146a85cc074c584d1062adb.png" class="math-inline" /> を施しても等しい。
- よって後続段階でも等式が成立。

**3. 極限段階（<img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> が極限順序数）**

- <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> を極限順序数とする。
- 帰納仮定：すべての <img src="tmp/c1924ed181f80bf1ed9c99d3d8f08ec8.png" class="math-inline" /> について
  

<div class="math-display-container"><img src="tmp/cceb1ca31c229210a84c401af91d479e.png" class="math-display" /></div>


  が成り立つと仮定する。
- 示すべきこと：
  

<div class="math-display-container"><img src="tmp/3fea9038888b9842405f35617b2ae045.png" class="math-display" /></div>


- 順序数の加法の定義（極限順序数に対する和は「それより小さい順序数での和の上限」）より：
  - 左辺： <img src="tmp/f68b93764d7c3b71ffb8f63c319a2a28.png" class="math-inline" />
  - 右辺： <img src="tmp/b4b5179e46395208325d4d1d7f911396.png" class="math-inline" />
- 帰納仮定より、各 <img src="tmp/c1924ed181f80bf1ed9c99d3d8f08ec8.png" class="math-inline" /> について <img src="tmp/b465391fbec9b0e5acec1be5c17b3af1.png" class="math-inline" /> なので、両者の上限も等しい。
- よって極限段階でも等式が成立。

以上より、<img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> に関する超限帰納法により、すべての <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> について結合則が成り立つ。  
（<img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> は任意だったので、すべての <img src="tmp/e8604fb1e70a912aa41b59393ecde955.png" class="math-inline" /> について成立。）

<div style="page-break-before:always"></div>



# Zornの補題と整列可能定理

## イデアル

イデアルの数学的な定義について、順を追って説明します。

### 1. イデアルとは何か（直感的なイメージ）

イデアルは、**環（ring）** という代数構造の中で定義される特別な部分集合です。  
直感的には、

- 「ある性質を保ったまま、0 に近い元の集まり」
- 「割り算の余りを無視するときに残る元の集まり」

と考えるとイメージしやすいです。

例：整数環 ℤ で「偶数全体の集合」を考えます。

- 偶数＋偶数＝偶数  
- 偶数×任意の整数＝偶数

このように、「偶数全体」は、足し算と整数倍に対して閉じています。  
この「偶数全体の集合」が、ℤ のイデアルの一例です。

### 2. 環（ring）の復習

イデアルを定義する前に、**環**の定義を簡単に確認します。

環とは、集合 <img src="tmp/a0f7a58538051aad071147a20c796777.png" class="math-inline" /> とその上の二つの演算

- 加法：<img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />（可換群をなす）
- 乗法：<img src="tmp/356513ded34229fab8ea72b65824828d.png" class="math-inline" />（結合的で、単位元 1 を持つことが多い）

が定義され、次の条件を満たす代数系です。

1. <img src="tmp/e23d22c3ed1394e2db2fb5143a2c30ba.png" class="math-inline" /> は可換群（加法についてアーベル群）
2. 乗法は結合的：<img src="tmp/9c2d4b1ce866bf7919c3822b73978c06.png" class="math-inline" />
3. 乗法の単位元 1 が存在：<img src="tmp/05c77721e7a085d7145c1fec12699b46.png" class="math-inline" />
4. 分配法則：  
   - <img src="tmp/9df7c1a8791128bdb1bf257e3c5d703d.png" class="math-inline" />  
   - <img src="tmp/6c098c26c035650ca33efab9a86f4bb5.png" class="math-inline" />

>__可換群（commutative group）__  
>**可換群** とは、**足し算のように「逆元が存在し、交換法則が成り立つ」演算を持つ集合**のことです。
>もう少し正確に言うと、集合 <img src="tmp/b15d92e08fe483b65746431228085e71.png" class="math-inline" /> とその上の演算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" /> が次の条件を満たすとき、<img src="tmp/6d6ba8235e95d8b8558d67f371bcc5b3.png" class="math-inline" /> は可換群です。
>1. **結合則**：<img src="tmp/78a873d60bee06923194c2921fd9d2c8.png" class="math-inline" />
>2. **単位元の存在**：ある元 <img src="tmp/cbded2a2600de3d27cf0eee2bf27b0ec.png" class="math-inline" /> が存在して、すべての <img src="tmp/3fff980604c814e2b9b235523d0573fe.png" class="math-inline" /> について <img src="tmp/80ca6baeadf61f465c67c21d16c09843.png" class="math-inline" />
>3. **逆元の存在**：各 <img src="tmp/3fff980604c814e2b9b235523d0573fe.png" class="math-inline" /> に対して、ある <img src="tmp/52253408585c2fddab3146a5ad928522.png" class="math-inline" /> が存在して <img src="tmp/2a602bf3561ab6054eab3831d570d813.png" class="math-inline" />
>4. **交換法則（可換性）** ：<img src="tmp/a57134a054c9aad120adb603ca3aef69.png" class="math-inline" />
>__具体例__  
>- 整数全体 ℤ と通常の足し算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />  
>- 実数全体 ℝ と足し算 <img src="tmp/bab28d11f6d7a30c1903327aeca7193a.png" class="math-inline" />  
>- ベクトル空間のベクトル同士の足し算
>これらはすべて可換群です。
>__環との関係__  
>環 <img src="tmp/ac09066d037a4fc2518d68cf9691a2cd.png" class="math-inline" /> では、
>- <img src="tmp/e23d22c3ed1394e2db2fb5143a2c30ba.png" class="math-inline" /> が可換群であることが要求されます。
>- つまり、環の「足し算部分」は常に可換群です。
>要するに、**可換群＝「足し算がきちんとできる集合」** と考えてください。


代表例：
- 整数全体 ℤ
- 実数係数の多項式環 ℝ[x]
- 行列環 <img src="tmp/afebcf945ec290c8fb8af6e4562217f8.png" class="math-inline" />

### 3. イデアルの定義（数学的に）

環 <img src="tmp/a0f7a58538051aad071147a20c796777.png" class="math-inline" /> の部分集合 <img src="tmp/d5411442254385b75f0ed36f34d8257f.png" class="math-inline" /> が**イデアル（ideal）** であるとは、次の2条件を満たすことです。

__(1) 加法について部分群である__
- <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> は加法について閉じている：  
  <img src="tmp/2ee53f86b439fd6d813dae0780406b25.png" class="math-inline" />
- 加法の逆元が <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> に属する：  
  <img src="tmp/56fddbb8d33f896c70046bdcac0e5e0b.png" class="math-inline" />
- 特に、<img src="tmp/7eaecb3426cb53752afdc30c6dce003e.png" class="math-inline" />（零元が含まれる）

まとめると、<img src="tmp/32b6d4c0c52243320111e20a0ae6d4af.png" class="math-inline" /> は <img src="tmp/e23d22c3ed1394e2db2fb5143a2c30ba.png" class="math-inline" /> の部分群です。

__(2) 「外からの掛け算」で閉じている__

任意の <img src="tmp/baa629cd0c53a8e2c7a7b2887ac0149a.png" class="math-inline" /> と任意の <img src="tmp/c72124c39d45ea1a6aa9a9bd1ed14986.png" class="math-inline" /> について、

- 左から掛けても <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> に入る：<img src="tmp/b326fd25f1d25fd5480b50ddfcd0994a.png" class="math-inline" />
- 右から掛けても <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> に入る：<img src="tmp/2ee6560b7196b1d5c54bcd29fc7dd772.png" class="math-inline" />

これをまとめて、



<div class="math-display-container"><img src="tmp/ad5302a9aa971a70b2be3607204f9bdd.png" class="math-display" /></div>



と書きます。  
可換環の場合は <img src="tmp/2ef4ca6142e213dd03fe8fc39014e940.png" class="math-inline" /> なので、どちらか一方の条件で十分です。

### 4. 左イデアル・右イデアル・両側イデアル

環が非可換（行列環など）の場合、条件を弱めた概念もあります。

- **左イデアル**：  
  <img src="tmp/ca6486e6ffd1800a3ba67d4779854a98.png" class="math-inline" />
- **右イデアル**：  
  <img src="tmp/6add1f1334dc045ab3f40c9a2e9a77dd.png" class="math-inline" />
- **両側イデアル（単にイデアル）**：  
  左イデアルかつ右イデアル

可換環（ℤ, ℝ[x] など）では、左・右・両側の区別はなく、単に「イデアル」と呼びます。

### 5. 具体例

__例1：整数環 ℤ のイデアル__

ℤ のイデアルは、すべて



<div class="math-display-container"><img src="tmp/f999380a3641fd8a5c5178bf4abbac9d.png" class="math-display" /></div>



の形をしています（<img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> は 0 以上の整数）。

- <img src="tmp/ac530716190fa3ec148b9fc37d3fc192.png" class="math-inline" />：<img src="tmp/4774377971f2d39bdd07f44c6eeb2e2b.png" class="math-inline" />（自明なイデアル）
- <img src="tmp/b1ade42e5253f896f50191fa3af8d74c.png" class="math-inline" />：ℤ 全体（単位イデアル）
- <img src="tmp/c366e5b050567aa8f7cbd6c8aa16694d.png" class="math-inline" />：偶数全体
- <img src="tmp/53e79e5cc9147f8deaa36b77c6fe6094.png" class="math-inline" />：3の倍数全体

これらはすべて、  
- 加法について閉じている  
- 任意の整数倍で閉じている  
ので、イデアルの条件を満たします。

__例2：多項式環 ℝ[x] のイデアル__

ℝ[x] のイデアルの例：

- 定数項 0 の多項式全体：  
  <img src="tmp/cca5a6bd64718bfe877931f3209395ab.png" class="math-inline" />
- ある多項式 <img src="tmp/3fdae9fef92593e4e976f831339ce3a1.png" class="math-inline" /> で割り切れる多項式全体：  
  <img src="tmp/e941c0a42d7636b7bba0a5d0cffde18e.png" class="math-inline" />

これらも、和と任意の多項式倍で閉じているのでイデアルです。

__例3:イデアルのイメージ1__

このコードでは、数直線上に整数をプロットし、イデアル <img src="tmp/e17b7dec162768e4e57a04d8e692f266.png" class="math-inline" /> に属する点を赤で、それ以外を青で表示します。これにより、「一定の間隔で並ぶ点の集まり」がイデアルであることが視覚的にわかります。

<img src="image/8_lemma_well_ordered/1782734430668.png" alt="" width="500" style="display: block; margin: 0 auto;">


```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_integer_ideal(n, xmin=-10, xmax=10):
    """
    整数環 ℤ のイデアル nℤ を可視化する。
    nℤ = { ..., -2n, -n, 0, n, 2n, ... }
    """
    # 整数点の生成
    integers = np.arange(xmin, xmax + 1)
    ideal_multiples = integers[integers % n == 0]
    non_ideal = integers[integers % n != 0]

    plt.figure(figsize=(10, 2))
    # イデアルに属する点（赤）
    plt.scatter(ideal_multiples, np.zeros_like(ideal_multiples),
               color='red', s=100, label=f'{n}ℤ (ideal)', zorder=3)
    # イデアルに属さない点（青）
    plt.scatter(non_ideal, np.zeros_like(non_ideal),
               color='blue', s=50, label=f'ℤ ∖ {n}ℤ', zorder=2)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.yticks([])
    plt.xlabel('Integers ℤ')
    plt.title(f'Visualization of the ideal {n}ℤ in the integer ring ℤ')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# 例：偶数全体 2ℤ
visualize_integer_ideal(2)
```

__例4:イデアルのイメージ2__

イデアル <img src="tmp/e17b7dec162768e4e57a04d8e692f266.png" class="math-inline" /> で割った世界（mod n）を、円周上の点として可視化します。

この可視化は、イデアル <img src="tmp/e17b7dec162768e4e57a04d8e692f266.png" class="math-inline" /> で割ることで、整数が「n で割った余り」だけに分類される→その余りを円周上の点として表す。
というイメージを示します。

<img src="image/8_lemma_well_ordered/1782734543503.png" alt="" width="400" style="display: block; margin: 0 auto;">

```python
def visualize_quotient_ring_mod_n(n):
    """
    剰余環 ℤ/nℤ を円周上の点として可視化する。
    イデアル nℤ で割ると、0,1,...,n-1 の n 個の元になる。
    """
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=100, color='red', zorder=3)
    for i in range(n):
        plt.text(x[i]*1.1, y[i]*1.1, str(i), fontsize=12,
                 ha='center', va='center')

    circle = plt.Circle((0,0), 1, fill=False, color='black', linestyle='--')
    plt.gca().add_artist(circle)
    plt.axis('equal')
    plt.title(f'Visualization of the quotient ring ℤ/{n}ℤ (mod {n} world)')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# 例：ℤ/5ℤ
visualize_quotient_ring_mod_n(5)
```


### 6. イデアルが重要な理由

イデアルは、次のような理由で数学的に重要です。

1. **剰余環（商環）の構成**  
   イデアル <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> があると、環 <img src="tmp/a0f7a58538051aad071147a20c796777.png" class="math-inline" /> を <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> で「割った」環 <img src="tmp/703058f292dd58a822993a1f4885f841.png" class="math-inline" /> を構成できます。  
   これは、整数の mod n の一般化です。

2. **環の構造の分析**  
   イデアルを通じて、環が単純かどうか（単純環）、既約分解できるかどうか（ネーター環・アルティン環）などを調べます。

3. **代数幾何との対応**  
   可換環のイデアルと、幾何的な対象（アフィン代数多様体）が一対一に対応します（ヒルベルトの零点定理）。

4. **数論・代数幾何・表現論など、多くの分野で基本概念**  
   整数の素因数分解の一般化（デデキント環）、多項式の零点集合の研究など、広く使われます。

__確認問題__

極大イデアルに関する確認問題を5題出題します。  
基礎的な定義から、Zornの補題を使った存在証明まで含みます。

__確認問題1：定義と基本性質__

可換環 <img src="tmp/a0f7a58538051aad071147a20c796777.png" class="math-inline" />（単位元 1 を持つ）のイデアル <img src="tmp/547254580c3bcb6f42dc1a222097c57f.png" class="math-inline" /> が**極大イデアル**であるとは、次の2条件を満たすことである。

1. <img src="tmp/a32068d465d03974ad681d43a50b9102.png" class="math-inline" />
2. <img src="tmp/f876fad0b6e5c8881b09f268594c2a6e.png" class="math-inline" /> となるイデアル <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> は存在しない。

このとき、次の問いに答えよ。

(1) 極大イデアル <img src="tmp/547254580c3bcb6f42dc1a222097c57f.png" class="math-inline" /> について、剰余環 <img src="tmp/5bcccf6c3c82a8f48ddf7e156a19d30c.png" class="math-inline" /> は体であることを示せ。  
(2) 逆に、<img src="tmp/5bcccf6c3c82a8f48ddf7e156a19d30c.png" class="math-inline" /> が体ならば <img src="tmp/547254580c3bcb6f42dc1a222097c57f.png" class="math-inline" /> は極大イデアルであることを示せ。

__確認問題2：整数環 ℤ の極大イデアル__

整数環 ℤ の極大イデアルをすべて求めよ。  
また、それぞれの剰余環 ℤ/<img src="tmp/547254580c3bcb6f42dc1a222097c57f.png" class="math-inline" /> が体であることを確認せよ。

__確認問題3：多項式環 ℝ[x] の極大イデアル__

実数係数の多項式環 ℝ[x] について、次の問いに答えよ。

(1) 1次多項式 <img src="tmp/5632438dd16d223036c4e56520feb86e.png" class="math-inline" />（<img src="tmp/a9c630de3c66e163d6c81154716a909f.png" class="math-inline" />）で生成されるイデアル <img src="tmp/9fd4fe60e5d0b432afaa473a1cb504d5.png" class="math-inline" /> は極大イデアルであることを示せ。  
(2) ℝ[x] の極大イデアルはすべて <img src="tmp/9fd4fe60e5d0b432afaa473a1cb504d5.png" class="math-inline" /> の形であるかどうか、理由とともに答えよ。


__解答__

__問題1__
- (1) <img src="tmp/5bcccf6c3c82a8f48ddf7e156a19d30c.png" class="math-inline" /> の0でない元 <img src="tmp/11c17f13db2c27fd29dc69a00d0ab0d3.png" class="math-inline" /> に対し、<img src="tmp/49df40f0cccbf0a43f85d452322fc047.png" class="math-inline" /> より <img src="tmp/ebdc58767046359c0c6836d2661df008.png" class="math-inline" /> となるので、逆元が存在。  
- (2) <img src="tmp/5bcccf6c3c82a8f48ddf7e156a19d30c.png" class="math-inline" /> が体なら、<img src="tmp/1a4ff8712cca51a6570daefff84feea6.png" class="math-inline" /> となるイデアル <img src="tmp/18d3e06c362797434b331bdd851d8255.png" class="math-inline" /> に対し、<img src="tmp/4c15c3f9dcf8ae92c64b12333924aa43.png" class="math-inline" /> は <img src="tmp/5bcccf6c3c82a8f48ddf7e156a19d30c.png" class="math-inline" /> の自明でないイデアルだが、体のイデアルは自明のみなので矛盾。

__問題2__
- ℤ の極大イデアルは <img src="tmp/18992832a95d2c4e44d194e2b6c2e855.png" class="math-inline" />（<img src="tmp/eb83e8cbc0f1e2e809a7c7e8a5cef02f.png" class="math-inline" /> は素数）。  
- ℤ/<img src="tmp/18992832a95d2c4e44d194e2b6c2e855.png" class="math-inline" /> は有限体。

__問題3__
- (1) ℝ[x]/(x-a) ≅ ℝ（評価写像）より体。  
- (2) いいえ。例：<img src="tmp/ba35faefc01943a25cc0c43ae5281fb4.png" class="math-inline" /> は ℝ[x] の極大イデアルだが、1次多項式ではない。


>__鎖（chain）__
>集合における**鎖（chain）** とは、**半順序集合の中で、すべての元が互いに比較できる部分集合**のことです。
>もう少し詳しく言うと：
>- 半順序集合 <img src="tmp/9464fc33a5218a96fea9b87677ec06cf.png" class="math-inline" /> の部分集合 <img src="tmp/26a42c3438cde3a14157b1a39b5ac1f2.png" class="math-inline" /> が鎖であるとは、
>  - 任意の <img src="tmp/2156f263db38b9f46fb4cc3153533573.png" class="math-inline" /> について、<img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> または <img src="tmp/708596235cba73cab553a9bb5933227c.png" class="math-inline" /> のどちらかが成り立つ
  ことです。
>つまり、**鎖は「全順序になっている部分集合」** です。
>以下絵のオレンジのノードをつなげた部分集合は、半順序集合の中で「互いに比較できる元の列」として表現されてます。これが鎖のイメージです。
><img src="image/8_lemma_well_ordered/1782817871135.png" alt="" width="500" style="display: block; margin: 0 auto;">
>例
>- 整数の通常の順序 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> で、<img src="tmp/3c5611768ab51283645b646540939de6.png" class="math-inline" /> は鎖（1<2<3<4<5）。
>- 自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> も鎖（0<1<2<...）。
>- 集合の包含関係で、<img src="tmp/439517d356cbdea8114211b221dbf5d6.png" class="math-inline" /> は鎖（<img src="tmp/316b3294e7c6c8f2c43b5789a4c37ef4.png" class="math-inline" />）。
>__なぜ重要か__
>- Zornの補題では、「任意の鎖に上界がある」という条件が鍵になります。
>- 整列集合は、特に「強い鎖」（どの部分集合にも最小元がある）と見なせます。
>- 鎖は、**順序構造を調べるための基本的な道具**として、集合論・順序論・代数などで広く使われます。
>要するに、**鎖＝互いに比較できる元の列**です。





## 帰納的順序とZornの補題


以下では、順序集合の一般論に基づいて「帰納的順序」と「Zornの補題」の数学的な定義を説明します。

### 1. 帰納的順序（inductive order）の定義

__1.1 準備：上界・極大元__

<img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" /> を**半順序集合**（反射的・反対称・推移的）とします。

- **上界（upper bound）**  
  部分集合 <img src="tmp/3f34510ec61991757bec0d031992e0d1.png" class="math-inline" /> に対し、<img src="tmp/3df7c5d08ea26c3346585533a07c62bd.png" class="math-inline" /> が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**上界**であるとは、
  

<div class="math-display-container"><img src="tmp/b0db4ede41f20f6092d3de4aca7c3a7d.png" class="math-display" /></div>


  が成り立つことをいう。

- **極大元（maximal element）**  
  元 <img src="tmp/e7fb43b3ae1ad2b729e48533f72bc3b6.png" class="math-inline" /> が**極大元**であるとは、
  

<div class="math-display-container"><img src="tmp/024205c7411327ba31f46d8d19734a56.png" class="math-display" /></div>


  が成り立つこと、すなわち「<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> より真に大きい元は存在しない」ことをいう。

__1.2 帰納的順序の定義__

半順序集合 <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" /> が**帰納的（inductive）** であるとは、次の条件を満たすことをいいます。

> **任意の全順序部分集合（鎖）<img src="tmp/26a42c3438cde3a14157b1a39b5ac1f2.png" class="math-inline" /> が、<img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> において上界を持つ。**

ここで「全順序部分集合（鎖）」とは、<img src="tmp/7da569b3f6d6a9ad1defdb55cccf0f5f.png" class="math-inline" /> の任意の2元が比較可能な部分集合のことです。

**言い換え**：
- <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> のどんな「一直線に並んだ部分」<img src="tmp/7da569b3f6d6a9ad1defdb55cccf0f5f.png" class="math-inline" /> を取っても、その「先」を指し示す元（上界）が <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> の中に存在する。
- この条件は、「鎖が無限に伸び続けることがあっても、その極限的な候補が <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> の中に存在する」ことを要請しています。

### 2. Zornの補題（Zorn’s lemma）の定義

__2.1 主張__

**Zornの補題**は、選択公理と同値な命題で、次のように述べられます。

> **任意の帰納的半順序集合は、少なくとも1つの極大元を持つ。**

記号で書くと：

- <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" />：半順序集合
- <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" /> が帰納的（任意の鎖が上界を持つ）ならば、
  

<div class="math-display-container"><img src="tmp/b81dcd1d635461bb825c509c95f51b92.png" class="math-display" /></div>



__2.2 直感的な意味__

- 帰納的順序は「鎖がどこまでも伸びるなら、その先を受け止める元が存在する」という条件。
- Zornの補題は、「そのような『先を受け止める元』の中に、それ以上大きくできない“極大な元”が必ず存在する」と主張します。
- これは「無限に長い鎖をたどっていくと、どこかで行き止まり（極大元）にぶつかる」というイメージです。

### 3. 具体例と使い方のイメージ

__3.1 帰納的順序の例__

- 集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の部分集合全体 <img src="tmp/8cfe81bf02d6a6b87429326a0c42d2a7.png" class="math-inline" /> に包含関係 <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" /> を入れたもの：
  - 任意の鎖（包含に関して全順序な部分集合族）<img src="tmp/cb41b9187d15e3b99613894f054220cd.png" class="math-inline" /> に対し、その合併 <img src="tmp/853ade47cfc0e7dbcede8ce461044bac.png" class="math-inline" /> が上界になる。
  - よって <img src="tmp/78f23e0414f7bc13b1d6bc7896b14956.png" class="math-inline" /> は帰納的。

- ベクトル空間の部分空間全体に包含関係を入れたもの：
  - 鎖の合併が再び部分空間になるので、帰納的。

__3.2 Zornの補題の典型的な使い方__

Zornの補題は、次のような「存在証明」に使われます。

- **基底の存在**：任意のベクトル空間は基底を持つ。
- **極大イデアルの存在**：単位的可換環には極大イデアルが存在する。
- **整列可能定理**：任意の集合には整列順序が存在する（選択公理と同値）。

**証明の流れ（概略）**：
1. 考えたい対象（例：線形独立な部分集合、真のイデアルなど）の集合 <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> を用意し、包含関係で半順序集合にする。
2. 任意の鎖 <img src="tmp/26a42c3438cde3a14157b1a39b5ac1f2.png" class="math-inline" /> に対して、その合併（または適切な極限）が <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> に属することを示し、<img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" /> が帰納的であることを確認する。
3. Zornの補題により、<img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> には極大元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> が存在する。
4. その <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> が求めるもの（基底、極大イデアルなど）であることを示す。

### 4. 選択公理との関係

- Zornの補題は、**選択公理（Axiom of Choice）と同値**です。
- すなわち、ZF公理系（選択公理を除いた通常の集合論）のもとで：
  - 選択公理 ⇒ Zornの補題
  - Zornの補題 ⇒ 選択公理
  の両方が証明できます。
- したがって、Zornの補題を用いる証明は、本質的に「選択公理を使っている」ことになります。

### 5. Zornの補題の重要性

Zornの補題は、**現代数学の多くの分野で「無限から有限的な構造を取り出す」ための基本的な道具**として、非常に重要です。

わざわざ補題として扱う理由は、大きく分けて次の3つです。

1. **選択公理と同値で、多くの定理の証明に必須だから**
2. **「極大元」や「基底」の存在を示すのに便利だから**
3. **無限集合の構造を調べる際の強力な武器だから**

順に説明します。

__1. 選択公理と同値で、多くの定理の証明に必須__

Zornの補題は、**選択公理（Axiom of Choice）と同値**な命題です。

- 選択公理：  
  「任意の集合族から、それぞれ1つずつ元を選ぶ関数（選択関数）が存在する」
- Zornの補題：  
  「（ある条件を満たす）半順序集合には極大元が存在する」

この同値性により、Zornの補題は**選択公理を使いたい場面で、より使いやすい形にしたもの**と見なせます。

**必要性**：  
選択公理そのものは抽象的で使いづらいことが多いですが、Zornの補題は

> 「半順序集合の条件」＋「任意の鎖に上界がある」

という、比較的扱いやすい形で与えられます。  
そのため、多くの定理の証明で「Zornの補題を使う」という形で選択公理が利用されます。

__2. 「極大元」や「基底」の存在を示すのに便利__

Zornの補題の典型的な使い方は、

> 「ある性質を満たすもの全体」を半順序集合とみなし、  
> その中に**極大元（それ以上大きくできない元）** が存在することを示す

というものです。

__例1：ベクトル空間の基底の存在__

- <img src="tmp/170cc9bd0fbfe8e1f3d73b7e9e6f4e60.png" class="math-inline" /> をベクトル空間とします。
- 「線形独立な部分集合全体」を考え、包含関係で半順序を入れます。
- 任意の鎖（全順序部分集合）には上界（和集合）が存在します。
- Zornの補題より、極大な線形独立集合が存在します。  
  これが**基底**です。

__例2：イデアルの極大性（極大イデアルの存在）__

- 環 <img src="tmp/a0f7a58538051aad071147a20c796777.png" class="math-inline" /> の、あるイデアルを含むイデアル全体を考えます。
- 包含関係で半順序を入れ、Zornの補題を適用すると、極大イデアルの存在が示せます。

**必要性**：  
無限次元のベクトル空間や一般の環では、基底や極大イデアルを具体的に構成するのは困難です。  
Zornの補題を使うと、「存在すること」だけを保証できます。

__3. 無限集合の構造を調べる際の強力な武器__

Zornの補題は、**無限集合の構造を調べる**ときに特に威力を発揮します。

__例3：整列可能定理__

- 任意の集合は、適当な順序を入れることで整列集合にできる（整列可能定理）。
- この定理は、Zornの補題（あるいは選択公理）と同値です。

__例4：代数的閉包の存在__

- 任意の体は、代数的閉包（すべての多項式が根を持つ拡大体）を持つ。
- この存在証明にZornの補題が使われます。

**必要性**：  
無限集合に対して「すべての～を満たすもの」や「極大な～」を直接構成するのは難しいことが多いです。  
Zornの補題は、**「極限操作」を通じて、そうした構造の存在を保証する**役割を果たします。

__4. なぜ「補題」なのか__

Zornの補題は、歴史的には

- 選択公理から導かれる命題として認識され
- 多くの定理の証明で「補助的な命題」として使われてきた

ため、「補題（lemma）」と呼ばれています。

しかし、その重要性は「定理」と呼ぶにふさわしいものです。  
実際、選択公理・整列可能定理・Zornの補題は**互いに同値**であり、どれか一つを公理として採用すれば、他は定理として導けます。



## 整列可能定理

以下では、**整列可能定理（Well-ordering theorem）** の数学的な定義と関連事項を説明します。

### 1. 整列可能定理の定義

**整列可能定理**は、次のように述べられます。

> **任意の集合は、ある整列順序（well-order）を持つ。**

もう少し詳しく書くと：

- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> を任意の集合とする。
- このとき、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の二項関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> で、次の条件を満たすものが存在する：
  1. <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は**全順序集合**である（任意の2元が比較可能）。
  2. <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は**整列集合**である：
     - 任意の空でない部分集合 <img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> は、<img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> に関する**最小元**を持つ。

このような <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> を、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**整列順序（well-order）** といいます。

### 2. 直感的な意味

- 通常の大小関係では、<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> や <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> は整列集合ではありません（正の部分に最小元がないなど）。
- しかし整列可能定理は、「**どんなに複雑な集合でも、うまく順序を定義すれば『一番小さい元・その次・その次…』と一直線に並べられる**」と主張します。
- この「うまく定義された順序」は、通常の大小関係とは異なり、**具体的に構成できるとは限らない**点が重要です。

### 3. 選択公理との関係

整列可能定理は、**選択公理（Axiom of Choice）と同値**な命題です。

- **選択公理**：任意の集合族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" />（各 <img src="tmp/b4cf280200862773bb25975b26b9248c.png" class="math-inline" /> は空でない）に対して、選択関数 <img src="tmp/be62cbf70adbd1b1759191ab450a3258.png" class="math-inline" />（<img src="tmp/274fa58d682e44ba86dc3e814c2b7797.png" class="math-inline" />）が存在する。
- **整列可能定理**：任意の集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> に対して、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の整列順序が存在する。

ZF公理系（選択公理を除いた通常の集合論）のもとで：

- 選択公理 ⇒ 整列可能定理
- 整列可能定理 ⇒ 選択公理

の両方が証明できます。  
したがって、「整列可能定理を用いる証明」は本質的に「選択公理を用いている」ことになります。

### 4. 順序数との関係

- 整列可能定理により、任意の集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> はある整列順序 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> を持ちます。
- 整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は、ある**順序数（ordinal number）** <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と順序同型になります。
- この <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> を、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**順序型（order type）** といい、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の「大きさ」を測る指標として使われます（基数の定義など）。

### 5. 具体例との対比

- <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" />：通常の大小関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> がそのまま整列順序です。
- <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />：通常の <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> は整列順序ではありませんが、例えば
  

<div class="math-display-container"><img src="tmp/74147bf032e6bafa945bb91e487c39da.png" class="math-display" /></div>


  という順序を入れると整列集合になります（最小元は <img src="tmp/b0782419bb8039fe8b6afc6b67925494.png" class="math-inline" />、その次は <img src="tmp/86f2ac6fe10c4df571d915e43a9d136a.png" class="math-inline" />、その次は <img src="tmp/28761683c454cb781fdb6aac6bd8a6df.png" class="math-inline" />、…）。
- <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" />：通常の <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> は整列順序ではありませんが、整列可能定理により「何らかの」整列順序が存在することは保証されます（ただし具体的な構成は選択公理に依存し、明示的には書けないことが多い）。

## 演習問題

### 問題

__問題1（定義の確認）__

(1) 半順序集合 <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" /> において、部分集合 <img src="tmp/3f34510ec61991757bec0d031992e0d1.png" class="math-inline" /> の**上界**と、元 <img src="tmp/e7fb43b3ae1ad2b729e48533f72bc3b6.png" class="math-inline" /> が**極大元**であることの定義を、論理式で書き下せ。

(2) 半順序集合 <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" /> が**帰納的（inductive）** であるとはどういうことか、定義を正確に述べよ。

(3) **Zornの補題**の主張を、帰納的順序と極大元を用いて述べよ。

(4) **整列可能定理**の主張を、整列順序（well-order）の定義を用いて述べよ。

__問題2（帰納的順序の具体例）__

次の半順序集合 <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" /> が帰納的であるかどうかを判定し、理由を簡潔に説明せよ。

(1) <img src="tmp/6a6c5538e4c354c9dc2df992e02fc7fb.png" class="math-inline" />（自然数のべき集合）、順序は包含関係 <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" />。

(2) <img src="tmp/262e494bd084b7731da8e46d72230b19.png" class="math-inline" />（実数全体）、順序は通常の大小関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" />。

(3) <img src="tmp/1219ce43027b12edaab5466dbbee7d60.png" class="math-inline" />、順序は包含関係 <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" />。

(4) <img src="tmp/28d8facf558211a307789fe8f962dea3.png" class="math-inline" />、順序は「<img src="tmp/a75236f5f6d2724bca30a489d8123612.png" class="math-inline" />」（各点ごとの大小）。

__問題3（Zornの補題の典型的な使い方）__

<img src="tmp/170cc9bd0fbfe8e1f3d73b7e9e6f4e60.png" class="math-inline" /> を体 <img src="tmp/2869f80330e3b768b3feec4cdf4c7583.png" class="math-inline" /> 上のベクトル空間とする。  
<img src="tmp/4bbb3638348e6ada287657e73a589b80.png" class="math-inline" /> とし、<img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> に包含関係 <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" /> で順序を入れる。

(1) <img src="tmp/ace52bf396d44d5ddcd70386cd3211c3.png" class="math-inline" /> が半順序集合であることを示せ。

(2) <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> の任意の鎖 <img src="tmp/26a42c3438cde3a14157b1a39b5ac1f2.png" class="math-inline" /> に対し、その合併 <img src="tmp/853ade47cfc0e7dbcede8ce461044bac.png" class="math-inline" /> が再び <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> に属する（すなわち線形独立である）ことを示し、<img src="tmp/ace52bf396d44d5ddcd70386cd3211c3.png" class="math-inline" /> が帰納的であることを示せ。

(3) Zornの補題を用いて、<img src="tmp/170cc9bd0fbfe8e1f3d73b7e9e6f4e60.png" class="math-inline" /> が基底を持つことを証明せよ（極大元が基底になることを示せ）。

__問題4（整列可能定理と選択公理）__

(1) 次の集合が、通常の大小関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> に関して整列集合であるかどうかを判定し、理由を述べよ。
- <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" />
- <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />
- <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" />
- <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" />

(2) 整列可能定理によれば、<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> にも何らかの整列順序が存在する。この事実は、選択公理とどのような関係にあるか、簡潔に説明せよ。

(3) 整列可能定理を用いて、「任意の集合はある順序数と順序同型になる」ことを説明せよ（順序数の定義は既知としてよい）。

__問題5（総合問題）__

(1) 選択公理・Zornの補題・整列可能定理の3つが互いに同値であることを、1〜2文でまとめよ。

(2) 次の命題が、選択公理（またはそれと同値な命題）を用いずに証明できるかどうか、理由とともに答えよ。
- 「任意のベクトル空間は基底を持つ。」
- 「任意の単位的可換環には極大イデアルが存在する。」
- 「<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> は整列順序を持つ。」

(3) 帰納的順序の定義において、「任意の鎖が上界を持つ」という条件を「任意の**有限**鎖が上界を持つ」に弱めた場合、Zornの補題は成り立つか。反例または証明の方針を述べよ。


### 解答

以下、各問題の解答です。

__問題1（定義の確認）__

__(1) 上界・極大元の定義__

半順序集合 <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" />、部分集合 <img src="tmp/3f34510ec61991757bec0d031992e0d1.png" class="math-inline" /> とする。

- **上界**：  
  元 <img src="tmp/3df7c5d08ea26c3346585533a07c62bd.png" class="math-inline" /> が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の上界であるとは、
  

<div class="math-display-container"><img src="tmp/b0db4ede41f20f6092d3de4aca7c3a7d.png" class="math-display" /></div>


  が成り立つことをいう。

- **極大元**：  
  元 <img src="tmp/e7fb43b3ae1ad2b729e48533f72bc3b6.png" class="math-inline" /> が極大元であるとは、
  

<div class="math-display-container"><img src="tmp/024205c7411327ba31f46d8d19734a56.png" class="math-display" /></div>


  が成り立つことをいう。  
  すなわち、「<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> より真に大きい元は存在しない」。

__(2) 帰納的順序の定義__

半順序集合 <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" /> が**帰納的**であるとは：

> 任意の全順序部分集合（鎖）<img src="tmp/26a42c3438cde3a14157b1a39b5ac1f2.png" class="math-inline" /> が、<img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> において上界を持つ。

すなわち、


<div class="math-display-container"><img src="tmp/327da3ccf61055977e9f2ae10ca22bd4.png" class="math-display" /></div>



__(3) Zornの補題の主張__

**Zornの補題**：

> 任意の帰納的半順序集合は、少なくとも1つの極大元を持つ。

論理式で書くと：
- <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" />：半順序集合
- <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" /> が帰納的ならば、
  

<div class="math-display-container"><img src="tmp/1aab416c2b340f92334b32e627a339ca.png" class="math-display" /></div>



__(4) 整列可能定理の主張__

**整列可能定理**：

> 任意の集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> は、ある整列順序（well-order）を持つ。

もう少し詳しく：
- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> を任意の集合とする。
- このとき、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の二項関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> で、
  1. <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は全順序集合（任意の2元が比較可能）
  2. <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は整列集合（任意の空でない部分集合が最小元を持つ）
  を満たすものが存在する。

__問題2（帰納的順序の具体例）__

__(1) <img src="tmp/c334c179a80e2c0d588a356ef384a11c.png" class="math-inline" />__

- **帰納的である**。
- 理由：任意の鎖（包含に関して全順序な部分集合族）<img src="tmp/f5c6e0fb532982f7ca85bd35de19b8b1.png" class="math-inline" /> に対し、その合併 <img src="tmp/853ade47cfc0e7dbcede8ce461044bac.png" class="math-inline" /> が上界になる。  
  - 各 <img src="tmp/9bc8744ef3a696866c64dbc1338e44d1.png" class="math-inline" /> について <img src="tmp/80d0c23a7833a200fd0b0bdf0a5cf49a.png" class="math-inline" /> なので、<img src="tmp/853ade47cfc0e7dbcede8ce461044bac.png" class="math-inline" /> は <img src="tmp/7da569b3f6d6a9ad1defdb55cccf0f5f.png" class="math-inline" /> の上界。
  - <img src="tmp/679a2c1b1068b29543baa3747709d943.png" class="math-inline" /> なので、上界は <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> 内に存在する。

__(2) <img src="tmp/ae7ea5f404149c105370b123dbe006af.png" class="math-inline" />（通常の大小）__

- **帰納的ではない**。
- 理由：例えば、鎖 <img src="tmp/33b06382e8623c76dd4c16dad3478a83.png" class="math-inline" />（自然数全体）を考えると、<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> 内に上界は存在しない（実数は上に有界でない）。  
  したがって、「任意の鎖が上界を持つ」は成り立たない。

__(3) <img src="tmp/98f21918a4000d70b88f2db1fc221422.png" class="math-inline" />__

- **帰納的ではない**。
- 理由：例えば、鎖
  

<div class="math-display-container"><img src="tmp/b701492487c6c2b1bdb3c74012ca2b5f.png" class="math-display" /></div>


  を考えると、その合併は <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> となり無限集合であるため、<img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> の元ではない。  
  <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> 内にこの鎖の上界は存在しない（有限集合しか許されないため）。

__(4) <img src="tmp/fb8b081fed9ae4e4da32b19f09b0e5fc.png" class="math-inline" />__

- **帰納的ではない**。
- 理由：例えば、鎖として
  

<div class="math-display-container"><img src="tmp/40842a449e220ec13ed640578f3d49e8.png" class="math-display" /></div>


  を考えると、各 <img src="tmp/4a968a7d94d58656bc09fd3ea17b1584.png" class="math-inline" /> は有限個の点以外で0なので <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> の元。  
  しかし、この鎖の「上限」となる関数は
  

<div class="math-display-container"><img src="tmp/d917b2f0f2597044ec954e92cdd5df47.png" class="math-display" /></div>


  であり、これは有限個の点以外で0ではないので <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> に属さない。  
  よって、この鎖の上界は <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> 内に存在せず、帰納的ではない。

__問題3（Zornの補題の典型的な使い方）__

<img src="tmp/170cc9bd0fbfe8e1f3d73b7e9e6f4e60.png" class="math-inline" />：体 <img src="tmp/2869f80330e3b768b3feec4cdf4c7583.png" class="math-inline" /> 上のベクトル空間  
<img src="tmp/530472e63fee3a67b30f74f787e45ff6.png" class="math-inline" />、順序は包含関係 <img src="tmp/aa94f9e6e470d1f422ddeb565859e87f.png" class="math-inline" />。

__(1) <img src="tmp/ace52bf396d44d5ddcd70386cd3211c3.png" class="math-inline" /> が半順序集合であること__

- **反射律**：任意の <img src="tmp/06e46fd07a475adf77f1215a5c3addc7.png" class="math-inline" /> について <img src="tmp/964bbf3e7f2b24379ffea6f2e856b311.png" class="math-inline" /> は明らか。
- **推移律**：<img src="tmp/f4ef2c25253e9b38307104fc553d4fd7.png" class="math-inline" /> かつ <img src="tmp/6ba648193c318f6a232c1c778e4a3d59.png" class="math-inline" /> なら <img src="tmp/d2a5a5b23eab947622d20fc37135aa0b.png" class="math-inline" />。
- **反対称律**：<img src="tmp/f4ef2c25253e9b38307104fc553d4fd7.png" class="math-inline" /> かつ <img src="tmp/3cd90584ecf1c9eb7be5c4f5737e4e31.png" class="math-inline" /> なら <img src="tmp/ebee15a454d654b00a8d56e152e2b801.png" class="math-inline" />（集合として等しい）。

よって <img src="tmp/ace52bf396d44d5ddcd70386cd3211c3.png" class="math-inline" /> は半順序集合。

__(2) <img src="tmp/ace52bf396d44d5ddcd70386cd3211c3.png" class="math-inline" /> が帰納的であること__

<img src="tmp/26a42c3438cde3a14157b1a39b5ac1f2.png" class="math-inline" /> を任意の鎖（包含に関して全順序な部分集合族）とする。

- 合併 <img src="tmp/4cb0ba55f1de42b782316bba9c9ef4d3.png" class="math-inline" /> を考える。
- <img src="tmp/2321eadfdfcad762b559d2c1971bf109.png" class="math-inline" /> が線形独立であることを示せば、<img src="tmp/624e58f8180a7a4fb867913bd75ccab7.png" class="math-inline" /> かつ <img src="tmp/2321eadfdfcad762b559d2c1971bf109.png" class="math-inline" /> が <img src="tmp/7da569b3f6d6a9ad1defdb55cccf0f5f.png" class="math-inline" /> の上界になる。

**線形独立性の確認**：  
有限個のベクトル <img src="tmp/d07d817dea00536b6a58080b9ab2bcd7.png" class="math-inline" /> とスカラー <img src="tmp/7cdf7c100ad74fe4adebfdfa7bccb4ad.png" class="math-inline" /> に対し、


<div class="math-display-container"><img src="tmp/4a71a0d049f9d811d410e1a3cb12d57d.png" class="math-display" /></div>


とする。各 <img src="tmp/b8cbdf420375aef3200b771ee44539c5.png" class="math-inline" /> はある <img src="tmp/aea282a568ddf151ddc2e5f5d445adbc.png" class="math-inline" /> に属する。<img src="tmp/7da569b3f6d6a9ad1defdb55cccf0f5f.png" class="math-inline" /> は鎖なので、これらの <img src="tmp/3399023c30bd75d8ba8393f0dd479b00.png" class="math-inline" /> の中に包含関係で最大のもの <img src="tmp/6acee9b15f70e6d55c0584f9aeb79ae0.png" class="math-inline" /> が存在する（有限個だから）。  
すると <img src="tmp/af596452f0f3bc8622a0ae1110669649.png" class="math-inline" /> であり、<img src="tmp/6acee9b15f70e6d55c0584f9aeb79ae0.png" class="math-inline" /> は線形独立なので <img src="tmp/4860886ec1551d1e7549fcce348e41a5.png" class="math-inline" />。  
よって <img src="tmp/2321eadfdfcad762b559d2c1971bf109.png" class="math-inline" /> は線形独立。

- 任意の <img src="tmp/986253ec940f9f860d92ea356fb761be.png" class="math-inline" /> について <img src="tmp/d2a5a5b23eab947622d20fc37135aa0b.png" class="math-inline" /> なので、<img src="tmp/2321eadfdfcad762b559d2c1971bf109.png" class="math-inline" /> は <img src="tmp/7da569b3f6d6a9ad1defdb55cccf0f5f.png" class="math-inline" /> の上界。
- <img src="tmp/624e58f8180a7a4fb867913bd75ccab7.png" class="math-inline" /> なので、<img src="tmp/ace52bf396d44d5ddcd70386cd3211c3.png" class="math-inline" /> は帰納的。

__(3) Zornの補題による基底の存在証明__

- (1),(2) より <img src="tmp/ace52bf396d44d5ddcd70386cd3211c3.png" class="math-inline" /> は帰納的半順序集合。
- Zornの補題により、<img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> には極大元 <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> が存在する。
- <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> が <img src="tmp/170cc9bd0fbfe8e1f3d73b7e9e6f4e60.png" class="math-inline" /> の基底であることを示す。

**線形独立性**：<img src="tmp/869e0df5d8b65988fa195872e5716dc0.png" class="math-inline" /> より、<img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は線形独立。

**生成**：<img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> が <img src="tmp/170cc9bd0fbfe8e1f3d73b7e9e6f4e60.png" class="math-inline" /> を生成しないと仮定する。  
すると、ある <img src="tmp/663f409d895c301bece5ec447a6046c0.png" class="math-inline" /> が <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の線形結合で書けない。このとき <img src="tmp/d6e9d037457a532ff77f502e7e9d64ef.png" class="math-inline" /> は線形独立（そうでないと <img src="tmp/c79a25dfc13c85e27258fabd4a355169.png" class="math-inline" /> が <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の線形結合で書けることになる）。  
よって <img src="tmp/fc3941601945ea06f738abc281e9c57a.png" class="math-inline" /> となり、<img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> が極大元であることに矛盾。  
したがって <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は <img src="tmp/170cc9bd0fbfe8e1f3d73b7e9e6f4e60.png" class="math-inline" /> を生成する。

以上より、<img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は <img src="tmp/170cc9bd0fbfe8e1f3d73b7e9e6f4e60.png" class="math-inline" /> の基底である。

__問題4（整列可能定理と選択公理）__

__(1) 通常の <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> での整列集合かどうか__

- <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" />：**整列集合**  
  任意の空でない部分集合は最小元を持つ（自然数の整列性）。

- <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />：**整列集合ではない**  
  例：<img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> 自身に最小元がない。

- <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" />：**整列集合ではない**  
  例：<img src="tmp/d93eb511d5ca0f8f6fad2b177fbc857b.png" class="math-inline" /> に最小元がない。

- <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" />：**整列集合ではない**  
  例：<img src="tmp/048f0412f7186b62fb74c701235f3359.png" class="math-inline" /> に最小元がない。

__(2) 整列可能定理と選択公理の関係__

- 整列可能定理は、**選択公理と同値**な命題である。
- ZF公理系（選択公理を除いた通常の集合論）のもとで：
  - 選択公理 ⇒ 整列可能定理
  - 整列可能定理 ⇒ 選択公理
  の両方が証明できる。
- したがって、「<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> に整列順序が存在する」という事実は、選択公理を用いなければ証明できない。

__(3) 「任意の集合はある順序数と順序同型になる」ことの説明__

- 整列可能定理により、任意の集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> にはある整列順序 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> が存在する。
- 整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は、ある**順序数** <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と順序同型になる（順序数の定義：整列集合の順序同型類）。
- したがって、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> は順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と順序同型である。

__問題5（総合問題）__

__(1) 選択公理・Zornの補題・整列可能定理の同値性__

- 選択公理・Zornの補題・整列可能定理は、ZF公理系のもとで**互いに同値**な命題である。
- すなわち、いずれか1つを仮定すれば残り2つが証明でき、逆も成り立つ。

__(2) 各命題が選択公理なしで証明できるか__

- 「任意のベクトル空間は基底を持つ。」  
  → **選択公理（または同値な命題）なしでは証明できない**。  
  実際、ZFだけでは「基底を持たないベクトル空間が存在し得る」ことが知られている。

- 「任意の単位的可換環には極大イデアルが存在する。」  
  → **選択公理（または同値な命題）なしでは証明できない**。  
  これはZornの補題の典型的な応用であり、選択公理と同値な命題に依存する。

- 「<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> は整列順序を持つ。」  
  → **選択公理（または同値な命題）なしでは証明できない**。  
  整列可能定理そのもの（あるいはそれと同値な命題）が必要。

__(3) 「任意の有限鎖が上界を持つ」場合のZornの補題__

- 条件を「任意の**有限**鎖が上界を持つ」に弱めると、Zornの補題は**成り立たない**。
- 反例のイメージ：
  - 例えば、自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> に通常の大小を入れた半順序集合を考える。
  - 任意の有限鎖（有限の全順序部分集合）は上界を持つ（その鎖の最大元自身が上界）。
  - しかし、<img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> 自体には極大元は存在しない。
- したがって、「有限鎖だけ」を考えた条件では、極大元の存在を保証できない。


<div style="page-break-before:always"></div>



# 順序数

## 順序数

順序数（ordinal number）は、**整列集合（well-ordered set）の「順序型」を抽象化したもの**で、集合論・数学基礎論で非常に重要な概念です。
以下、順を追って説明します。

### 1. 準備：整列集合と順序同型

__整列集合（well-ordered set）__

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の二項関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> が次の条件を満たすとき、<img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> を**整列集合**といいます。

1. **順序**：任意の <img src="tmp/aafcdd2e8da6043624a2be58ad91ff0e.png" class="math-inline" /> について <img src="tmp/d6813ad06e5de914d8e58ace1ab5ed25.png" class="math-inline" /> または <img src="tmp/708596235cba73cab553a9bb5933227c.png" class="math-inline" />。
2. **整列性**：任意の空でない部分集合 <img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> が**最小元**を持つ。

例：

- <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" />：自然数全体は整列集合（最小元は 1）。
- <img src="tmp/5608f64ad5726b0cec3f4b4905f05ef3.png" class="math-inline" />：整数全体は整列集合ではない（負の数全体に最小元がない）。

順序数 <img src="tmp/18991571b007a883e99c3893b3740bdf.png" class="math-inline" /> を数直線上の点として並べ、「順序数＝整列した番号」が(有限)順序数です。

<img src="image/9_ordered_number/1782853709544.png" alt="" width="500" style="display: block; margin: 0 auto;">

__順序同型（order isomorphism）__

2つの整列集合 <img src="tmp/b809c81579da26b8a4a8a36565a8082f.png" class="math-inline" /> と <img src="tmp/98b779eceb779a56065b921e642dfbca.png" class="math-inline" /> が**順序同型**であるとは、全単射 <img src="tmp/c0f904752233741fed57139839fdf3ec.png" class="math-inline" /> で



<div class="math-display-container"><img src="tmp/9c96c2c33ab8bd44e3936f70573f024a.png" class="math-display" /></div>



を満たすものが存在することをいいます。
このとき、2つの整列集合は「順序構造まで含めて同じ形」とみなせます。

順序数＝「整列順序の同型類（型）」 です。

2つの整列集合 A, B がともに順序数 3 の型を持つことを示し、順序同型写像（a↔x, b↔y, c↔z）の存在を矢印で表現します。
という順序数のイメージを伝えると以下のようになります。

<img src="image/9_ordered_number/1782853189663.png" alt="" width="500" style="display: block; margin: 0 auto;">



### 2. 順序数の数学的な定義（フォン・ノイマンの構成）

順序数は、**整列集合の順序同型類**として定義されますが、集合論ではより具体的に「フォン・ノイマン順序数」として実現します。

__フォン・ノイマン順序数の定義__

集合 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が**順序数**であるとは、次の2条件を満たすことをいいます。

1. **推移的（transitive）** ：任意の <img src="tmp/d78c489d602325dcbc17014eef6719c2.png" class="math-inline" /> について <img src="tmp/464347a1a6670d9a4d39423edaa744c9.png" class="math-inline" />。（<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> の元は <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> の部分集合でもある）
2. **<img src="tmp/703433fde77b69514cee13f320ecf601.png" class="math-inline" /> に関して整列集合** ：
   <img src="tmp/7a1c574eadd58f5dcda2010e1ba7e2ec.png" class="math-inline" /> が整列集合である。
   ここで、<img src="tmp/b8b1690b73a937e2b529a90a8261a8bf.png" class="math-inline" /> を「<img src="tmp/24c9e6d4b22616cc4d909e1612426371.png" class="math-inline" />」と解釈します。

この定義のもとで、次のように順序数を構成できます。

- <img src="tmp/e74407ba198fa7e591b2603f09b67ec7.png" class="math-inline" />（空集合）
- <img src="tmp/afd2028a3fa0be36133231ceb755a765.png" class="math-inline" />
- <img src="tmp/e0ac09f4ab00d45b041c9cd3456cde1e.png" class="math-inline" />
- <img src="tmp/8a9242ab1f04f5f26272bc2637fd9afa.png" class="math-inline" />
- <img src="tmp/92f3224c3f467f219ee55396cf68c871.png" class="math-inline" />
- <img src="tmp/af74e1e4b747b92e0cb97e90219d659f.png" class="math-inline" />（自然数全体）
- <img src="tmp/51c164380956106791af0a99f5f4b1c8.png" class="math-inline" />
- <img src="tmp/041d8c351e2b02c73eaf9d450864df05.png" class="math-inline" />
- <img src="tmp/92f3224c3f467f219ee55396cf68c871.png" class="math-inline" />

このように、**各順序数は、それより小さいすべての順序数の集合**として定義されます。

### 3. 順序数の基本性質

__(1) 順序数の大小関係__

順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> に対し、



<div class="math-display-container"><img src="tmp/0213ed2a3d0d3b05efce44630a708f42.png" class="math-display" /></div>



と定義します。
この順序は**全順序**であり、任意の2つの順序数は比較可能です。

また、順序数全体のクラスは**整列クラス**（proper class）ですが、任意の順序数の集合は整列集合になります。

__(2) 後続順序数と極限順序数__

- **後続順序数（successor ordinal）** ：ある順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が存在して <img src="tmp/9e2bc60f23b538cbc380ff47951e0741.png" class="math-inline" /> と書けるとき、<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> を後続順序数といいます。

  - 例：<img src="tmp/c8ec12cc0d676429b26b61085f34855c.png" class="math-inline" />
- **極限順序数（limit ordinal）** ：後続順序数でなく、かつ <img src="tmp/b0782419bb8039fe8b6afc6b67925494.png" class="math-inline" /> でもない順序数。

  - 例：<img src="tmp/490433c10e92dfe076f0174b338a2c3b.png" class="math-inline" />

__(3) 超限帰納法（transfinite induction）__

順序数を用いると、**無限段階の帰納法**が可能になります。

> 順序数全体にわたる命題 <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> が次の2条件を満たすなら、すべての順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> が成り立つ。
>
> 1. **基底**：<img src="tmp/b27dacc5676ded5b21aa21ada36bd16a.png" class="math-inline" /> が成り立つ。
> 2. **後続段階**：任意の <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について、<img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> ならば <img src="tmp/b26b09e68079461ba08edbb44f49871e.png" class="math-inline" />。
> 3. **極限段階**：任意の極限順序数 <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> について、すべての <img src="tmp/a3dcd668e156e5eab97d6545beb2a0d8.png" class="math-inline" /> で <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> が成り立つなら <img src="tmp/d2d07d491ce308b6749dfc422f75d6b4.png" class="math-inline" /> も成り立つ。

__(4) 超限再帰（transfinite recursion）__

順序数に沿って、関数や集合を**超限再帰的に定義**できます。

例：累積階層（cumulative hierarchy）

- <img src="tmp/c449a50701441a48cf6d3b2848ea7fab.png" class="math-inline" />
- <img src="tmp/f0d2e27d605b08eb961e98c4655f4dfa.png" class="math-inline" />
- <img src="tmp/d8ad7bd408b443cbd81116e5d02c7573.png" class="math-inline" />（<img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> は極限順序数）

これにより、集合全体の宇宙 <img src="tmp/e8dc511d502f1fc0280b4a5be4aab030.png" class="math-inline" /> が定義されます（累積階層）。

### 4. 順序数と基数の関係

__基数（cardinal number）__

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の**基数**とは、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> と対等（全単射が存在する）な順序数の**最小のもの**です。記号では <img src="tmp/7e092df0fd877e5cdc6898eb0a10c1ba.png" class="math-inline" /> と書きます。

- 有限集合の基数：<img src="tmp/8677a64c6e78ad82edd3735f5a28ff84.png" class="math-inline" />（自然数）
- 可算無限集合の基数：<img src="tmp/95c7bb5202184ec26cfae308322b9605.png" class="math-inline" />
- 連続体濃度：<img src="tmp/fc0341960ed236754f7616f4069d9c84.png" class="math-inline" />

__順序数としての基数__

- 基数は特別な順序数です（「その順序型を持つ整列集合の中で最小のもの」）。
- 例：
  - <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は最小の無限順序数であり、同時に可算基数 <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> でもあります。
  - <img src="tmp/da225e47f266585ccea845016cb6c238.png" class="math-inline" /> は最小の非可算順序数であり、同時に基数 <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" /> です。

一般に、**基数は「<img src="tmp/703433fde77b69514cee13f320ecf601.png" class="math-inline" /> に関して始片（initial segment）となる」性質を持つ順序数**として特徴づけられます。

### 5. なぜ順序数が必要なのか

1. **無限の「長さ」や「段階」を測るため**自然数は有限の個数を数えますが、順序数は「無限に続く列」の長さや段階を測るために使われます。
2. **超限帰納法・超限再帰の基礎**数学の証明や定義を「無限回」繰り返すための枠組みを提供します。
3. **集合の宇宙の構造を記述するため**累積階層 <img src="tmp/21f89b083ba97a37b2b9ab93a9d4a430.png" class="math-inline" /> は、集合の宇宙がどのように「段階的に」構成されるかを示します。
4. **基数理論の土台**
   無限集合の「大きさ」を比較する基数は、順序数の特別なクラスとして定義されます。

### 6. 順序数の具体例

順序数の具体例を、**有限順序数**と**無限順序数**に分けて簡潔に説明します。

__1. 有限順序数（自然数）__

- **0**：空集合 ∅ の順序数  
  - 整列集合：∅（空集合）
- **1**：1元集合 {0} の順序数  
  - 整列集合：{0}（0のみ）
- **2**：2元集合 {0,1} の順序数  
  - 整列集合：{0,1}（0<1）
- **3**：3元集合 {0,1,2} の順序数  
  - 整列集合：{0,1,2}（0<1<2）
- **n**：n元集合 {0,1,…,n-1} の順序数  
  - 整列集合：{0,1,…,n-1}（通常の順序）

これらは、**自然数そのもの**と同一視されます。

__2. 最初の無限順序数 ω__

- **ω**：自然数全体 ℕ = {0,1,2,…} の順序数  
  - 整列集合：ℕ（0<1<2<…）
  - 性質：有限順序数すべてより大きい最小の順序数

__3. ω より大きい順序数の例__

- **ω+1**：ℕ ∪ {ω} に順序 0<1<2<…<ω を入れた整列集合の順序数
- **ω+2**：ℕ ∪ {ω, ω+1} に 0<1<2<…<ω<ω+1 を入れた整列集合の順序数
- **ω+ω = ω·2**：  
  - 整列集合：{0,1,2,…} ∪ {ω, ω+1, ω+2, …} に  
    0<1<2<…<ω<ω+1<ω+2<… を入れたもの



## 順序数の大小

順序数の「大小」は、**集合の所属関係 <img src="tmp/703433fde77b69514cee13f320ecf601.png" class="math-inline" />** を使って定義されます。
以下、順序数の大小の定義と、その特徴（全順序性・整列性・順序同型との関係など）を順に説明します。

### 1. 順序数の大小の定義

__フォン・ノイマン順序数の定義（再確認）__

集合 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が**順序数**であるとは：

1. **推移的**：<img src="tmp/4c61f40b06ad41a41b6377a7b51f0d5b.png" class="math-inline" />
2. **<img src="tmp/703433fde77b69514cee13f320ecf601.png" class="math-inline" /> に関して整列集合**：<img src="tmp/7a1c574eadd58f5dcda2010e1ba7e2ec.png" class="math-inline" /> が整列集合

を満たすことです。

__大小関係の定義__

順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> に対し、**大小関係**を次のように定義します。

- <img src="tmp/ade1f69048f6124b4357341bee821f8e.png" class="math-inline" />
- <img src="tmp/1d3976e38326ee65bcc9a5cee3dc5974.png" class="math-inline" /> または <img src="tmp/37692c5fcbf9c794c39abb122b1dbafd.png" class="math-inline" />

つまり、**「小さい順序数は、大きい順序数の要素として含まれる」** という関係です。

例：

- <img src="tmp/c8409bf011dba8335cb0b1d24a0572da.png" class="math-inline" /> なので <img src="tmp/a1d4cebd5d11768151238148e7990df7.png" class="math-inline" />、よって <img src="tmp/45032a0e90e989564b8b4665e442904c.png" class="math-inline" />
- <img src="tmp/cd2ccfdd552f299c570394d2d061b587.png" class="math-inline" /> なので <img src="tmp/b989b62d5f401d735d65281140d1a87d.png" class="math-inline" />、よって <img src="tmp/baa8096b81b0f3fc2ee11c601440acf9.png" class="math-inline" />
- <img src="tmp/336bfd470c99b285186264ff555d0a47.png" class="math-inline" /> なので、任意の自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> について <img src="tmp/88548ba1210d4a7e601e642df8cc2583.png" class="math-inline" />、よって <img src="tmp/6263c987771690e784009fcdb7d0955c.png" class="math-inline" />
- <img src="tmp/0a69a4db7713fdaf8d23a0652684bca9.png" class="math-inline" /> なので <img src="tmp/649e1fb325b555bd44739b1e2a8bfad2.png" class="math-inline" />、よって <img src="tmp/fc12fce9e7de74df9aecfaa3f918858e.png" class="math-inline" />

### 2. 大小関係の基本性質

__(1) 全順序性（任意の2つの順序数は比較可能）__

**定理**：任意の2つの順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> について、次のいずれか1つだけが成り立つ。

- <img src="tmp/29b7b525deaf992bcdd385ebe7253468.png" class="math-inline" />
- <img src="tmp/37692c5fcbf9c794c39abb122b1dbafd.png" class="math-inline" />
- <img src="tmp/b4d58078d6159ee352ef89f89ce6e36c.png" class="math-inline" />

すなわち、順序数のクラスは**全順序クラス**です。

証明の概略：

- 順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> に対し、<img src="tmp/ae6ded84d46cd69814ecdc0763afc2b9.png" class="math-inline" /> はどちらかの**始片（initial segment）** になる。
- 整列集合の比較定理により、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> は順序同型か、一方が他方の真の始片になる。
- 順序数は「自分より小さい順序数全体」として定義されているので、順序同型なら等しい。
- したがって、<img src="tmp/24f830963b96201ad64b529a12de7d9c.png" class="math-inline" />（<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の始片）か、その逆か、あるいは等しいかのいずれかになる。

__(2) 整列性（順序数の任意の集合は整列集合）__

**定理**：順序数の任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は、<img src="tmp/703433fde77b69514cee13f320ecf601.png" class="math-inline" /> に関して整列集合である。

- すなわち、任意の空でない <img src="tmp/7f40e5ee3434c624a4a0ad1969cb098d.png" class="math-inline" /> には**最小元**が存在する。
- 最小元は「<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の元の中で、どの元よりも小さい（＝どの元の要素にもなっていない）順序数」として特徴づけられます。

例：

- <img src="tmp/f13d25e6f75d7fc9fbfabe1bf190f010.png" class="math-inline" /> の最小元は <img src="tmp/748d014a08c85df236ffbfbc57dfcde3.png" class="math-inline" />（<img src="tmp/1662ac47ebb542d9f0f2e4cf6f77303e.png" class="math-inline" /> かつ <img src="tmp/4594aca86bcfb491a0b4ac3075b9ff2a.png" class="math-inline" /> なので、<img src="tmp/748d014a08c85df236ffbfbc57dfcde3.png" class="math-inline" /> は <img src="tmp/b3bd959441ae0a45b4744e146d262827.png" class="math-inline" /> より小さい）
- <img src="tmp/637a42d21214b6dfe5b45e3a34f8ca42.png" class="math-inline" /> の最小元は <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" />

__(3) 順序数は「自分より小さい順序数全体」__

任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について、



<div class="math-display-container"><img src="tmp/f22c0ec6d7fe1c2bf004512a794c13bb.png" class="math-display" /></div>



が成り立ちます。

- これはフォン・ノイマン順序数の定義そのものです。
- したがって、順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は「<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> 未満のすべての順序数の集合」として一意に決まります。

### 3. 順序数の演算と大小の関係

順序数には、和・積・べき乗などの演算が定義され、それらは大小関係と整合的です。

__(1) 順序数の和 <img src="tmp/523ee0a625bb9be4d02ba6cf448d12be.png" class="math-inline" />__

直観的には、「<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> のコピーの後ろに <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> のコピーを並べた」整列集合の順序型です。

- 例：<img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> は「自然数の列の後に1つ要素を追加した」順序型。
- 大小関係：<img src="tmp/1b4d47fd3f125990e39e0c3b79d75f2d.png" class="math-inline" />、また <img src="tmp/dbfef2bb46c9a82a4feb3413ab8e8836.png" class="math-inline" /> なら <img src="tmp/61a21e7fa1c270d0ba9a56d1978e727f.png" class="math-inline" />。

__(2) 順序数の積 <img src="tmp/42585268f08ccf21650310fe301440ba.png" class="math-inline" />__

「長さ <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の列の各要素を <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> で置き換えた」順序型です。

- 例：<img src="tmp/06c09cb0fc78c022cfd655b4500164dc.png" class="math-inline" />（自然数の列を2つ並べたもの）
- 大小関係：<img src="tmp/57873d8ff4c0547775923f4b1220a27f.png" class="math-inline" /> なら <img src="tmp/7bd5458f616222d2a1ee8a27dbd928b8.png" class="math-inline" />。

__(3) 順序数のべき乗 <img src="tmp/a4c0814d85b3414f4b1dedbed502d5fb.png" class="math-inline" />__

「長さ <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の列で、各項が <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> 未満の順序数であるような列」を辞書式順序で並べた順序型です。

- 例：<img src="tmp/fd008f3c43680f61586487c6a823220f.png" class="math-inline" /> は「有限列の自然数」を辞書式順序で並べたものの順序型。
- 大小関係：<img src="tmp/4c8f906bfa13258a13c95800f912f22c.png" class="math-inline" /> なら <img src="tmp/317ecdd9547deec3ee8e712ce95b5e51.png" class="math-inline" />（十分大きい <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> に対して）。

### 4. 順序数の大小と整列集合の順序型

__整列集合の順序型としての順序数__

任意の整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は、ある順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と**順序同型**になります。この <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> を <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の**順序型（order type）** といいます。

- 例：
  - <img src="tmp/3ffccda9fa2605e1c6e581bdb2d9bb03.png" class="math-inline" /> の順序型は <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />
  - <img src="tmp/4642d4762ecf3786ee3433dacc3e27aa.png" class="math-inline" /> の順序型は <img src="tmp/04e7c4e120880e26f5110d0d71495fab.png" class="math-inline" />
  - <img src="tmp/8d8fbd746e5bfe292af3e013ec0097cc.png" class="math-inline" /> の順序型は <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" />

__大小関係と順序型の比較__

2つの整列集合 <img src="tmp/52a998ff58794696b45fd3bf8b1d755b.png" class="math-inline" /> の順序型を <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> とすると：

- <img src="tmp/29b7b525deaf992bcdd385ebe7253468.png" class="math-inline" /> であることと、<img src="tmp/b809c81579da26b8a4a8a36565a8082f.png" class="math-inline" /> が <img src="tmp/98b779eceb779a56065b921e642dfbca.png" class="math-inline" /> の**真の始片**と順序同型であることは同値です。
- つまり、順序数の大小は「一方が他方の真の前方部分として埋め込めるかどうか」を反映しています。

### 5. 順序数のクラスの性質

__順序数全体は真のクラス（proper class）__

- 順序数全体の集まり <img src="tmp/24df8a72caec0be7051202ec37ad58da.png" class="math-inline" /> は集合にはなりません（**Burali-Fortiのパラドックス**）。
- したがって、「すべての順序数からなる集合」は存在しませんが、任意の順序数の集合は整列集合になります。

__順序数の上限（supremum）__

順序数の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に対し、その**上限**は



<div class="math-display-container"><img src="tmp/e5272e43be80d35061ffaf7e0c05a889.png" class="math-display" /></div>



で定義されます。

- <img src="tmp/f179083ab7e8aba75cff85adb7173f92.png" class="math-inline" /> は「<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> のどの元よりも小さくない最小の順序数」です。
- 例：
  - <img src="tmp/af317f02342c68588b3956c8a6b4543e.png" class="math-inline" />
  - <img src="tmp/869cb92427811c922fe5538131e97216.png" class="math-inline" />

## 最小値原理

代表的な2つの意味（整列性による最小値原理、自然数の最小値原理）と、その特徴を説明します。

### 1. 整列集合の最小値原理（well-ordering principle）

__定義__

集合 <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 上の二項関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> が**整列順序（well-order）** であるとは、次の条件を満たすことをいいます。

1. <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> は**全順序集合**（任意の2元が比較可能）。
2. **最小値原理**：任意の空でない部分集合 <img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> が**最小元**を持つ。
   すなわち、
   

<div class="math-display-container"><img src="tmp/b50c1e03d6d9f776d2525063a323a3a3.png" class="math-display" /></div>



この2番目の条件を「**最小値原理**」と呼ぶことがあります。

__特徴__

- **整列集合の定義そのもの**：最小値原理は、整列集合を特徴づける核心的な性質です。
- **超限帰納法の基礎**：整列集合上の命題 <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> を証明する際、「<img src="tmp/bc8aea6a9c97ea0e2af3b2792d34136f.png" class="math-inline" /> がすべての <img src="tmp/01600b17387d649e248e701d68af7633.png" class="math-inline" /> で成り立つなら <img src="tmp/0825d55354d62b227ddfbf50fc31b601.png" class="math-inline" /> も成り立つ」という形の**超限帰納法**が使えます。これは最小値原理（＝空でない反例集合があればその最小元が存在する）に基づいています。
- **順序数との関係**：
  任意の整列集合は、ある順序数と順序同型になります。順序数は「自分より小さい順序数全体の集合」として定義され、最小値原理を満たします。

### 2. 自然数の最小値原理（well-ordering principle of natural numbers）

__定義__

自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" />（または正の整数全体 <img src="tmp/0c090c70bf691dad9c28c411771ea2b7.png" class="math-inline" />）について、次の性質を**自然数の最小値原理**といいます。

> 任意の空でない自然数の部分集合は、最小元を持つ。

論理式で書くと：



<div class="math-display-container"><img src="tmp/7a10c932febd737955065718ae11cb36.png" class="math-display" /></div>



__特徴__

- **数学的帰納法と同値**：自然数の最小値原理は、**数学的帰納法の原理**と同値です。

  - 最小値原理 ⇒ 数学的帰納法：命題 <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> が <img src="tmp/ac530716190fa3ec148b9fc37d3fc192.png" class="math-inline" /> で成り立ち、かつ「<img src="tmp/ce4770d648c64b583ecb3b6daa2bd4b9.png" class="math-inline" /> がすべての <img src="tmp/da5870d79d8d240b90ae101c82490969.png" class="math-inline" /> で成り立つなら <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> も成り立つ」と仮定する。<img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> が偽になる <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> の集合が空でないとすると、その最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> が存在するが、<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> より小さいすべての自然数では <img src="tmp/984e139f0b00ce535d63d7d9e1fbb19a.png" class="math-inline" /> が真なので仮定により <img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> も真となり矛盾。よってそのような <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> は存在せず、すべての <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> で <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> が真。
  - 数学的帰納法 ⇒ 最小値原理：
    空でない自然数の部分集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が最小元を持たないと仮定する。
    <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" />：「<img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の元ではない」とおくと、帰納法によりすべての自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> について <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> が真となり <img src="tmp/00b560c16807a5ea1d0107e6f8086555.png" class="math-inline" /> となるが、これは矛盾。
- **整数論・組合せ論での基本的ツール**：

  - ディオファントス方程式の解の存在証明（「解が存在すれば最小解が存在する」として矛盾を導く）
  - グラフ理論や組合せ論での「極小反例」を用いた証明
    など、**最小の反例を取って矛盾を導く**証明手法の根拠になります。
- **選択公理とは独立**：
  自然数の最小値原理は、ZF公理系（選択公理なし）でも成り立ちます。
  一方、「任意の集合が整列順序を持つ」という**整列可能定理**は選択公理と同値であり、より強い主張です。

### 3. 一般の半順序集合における「極小元の存在」としての最小値原理

半順序集合 <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" /> において、次の性質を「最小値原理」と呼ぶこともあります。

> 任意の空でない部分集合 <img src="tmp/3f34510ec61991757bec0d031992e0d1.png" class="math-inline" /> が**極小元**を持つ。

ここで極小元とは：

- <img src="tmp/9507e0235f1b54262f27fa314e741f93.png" class="math-inline" /> が極小元であるとは、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の中に <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> より真に小さい元が存在しないこと：
  

<div class="math-display-container"><img src="tmp/27e9abdba1c492ae5fba6fb78504ef9e.png" class="math-display" /></div>



__特徴__

- **整列集合では最小元＝極小元**ですが、一般の半順序集合では異なります。
- この形の「極小元の存在」は、**Zornの補題**や**極大原理**と関連します。
  - 例：ベクトル空間の基底の存在証明では、「線形独立な集合全体」の半順序集合が極大元（＝基底）を持つことを示しますが、これは「極小元の存在」の双対版です。

### 4. 最小値原理の応用例

__(1) 整除関係での最小値原理__

正の整数全体 <img src="tmp/0c090c70bf691dad9c28c411771ea2b7.png" class="math-inline" /> に整除関係 <img src="tmp/710249df7ba4bebc84364a10b6e0ebe6.png" class="math-inline" /> で順序を入れると、これは整列順序ではありませんが、**各数の約数からなる集合**には最小値原理が成り立ちます。

- 任意の正の整数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> に対し、その正の約数の集合は最小元 <img src="tmp/28761683c454cb781fdb6aac6bd8a6df.png" class="math-inline" /> を持ちます。
- これは自然数の最小値原理の応用です。

__(2) 多項式環のイデアルにおける極小条件__

ネーター環（Noetherian ring）では、**イデアルの昇鎖条件**が成り立ち、これは「イデアルの集合が極小条件を満たす」ことと同値です。

- 任意の空でないイデアルの集合には、包含関係に関して極小元が存在します。
- これは半順序集合における「極小元の存在」としての最小値原理の一例です。

## 順序数の和

順序数の**和**は、2つの整列集合を「前後に並べた」ときの順序型として定義されます。
以下、数学的な定義と特徴を順に説明します。

### 1. 順序数の和の定義（直和による定義）

__直観的なイメージ__

2つの整列集合 <img src="tmp/76d737f0fd6643bedce44d8996c3db7f.png" class="math-inline" /> と <img src="tmp/3f366266c3d64d9ec26403b5741aede4.png" class="math-inline" /> があるとき、これらを**前後に並べた**新しい整列集合を作ります。

- 台集合：<img src="tmp/cb335d6740d1bd11636fb3cc54d74a83.png" class="math-inline" />（直和：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> を互いに素なコピーとして考える）
- 順序：
  - <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の元同士は <img src="tmp/9f77b640f8c0c0ea881294ed31e895c4.png" class="math-inline" /> の順序
  - <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の元同士は <img src="tmp/25131f566c061324dc3663273d436727.png" class="math-inline" /> の順序
  - <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の元はすべて <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の元より小さい：<img src="tmp/ec1a8b6c5515e18448e2aa36bb105ec5.png" class="math-inline" />

この整列集合の順序型を、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の順序型 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の順序型 <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の**和** <img src="tmp/523ee0a625bb9be4d02ba6cf448d12be.png" class="math-inline" /> と定義します。

__順序数としての形式的定義__

順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> に対し、<img src="tmp/523ee0a625bb9be4d02ba6cf448d12be.png" class="math-inline" /> を次のように定義できます。

- まず、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> を**互いに素**な整列集合として実現する（例えば <img src="tmp/3915e1f41d55855c7ea8d3911978f6ea.png" class="math-inline" />）。
- これらの直和に、上記の順序（<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の元をすべて <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の元より小さいとする）を入れる。
- この整列集合の順序型が <img src="tmp/523ee0a625bb9be4d02ba6cf448d12be.png" class="math-inline" />。

より集合論的には、超限再帰によって次のように定義することもできます。

1. **基底**：<img src="tmp/c241ee17471fbbd5d3fee092d46a3e52.png" class="math-inline" />
2. **後続順序数**：<img src="tmp/70de2424c935a6244f37f22c6107f26e.png" class="math-inline" />
3. **極限順序数**：<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> が極限順序数のとき、<img src="tmp/6796a8d94c4ee9366e9cba2943156eb5.png" class="math-inline" />

### 2. 和の基本的な性質

__(1) 結合律__

順序数の和は**結合的**です：



<div class="math-display-container"><img src="tmp/4ea8db7dc4cce385182968de2274978e.png" class="math-display" /></div>



証明の概略：

- 左辺は「まず <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> を並べ、その後に <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> を並べた」順序型。
- 右辺は「<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> の後に、<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> と <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> を並べたものを並べた」順序型。
- いずれも「<img src="tmp/e8604fb1e70a912aa41b59393ecde955.png" class="math-inline" /> をこの順に並べた」整列集合と順序同型になる。

__(2) 非可換性__

順序数の和は一般に**可換ではありません**。

例：

- <img src="tmp/733f3fc21558a8ba4ba2f64c03a0c248.png" class="math-inline" /> と <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> を比較する。
  - <img src="tmp/733f3fc21558a8ba4ba2f64c03a0c248.png" class="math-inline" />：自然数 <img src="tmp/4774377971f2d39bdd07f44c6eeb2e2b.png" class="math-inline" /> の後に自然数全体 <img src="tmp/2e5d5611945830cecdaca65e8ed2d466.png" class="math-inline" /> を並べたもの。これは自然数全体 <img src="tmp/2e5d5611945830cecdaca65e8ed2d466.png" class="math-inline" /> と順序同型なので、順序型は <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />。
  - <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" />：自然数全体 <img src="tmp/2e5d5611945830cecdaca65e8ed2d466.png" class="math-inline" /> の後に1つの元 <img src="tmp/d0571cec34ee5ee3f2f4763980f82840.png" class="math-inline" /> を並べたもの。
    これは自然数とは順序同型でない（最大元がある）ので、順序型は <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" />。

したがって、



<div class="math-display-container"><img src="tmp/fdf6567912e8e35053fd9fe427adb3b9.png" class="math-display" /></div>



となり、和は可換ではありません。

__(3) 左簡約律と右簡約律__

- **左簡約律**：<img src="tmp/24b3a786367b592ff2033bd40fb7fcfc.png" class="math-inline" />
- **右簡約律**は一般に成り立ちません。
  例：<img src="tmp/456db3c3fb38d9841337882e3a382613.png" class="math-inline" /> だが <img src="tmp/ac92d1f6d76d47af94eca9b34a381342.png" class="math-inline" />

__(4) 単調性__

- **左からの和**は単調：<img src="tmp/93b471a97c3c4a9845a6f4dc0aab01ff.png" class="math-inline" />
- **右からの和**は単調：<img src="tmp/c5fbc7530237821f8cae395c6ecf65ce.png" class="math-inline" />

ただし、左からの和は狭義単調、右からの和も狭義単調です。

### 3. 和の具体例

__(1) 有限順序数との和__

有限順序数 <img src="tmp/9386bc62898846f38fe79180d9f75b6e.png" class="math-inline" /> に対しては、通常の自然数の和と一致します。

- 例：<img src="tmp/5c6a15732a5c2c12dbbee85cc115ca1a.png" class="math-inline" />（順序数としても同じ）

__(2) 無限順序数との和__

- <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" />：自然数の列の後に1つ要素を追加した順序型。
- <img src="tmp/7268ffa5be887fcc0ac6e9f6f8f1b59a.png" class="math-inline" />：自然数の列の後に2つ、3つ、…要素を追加。
- <img src="tmp/e05488010dad4e4bdb08e134845a24b0.png" class="math-inline" />：自然数の列を2つ並べた順序型。
- <img src="tmp/48cccadf9d5b19d473f7145a13c1f8ab.png" class="math-inline" />：自然数の列を3つ並べた順序型。

__(3) 極限順序数との和__

- <img src="tmp/f968bc3229fbdcea4d0577a39ad60e95.png" class="math-inline" />：<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> の後に自然数全体を並べた順序型。
  - 例：<img src="tmp/9597cb7a7e31588fd8ec0500cad9e2f3.png" class="math-inline" />（順序同型）
- 一般に、<img src="tmp/f24af8e099a6780f49779a4d8bdfcdeb.png" class="math-inline" />（<img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> は極限順序数）は、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> の後に「型 <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> の整列集合」を並べたもの。

### 4. 和の幾何学的な解釈

順序数の和は、「直線上の区間を前後に並べる」操作として理解できます。

- <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" />：長さ <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> の区間
- <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" />：長さ <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の区間
- <img src="tmp/523ee0a625bb9be4d02ba6cf448d12be.png" class="math-inline" />：これらを**左から右に**並べたときの「全体の長さ（順序型）」

このイメージから、非可換性も自然に理解できます。

- 「1メートルの区間＋無限に続く区間」＝無限に続く区間（型 <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />）
- 「無限に続く区間＋1メートルの区間」＝無限に続く区間の後に1メートルが付いたもの（型 <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" />）

### 5. 和と他の演算の関係

__(1) 和と積の関係__

順序数の積 <img src="tmp/42585268f08ccf21650310fe301440ba.png" class="math-inline" /> は、「長さ <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の列の各要素を <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> で置き換えた」順序型です。和は積の特殊な場合として現れます。

- 例：<img src="tmp/e05488010dad4e4bdb08e134845a24b0.png" class="math-inline" />
- 一般に、<img src="tmp/28a145ffee924c8ab459d44ae29a75cc.png" class="math-inline" />（<img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> 回）＝ <img src="tmp/8dc30127789cdc93b26ed33bd18c4fc8.png" class="math-inline" />

__(2) 和と巾の関係__

巾 <img src="tmp/a4c0814d85b3414f4b1dedbed502d5fb.png" class="math-inline" /> は「長さ <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の列の辞書式順序」ですが、和は「列の連結」に対応します。

- 例：<img src="tmp/fd008f3c43680f61586487c6a823220f.png" class="math-inline" /> は有限長の自然数列全体の辞書式順序。
- <img src="tmp/d0b8327b0e7143e373e1983e418cec17.png" class="math-inline" /> は、まず自然数の列、その後ろに有限長の自然数列全体を並べた順序型。

### 6. 応用：順序数の標準形（Cantor標準形）における和

任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は、次の**Cantor標準形**に一意に表せます。



<div class="math-display-container"><img src="tmp/469c7c2a9607c91bf4753af5cab0df2b.png" class="math-display" /></div>



ここで、

- <img src="tmp/0c981f8a25ba73f39eb99d26092dece7.png" class="math-inline" />
- <img src="tmp/bb564be474656ead5d6d88f7eb7b521d.png" class="math-inline" /> は正の自然数

この表示では、和が**主要な構成要素**として現れます。
すなわち、順序数は「<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の巾」の**重み付き和**として分解されます。

## 順序数の積

順序数の**積**は、和よりも少し複雑ですが、「長さ <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の列の各項を <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> で置き換えた」整列集合の順序型として理解できます。
以下、数学的な定義と特徴を順に説明します。

### 1. 順序数の積の定義（直積の辞書式順序）

__直観的なイメージ__

2つの整列集合 <img src="tmp/76d737f0fd6643bedce44d8996c3db7f.png" class="math-inline" />（順序型 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" />）と <img src="tmp/3f366266c3d64d9ec26403b5741aede4.png" class="math-inline" />（順序型 <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" />）があるとき、**直積** <img src="tmp/9aa33bf4c800bdd83164f41c1c878b1b.png" class="math-inline" /> に次のような**辞書式順序**を入れます。

- 要素：ペア <img src="tmp/d06c6d246161dd0896775d08d0eb36bb.png" class="math-inline" />（<img src="tmp/aa0b4e95b310fd021c874720d3c0e076.png" class="math-inline" />）
- 順序：
  まず第2成分 <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> を比較し、それで決まらなければ第1成分 <img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> を比較する：
  

<div class="math-display-container"><img src="tmp/540e993fc085d61d2c82778c583837bb.png" class="math-display" /></div>



この整列集合 <img src="tmp/b60f83ade333a773194083cc6bc3b357.png" class="math-inline" /> の順序型を、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の**積** <img src="tmp/f8af908fff6e1fbdacb4ae574e67e478.png" class="math-inline" /> と定義します。

__順序数としての形式的定義（超限再帰）__

順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> に対し、<img src="tmp/f8af908fff6e1fbdacb4ae574e67e478.png" class="math-inline" /> を超限再帰によって次のように定義することもできます。

1. **基底**：

   

<div class="math-display-container"><img src="tmp/d7567b0e36567a7edbcdd2f611ff0de4.png" class="math-display" /></div>


2. **後続順序数**：

   

<div class="math-display-container"><img src="tmp/dd59ec09cc9916c4e16f8f5597c9149f.png" class="math-display" /></div>


3. **極限順序数**：
   <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> が <img src="tmp/b0782419bb8039fe8b6afc6b67925494.png" class="math-inline" /> でない極限順序数のとき、

   

<div class="math-display-container"><img src="tmp/0b8e64c7fa43b3e58b2fc4cd2e337bd2.png" class="math-display" /></div>



この定義は、直積の辞書式順序による定義と一致します。

### 2. 積の基本的な性質

__(1) 結合律__

順序数の積は**結合的**です：



<div class="math-display-container"><img src="tmp/148af923cf5f4324708c7ab8f3a6eed8.png" class="math-display" /></div>



証明の概略：

- 左辺は「長さ <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> の列の各項を <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> で置き換え、さらに各 <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> を <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> で置き換えた」順序型。
- 右辺は「長さ <img src="tmp/b3514d951f7b4fdb4fab57dc5800ac3b.png" class="math-inline" /> の列の各項を <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> で置き換えた」順序型。
- いずれも「長さ <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> の列の各項が、さらに長さ <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の列になっている」ような2重の列の辞書式順序と順序同型になる。

__(2) 非可換性__

順序数の積は一般に**可換ではありません**。

例：

- <img src="tmp/9ad0524c1b3b3c22025739a7ef6abe1f.png" class="math-inline" /> と <img src="tmp/3a57b297e98d4341dd09fa9d7182a486.png" class="math-inline" /> を比較する。
  - <img src="tmp/9ad0524c1b3b3c22025739a7ef6abe1f.png" class="math-inline" />：長さ <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の列の各項を <img src="tmp/d8429da08828f207b99be93378783e65.png" class="math-inline" /> で置き換えたもの。これは自然数全体と順序同型になる（可算順序数）ので、順序型は <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />。
  - <img src="tmp/3a57b297e98d4341dd09fa9d7182a486.png" class="math-inline" />：長さ <img src="tmp/748d014a08c85df236ffbfbc57dfcde3.png" class="math-inline" /> の列の各項を自然数全体で置き換えたもの。
    これは「自然数の列＋自然数の列」なので、順序型は <img src="tmp/e05488010dad4e4bdb08e134845a24b0.png" class="math-inline" />。

したがって、



<div class="math-display-container"><img src="tmp/9e5fb8a8e40dfc131e50d28e009ecd51.png" class="math-display" /></div>



となり、積は可換ではありません。

__(3) 分配律__

順序数の積は、和に対して**左分配律**を満たします：



<div class="math-display-container"><img src="tmp/d26679b5936faf84bb8f9e5ab185b737.png" class="math-display" /></div>



**右分配律**は一般に成り立ちません。
例：<img src="tmp/315fc20588a16d756a9ac1e1e7cd285d.png" class="math-inline" />
一方、<img src="tmp/61a4e926c7ac7ae615590a926f1bf80c.png" class="math-inline" /> なので <img src="tmp/80ce3bbd5a60555b0ba42ddef0b41243.png" class="math-inline" />

__(4) 単調性__

- **左からの積**は単調：<img src="tmp/54b09bdce9ad4979719c32eba5830a67.png" class="math-inline" />（<img src="tmp/ebb816e4d170b580832e1d18f302e2b5.png" class="math-inline" /> のとき）
- **右からの積**も単調：<img src="tmp/4f9913a3886a568c6ac849828c2e9737.png" class="math-inline" />（<img src="tmp/dbfef2bb46c9a82a4feb3413ab8e8836.png" class="math-inline" /> のとき）

### 3. 積の具体例

__(1) 有限順序数との積__

有限順序数 <img src="tmp/9386bc62898846f38fe79180d9f75b6e.png" class="math-inline" /> に対しては、通常の自然数の積と一致します。

- 例：<img src="tmp/e67075b8c7ccdccf514a4400eaf59a20.png" class="math-inline" />（順序数としても同じ）

__(2) 無限順序数との積__

- <img src="tmp/c6164b27eaf5edfa8ca0bad575cc6a44.png" class="math-inline" />：自然数の列を2つ並べた順序型。
- <img src="tmp/3822991809cef4a9a64d692e84267603.png" class="math-inline" />：自然数の列を3つ並べた順序型。
- <img src="tmp/0eacf2194a222faaa50252bd2b72fe33.png" class="math-inline" />：自然数のペア <img src="tmp/37b8a07fcc40b9fa20f047fcc39dde7e.png" class="math-inline" /> を辞書式順序で並べた順序型。
- <img src="tmp/1d955c59b54b917f6c33f7954c829f6d.png" class="math-inline" />：長さ <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の0-1列の辞書式順序は自然数全体と順序同型。

__(3) 極限順序数との積__

- <img src="tmp/8ce949463d54df136ef9d903d18e62ed.png" class="math-inline" />：長さ <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の列の各項を <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> で置き換えた順序型。
  - 例：<img src="tmp/2a238c0c1fbf13c013d727624aeae0d4.png" class="math-inline" />（順序同型）
- 一般に、<img src="tmp/498df24666b947174dd54206b7ca91f0.png" class="math-inline" />（<img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> は極限順序数）は、「各項が型 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> の整列集合である長さ <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> の列」の辞書式順序の順序型。

### 4. 幾何学的な解釈

順序数の積は、「平面や高次元の格子」の整列順序として理解できます。

- <img src="tmp/f8af908fff6e1fbdacb4ae574e67e478.png" class="math-inline" />：
  「<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> 本の縦線」のそれぞれに「<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> 個の点」が並んでいると考え、**下から上へ、左から右へ**と辞書式順序で並べたときの順序型。

例：

- <img src="tmp/3a57b297e98d4341dd09fa9d7182a486.png" class="math-inline" />：1列目に自然数 <img src="tmp/18991571b007a883e99c3893b3740bdf.png" class="math-inline" />、2列目に自然数 <img src="tmp/18991571b007a883e99c3893b3740bdf.png" class="math-inline" /> を並べ、1列目をすべて読み終えてから2列目を読む。
- <img src="tmp/9ad0524c1b3b3c22025739a7ef6abe1f.png" class="math-inline" />：
  各列に2つの点（0と1）があり、列が無限に続く。このとき、読み方は「各列の0を読んでから1を読む」ではなく、「まずすべての列の0を読み、次にすべての列の1を読む」という辞書式順序になるため、自然数全体と順序同型になる。

### 5. 積と他の演算の関係

__(1) 積と和の関係__

- 和は積の特殊な場合として現れることがあります：
  

<div class="math-display-container"><img src="tmp/bd07058ff70b30b927774e0221775ca8.png" class="math-display" /></div>


- 一方、積は和よりも「高次元の構造」を表現します。

__(2) 積と巾の関係__

巾 <img src="tmp/a4c0814d85b3414f4b1dedbed502d5fb.png" class="math-inline" /> は、「長さ <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の列の各項が <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> 未満の順序数であるような列」の辞書式順序の順序型です。積 <img src="tmp/f8af908fff6e1fbdacb4ae574e67e478.png" class="math-inline" /> は、各項が**ちょうど <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" />** であるような列の辞書式順序と見なせます。

- 例：<img src="tmp/0eacf2194a222faaa50252bd2b72fe33.png" class="math-inline" /> は、長さ <img src="tmp/748d014a08c85df236ffbfbc57dfcde3.png" class="math-inline" /> の自然数列の辞書式順序としても、長さ <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の自然数列（各項が自然数）の辞書式順序としても理解できます。

### 6. 応用：順序数の標準形（Cantor標準形）における積

任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は、次の**Cantor標準形**に一意に表せます。



<div class="math-display-container"><img src="tmp/469c7c2a9607c91bf4753af5cab0df2b.png" class="math-display" /></div>



ここで、

- <img src="tmp/0c981f8a25ba73f39eb99d26092dece7.png" class="math-inline" />
- <img src="tmp/bb564be474656ead5d6d88f7eb7b521d.png" class="math-inline" /> は正の自然数

この表示では、積が**係数部分**として現れます。
すなわち、順序数は「<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の巾」に自然数を**掛けたもの**の和として分解されます。

## 順序数の巾

順序数の**巾（べき乗）** は、順序数の演算の中で最も複雑で、かつ重要なものの1つです。
以下、数学的な定義と特徴を順に説明します。

### 1. 順序数の巾の定義（超限再帰による定義）

順序数の巾 <img src="tmp/a4c0814d85b3414f4b1dedbed502d5fb.png" class="math-inline" /> は、**超限再帰（transfinite recursion）** によって定義されます。

__基底と再帰段階__

順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> に対し、<img src="tmp/a4c0814d85b3414f4b1dedbed502d5fb.png" class="math-inline" /> を次のように定義します。

1. **基底**：

   

<div class="math-display-container"><img src="tmp/d4c63c502ad8fa9d21bdea302b1a3315.png" class="math-display" /></div>



   （ここで <img src="tmp/28761683c454cb781fdb6aac6bd8a6df.png" class="math-inline" /> は順序数 <img src="tmp/4774377971f2d39bdd07f44c6eeb2e2b.png" class="math-inline" />）
2. **後続順序数に対する定義**：<img src="tmp/561149dd161e99a34a7eaaf0de0f54c0.png" class="math-inline" />（後続順序数）のとき、

   

<div class="math-display-container"><img src="tmp/648b040e0b43a6788eeb6ad2ed93645b.png" class="math-display" /></div>



   ここで <img src="tmp/356513ded34229fab8ea72b65824828d.png" class="math-inline" /> は順序数の積（後述）。
3. **極限順序数に対する定義**：
   <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> が <img src="tmp/b0782419bb8039fe8b6afc6b67925494.png" class="math-inline" /> でない極限順序数のとき、

   

<div class="math-display-container"><img src="tmp/fab41a2b644302a4b1623be2c24d0164.png" class="math-display" /></div>



   すなわち、<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> 未満の順序数に対する巾の**上限**。

この定義により、任意の順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> に対して <img src="tmp/a4c0814d85b3414f4b1dedbed502d5fb.png" class="math-inline" /> が一意に定まります。

### 2. 直観的な意味：長さ <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の列の辞書式順序

順序数の巾 <img src="tmp/a4c0814d85b3414f4b1dedbed502d5fb.png" class="math-inline" /> は、次のような**整列集合の順序型**として理解できます。

- 要素：長さ <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の列 <img src="tmp/9f134a5fec0d361595e58bfc2789f3b9.png" class="math-inline" /> で、各 <img src="tmp/346c917c8d9af79abb1da5b179089ab9.png" class="math-inline" /> は <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> 未満の順序数（<img src="tmp/7c788381f06d59ce0555800adf1ef326.png" class="math-inline" />）。
- 順序：**辞書式順序（lexicographic order）**
  2つの列 <img src="tmp/2e8ddc4bd31762081277bdb851e5da29.png" class="math-inline" /> に対し、最初に値が異なる添字 <img src="tmp/ecc67237aa5f00dfb324a03ad9ea92bf.png" class="math-inline" /> で
  

<div class="math-display-container"><img src="tmp/dd03587d8b941144a4851187ccc0acf9.png" class="math-display" /></div>



  なら <img src="tmp/bfef9ab829cdf901c9a493d7e2c98f78.png" class="math-inline" /> と定義する。

この整列集合の順序型が <img src="tmp/a4c0814d85b3414f4b1dedbed502d5fb.png" class="math-inline" /> です。

例：

- <img src="tmp/2e449a98660aef94b3f42d32a3e351be.png" class="math-inline" />（2元集合 <img src="tmp/d8429da08828f207b99be93378783e65.png" class="math-inline" />）のとき：
  - <img src="tmp/edf6abb9fb79e3c90b5730fe145ed4d8.png" class="math-inline" /> は長さ <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> の0-1列の辞書式順序の順序型（有限順序数としての <img src="tmp/edf6abb9fb79e3c90b5730fe145ed4d8.png" class="math-inline" /> と一致）。
- <img src="tmp/27cdecc003f972c290f30bf87a0eb395.png" class="math-inline" /> のとき：
  - <img src="tmp/fd008f3c43680f61586487c6a823220f.png" class="math-inline" /> は「有限長の自然数列」を辞書式順序で並べた整列集合の順序型。

### 3. 巾の基本的な性質

__(1) 指数法則__

順序数の巾は、次の指数法則を満たします。

1. **積の巾**：

   

<div class="math-display-container"><img src="tmp/eeb87d8f97543a6edfa7202bf986fd53.png" class="math-display" /></div>


2. **巾の巾**：

   

<div class="math-display-container"><img src="tmp/36943db085bf76c90a6b065a11b5b8c0.png" class="math-display" /></div>



ただし、**和の巾**については一般に成り立ちません。

- 例：<img src="tmp/16e57c3dc41ea365e895b784c6f13bd2.png" class="math-inline" />
  一方、<img src="tmp/02c55c115f8fe3f53463ae13d02809fe.png" class="math-inline" /> なので <img src="tmp/43df3ec3d74bfe030579a076a75a2eed.png" class="math-inline" />

__(2) 単調性__

- <img src="tmp/070b97c8b8a8dbb0816e17044e6ff4bf.png" class="math-inline" /> のとき、<img src="tmp/7ad0980e2685d5358f4fd9aab352b238.png" class="math-inline" /> は**狭義単調増加**：
  

<div class="math-display-container"><img src="tmp/7e4807e0f0360808c285c23eb8e4aeed.png" class="math-display" /></div>


- <img src="tmp/dbfef2bb46c9a82a4feb3413ab8e8836.png" class="math-inline" /> のとき、<img src="tmp/70cc653762314f46abdc8a6ab247be36.png" class="math-inline" /> も狭義単調増加：
  

<div class="math-display-container"><img src="tmp/b55f3ac51da93512bf1f7ee14fd8a43e.png" class="math-display" /></div>



__(3) 有限順序数との関係__

- 有限順序数 <img src="tmp/9386bc62898846f38fe79180d9f75b6e.png" class="math-inline" /> に対する <img src="tmp/a5a61c9026ff18134f478acfb12ea6d9.png" class="math-inline" /> は、通常の自然数のべき乗と一致します。
- 例：<img src="tmp/769dc63165bdc530f2a86b57d18d1298.png" class="math-inline" />（順序数としても同じ）

### 4. 重要な例と特徴

__(1) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の巾__

- <img src="tmp/9f41876bc45225a0601c006bd341daff.png" class="math-inline" />
- <img src="tmp/720faebaf35561abc302debf0f13e910.png" class="math-inline" />（自然数のペア <img src="tmp/37b8a07fcc40b9fa20f047fcc39dde7e.png" class="math-inline" /> を辞書式順序で並べたものの順序型）
- <img src="tmp/0352c68d9af77919adb7fca6e5913266.png" class="math-inline" /> も同様に定義され、いずれも可算順序数です。
- <img src="tmp/fd008f3c43680f61586487c6a823220f.png" class="math-inline" />：有限長の自然数列全体の辞書式順序の順序型（可算順序数）。
- 一般に、<img src="tmp/c9e9a8634b3a854726184360a0ebc82b.png" class="math-inline" />（<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> 可算）は可算順序数です。

__(2) 連続関数としての巾__

- 関数 <img src="tmp/7ad0980e2685d5358f4fd9aab352b238.png" class="math-inline" /> は、極限順序数で**連続**です：
  

<div class="math-display-container"><img src="tmp/0bbc5683acaa50577e3e2bf6924bf168.png" class="math-display" /></div>



  が定義そのものです。

__(3) 正規関数（normal function）__

- <img src="tmp/070b97c8b8a8dbb0816e17044e6ff4bf.png" class="math-inline" /> のとき、<img src="tmp/7ad0980e2685d5358f4fd9aab352b238.png" class="math-inline" /> は**正規関数**です：
  1. 狭義単調増加
  2. 極限順序数で連続

正規関数は、**不動点**（<img src="tmp/41488bfa5cd3c34eb68a954a8214b36c.png" class="math-inline" /> となる <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" />）を無数に持つことが知られています。
例：<img src="tmp/0e7d2167ade1e74e978b9bcefe7d2dc9.png" class="math-inline" /> 数（<img src="tmp/6b408c43adbac51787b5d519540b50f4.png" class="math-inline" /> を満たす順序数）は、この不動点として定義されます。

### 5. 巾と基数のべき乗の違い

順序数の巾と、**基数のべき乗**は別物ですので注意が必要です。

- 順序数の巾：**順序構造**を考慮した演算（辞書式順序による整列集合の順序型）。
- 基数のべき乗：**濃度**だけを考えた演算（集合のべき集合や関数空間の濃度）。

例：

- 順序数として：<img src="tmp/544445452cf5d32b8f82ee8ae22788ae.png" class="math-inline" />（可算順序数）
- 基数として：<img src="tmp/85266fb621b032f85f918baf8c89cf88.png" class="math-inline" />（連続体濃度、非可算）

この違いは、**順序構造を忘れて濃度だけを見るかどうか**によって生じます。

### 6. 応用：順序数の標準形（Cantor標準形）

順序数の巾を用いると、任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は次の形に一意に表せます（**Cantor標準形**）。



<div class="math-display-container"><img src="tmp/469c7c2a9607c91bf4753af5cab0df2b.png" class="math-display" /></div>



ここで、

- <img src="tmp/0c981f8a25ba73f39eb99d26092dece7.png" class="math-inline" />（順序数）
- <img src="tmp/bb564be474656ead5d6d88f7eb7b521d.png" class="math-inline" /> は正の自然数

この表示は、順序数の**構造**を明示的に表すのに役立ちます。

### 例題

__問題__

<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> を任意の**整列集合**（well-ordered set）とする。
このとき、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と順序同型となる順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が**唯一存在**する。

上記命題を証明せよ。

---

__証明の概略__

__(1) 存在性：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と順序同型な順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が存在する__

**方針**：超限再帰（transfinite recursion）を用いて、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から順序数への順序同型写像を構成する。

1. <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の最小元 <img src="tmp/fa90aad2c4990b7bc44434aef9fe632c.png" class="math-inline" /> を <img src="tmp/b0782419bb8039fe8b6afc6b67925494.png" class="math-inline" />（順序数）に対応させる。
2. すでに <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の始片 <img src="tmp/3b8ef8115b9405e647b84ccc9f2f83cc.png" class="math-inline" /> が順序数 <img src="tmp/42e88e1363adee2084f1afbdda2b62f7.png" class="math-inline" /> と順序同型になっていると仮定する。
3. <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> の**直後の元** <img src="tmp/5da2170601078c1d56bbd43046c0713a.png" class="math-inline" />（存在すれば）に対しては、<img src="tmp/97ef899e043a4cc185868dadd36778b6.png" class="math-inline" /> と定める。
4. <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> が極限点（それより小さい元の上限）のときは、<img src="tmp/2431d55ee9de4d724801a85f2b2310b3.png" class="math-inline" /> と定める。
5. この構成により、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> 全体はある順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と順序同型になる。

より厳密には、次のような関数 <img src="tmp/a8efce012df6fecaf676e18ae3bad564.png" class="math-inline" /> を超限再帰で定義します。

- <img src="tmp/6a071efc5237bba47e8ecebedca03430.png" class="math-inline" />

この <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> が順序同型写像を与え、その像はある順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> になります。

__(2) 一意性：その順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は一意に定まる__

**方針**：2つの順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> がともに <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と順序同型だとすると、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> も順序同型になるが、順序数の性質から <img src="tmp/da46504ea44088ee9b39f2bcc56d0a2a.png" class="math-inline" /> となる。

1. <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と順序同型、かつ <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> とも順序同型だと仮定する。
2. すると <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> は順序同型である。
3. 順序数の比較定理により、2つの順序数が順序同型ならば**等しい**。
   - 実際、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の真の始片と順序同型なら、<img src="tmp/00ad5e80abd29770117b1b6dff1c56e3.png" class="math-inline" /> となるが、これは <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> 自身が順序数であることと矛盾する。
4. よって <img src="tmp/37692c5fcbf9c794c39abb122b1dbafd.png" class="math-inline" />。

したがって、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と順序同型な順序数は一意に定まります。

__3. 「任意の整数集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" />」についての注意__

元の命題「<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> を任意の整数集合とする」には問題があります。

- 整数全体 <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> は、通常の大小関係 <img src="tmp/de581dcc170a4139b7289290f1c37c7f.png" class="math-inline" /> に関して**整列集合ではありません**（例：負の整数全体に最小元がない）。
- したがって、<img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> はどの順序数とも順序同型にはなりません。
- 部分集合 <img src="tmp/0d0da8f69556f109f1bcdeb8190d5bc3.png" class="math-inline" /> についても、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が整列集合であるとは限りません。

**反例**：

- <img src="tmp/8be68fc2a1c3bf559f55ec9d1027a5b0.png" class="math-inline" />（整数全体）は整列集合でないので、順序同型な順序数は存在しない。
- <img src="tmp/2a353911e1087f5856c81c5753987ae2.png" class="math-inline" />（負の整数全体）も最小元がないので整列集合でない。

一方、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が**下方に有界**な整数集合（例：自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" />）なら整列集合であり、順序同型な順序数（この場合は <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />）が存在します。

__4. まとめ__

- 正しい主張：**任意の整列集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は、ある順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と順序同型であり、その <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は一意に定まる**。
- 存在性は超限再帰による順序同型写像の構成で示され、一意性は「順序同型な2つの順序数は等しい」ことから従う。
- 「任意の整数集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" />」に対しては、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が整列集合とは限らないため、順序同型な順序数が存在するとは限らない。

---

### 置換公理

「置換公理（Axiom of Replacement）」は、**ZF公理系（Zermelo–Fraenkel set theory）** の重要な公理の1つです。
以下、その内容を説明し、その「証明」についても正確に述べます。

__1. 置換公理の内容__

__直観的な意味__

置換公理は、おおまかに言うと次のようなことを保証します。

> 集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の各元 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に対して、何らかの「対象」 <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> を**一意に**対応させる規則（関数のようなもの）が与えられたとき、
> その対応で得られる <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> 全体もまた集合になる。

つまり、「**関数の像は集合である**」ということを公理として要請します。

__形式的な記述__

置換公理は、次のような**公理図式（axiom schema）** として与えられます。

> 任意の論理式 <img src="tmp/288fa8457940d6558bf3c4b50a6a1651.png" class="math-inline" /> について、次が成り立つとする：


<div class="math-display-container"><img src="tmp/ada3eb7785abfd993d0c099a6be29c3a.png" class="math-display" /></div>


> （すなわち、<img src="tmp/86bea25b35a6ac9761eb775cc73398d4.png" class="math-inline" /> は「各 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に対して高々1つの <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> を定める」）
>
> このとき、任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に対して、


<div class="math-display-container"><img src="tmp/4f5bbb9768005e3430e66f79496abe37.png" class="math-display" /></div>


> が成り立つ。

言葉で言い換えると：

- <img src="tmp/288fa8457940d6558bf3c4b50a6a1651.png" class="math-inline" /> が「関数のような性質」（各 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に対して <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> が一意に決まる）を持つなら、
- 任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に対し、その「像」
  

<div class="math-display-container"><img src="tmp/f63942bafe88cff43b02ae511a235161.png" class="math-display" /></div>



  も集合である。

この集合を <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> と書きます。

__2. なぜ置換公理が必要か__

__歴史的背景__

- ツェルメロ（Zermelo）の元々の公理系には置換公理は含まれていませんでした。
- しかし、**順序数全体のクラス**や**累積階層 <img src="tmp/21f89b083ba97a37b2b9ab93a9d4a430.png" class="math-inline" />** など、「大きな」集合を扱う際に、置換公理がないと不都合が生じることがわかりました。
- フレンケル（Fraenkel）らによって追加され、現在の **ZF公理系** ができました。

__具体例：順序数全体は集合ではない__

- 順序数全体の集まり <img src="tmp/24df8a72caec0be7051202ec37ad58da.png" class="math-inline" /> は**真のクラス（proper class）** であり、集合にはなりません（Burali-Fortiのパラドックス）。
- しかし、**任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> に対して、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> 未満の順序数全体の集合** <img src="tmp/3eebc77e61de795cb74fe3a205dc991d.png" class="math-inline" /> は集合です。
- この「各順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> に対して集合 <img src="tmp/017a8301636842624f023aee49675c53.png" class="math-inline" /> を対応させる」操作が、置換公理によって正当化されます。

## 演習問題

### 問題

__問1（順序数の定義と基本性質）__

(1) フォン・ノイマン順序数の定義を述べ、その定義のもとで次を示せ。

- (a) 任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について、<img src="tmp/7d12786edc71c93b2272bc848a8f9933.png" class="math-inline" /> が成り立つ。
- (b) 順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> に対し、<img src="tmp/29b7b525deaf992bcdd385ebe7253468.png" class="math-inline" /> であることと <img src="tmp/00ad5e80abd29770117b1b6dff1c56e3.png" class="math-inline" /> であることは同値である。

(2) 順序数 <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />（自然数全体の順序型）について、次を答えよ。

- (a) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は後続順序数か、極限順序数か。
- (b) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の直前要素は存在するか。存在する場合は何か。
- (c) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> が極限順序数である理由を、直感的に説明せよ。

(3) 順序数 <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> について、次を答えよ。

- (a) <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> は後続順序数か、極限順序数か。
- (b) <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> の直前要素は何か。

__問2（順序数の大小と整列集合）__

(1) 順序数の大小関係 <img src="tmp/29b7b525deaf992bcdd385ebe7253468.png" class="math-inline" /> を <img src="tmp/00ad5e80abd29770117b1b6dff1c56e3.png" class="math-inline" /> で定義するとき、次を示せ。

- (a) 任意の2つの順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> は比較可能である（全順序性）。
- (b) 順序数の任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は、<img src="tmp/703433fde77b69514cee13f320ecf601.png" class="math-inline" /> に関して整列集合である。

(2) 整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> が与えられたとき、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> と順序同型となる順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が**唯一存在**することを証明せよ。

（ヒント：超限再帰を用いて <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から順序数への順序同型写像を構成し、一意性は順序数の比較定理から示す。）

__問3（最小値原理）__

(1) 整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の「最小値原理」を定義し、それが整列集合の定義とどのように関係するかを説明せよ。

(2) 自然数の最小値原理（任意の空でない自然数の部分集合は最小元を持つ）と、数学的帰納法の原理が同値であることを示せ。

(3) 半順序集合 <img src="tmp/e0b9e665ca60e72a6740e0d7751e4408.png" class="math-inline" /> において、「任意の空でない部分集合が極小元を持つ」という性質を「最小値原理」と呼ぶことがある。  
この性質と、整列集合の最小値原理との違いを具体例を挙げて説明せよ。

__問4（順序数の和）__

(1) 順序数の和 <img src="tmp/523ee0a625bb9be4d02ba6cf448d12be.png" class="math-inline" /> の定義（直和による定義または超限再帰による定義）を述べよ。

(2) 次の等式が成り立つかどうかを判定し、理由を簡潔に述べよ。

- (a) <img src="tmp/4f3680ce3f0315b0ffae3ca963d4bf8c.png" class="math-inline" />
- (b) <img src="tmp/3768e1be5fe83d9b27a16e0171e01478.png" class="math-inline" />
- (c) <img src="tmp/139acda3c8e8512a6d259e5a9c39dcdf.png" class="math-inline" />（一般の順序数 <img src="tmp/e8604fb1e70a912aa41b59393ecde955.png" class="math-inline" /> に対して）

(3) 順序数の和が一般に可換でないことを、具体例を用いて説明せよ。

__問5（順序数の積）__

(1) 順序数の積 <img src="tmp/42585268f08ccf21650310fe301440ba.png" class="math-inline" /> の定義（直積の辞書式順序による定義または超限再帰による定義）を述べよ。

(2) 次の等式が成り立つかどうかを判定し、理由を簡潔に述べよ。

- (a) <img src="tmp/97c1ad8c4ce92f5498cd992ed8ef9927.png" class="math-inline" />
- (b) <img src="tmp/224c99b38fc0ce87f05db0448b649dd3.png" class="math-inline" />
- (c) <img src="tmp/ce94bbf9893dcfa0fa975d24da97480f.png" class="math-inline" />（一般の順序数 <img src="tmp/e8604fb1e70a912aa41b59393ecde955.png" class="math-inline" /> に対して）

(3) 順序数の積が一般に可換でないことを、具体例を用いて説明せよ。

__問6（順序数の巾）__

(1) 順序数の巾 <img src="tmp/a4c0814d85b3414f4b1dedbed502d5fb.png" class="math-inline" /> の定義（超限再帰による定義）を述べよ。

(2) 次の値を求め、その順序型がどのような整列集合に対応するか説明せよ。

- (a) <img src="tmp/b47c72b7b01e3f09b09ce354c8b1c997.png" class="math-inline" />
- (b) <img src="tmp/d5e3a396a6bab873425ff37ef4ff2668.png" class="math-inline" />
- (c) <img src="tmp/fd008f3c43680f61586487c6a823220f.png" class="math-inline" />

(3) 順序数の巾について、次の指数法則が成り立つかどうかを判定し、理由を簡潔に述べよ。

- (a) <img src="tmp/78c719df47b8b259ca38798980b66d7c.png" class="math-inline" />
- (b) <img src="tmp/03a16a8ebab2e939a103bef56af67971.png" class="math-inline" />
- (c) <img src="tmp/62b75e725de500adac913334ba789138.png" class="math-inline" />

__問7（置換公理）__

(1) 置換公理（Axiom of Replacement）の内容を、論理式を用いて正確に述べよ。

(2) 置換公理が「関数の像は集合である」ことを保証する理由を、直感的に説明せよ。

(3) 置換公理を用いて、次のことを示せ。

- 任意の順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> に対して、<img src="tmp/a31833572a4dcad5c9cf45fc25b90445.png" class="math-inline" /> は集合である。

（ヒント：<img src="tmp/288fa8457940d6558bf3c4b50a6a1651.png" class="math-inline" /> を「<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は順序数で、<img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> 未満の順序数全体の集合である」とおき、置換公理を適用する。）

__問8（Cantor標準形）__

(1) 順序数のCantor標準形



<div class="math-display-container"><img src="tmp/469c7c2a9607c91bf4753af5cab0df2b.png" class="math-display" /></div>



について、次の問いに答えよ。

- (a) 各 <img src="tmp/4feb03feb7c732c65152165c52fde9ce.png" class="math-inline" /> と <img src="tmp/bb564be474656ead5d6d88f7eb7b521d.png" class="math-inline" /> が満たす条件を述べよ。
- (b) この表示が一意である理由を、直感的に説明せよ。

(2) 次の順序数をCantor標準形で表せ。

- (a) <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" />
- (b) <img src="tmp/838230f39aa4893752cb4d57f76fdc99.png" class="math-inline" />
- (c) <img src="tmp/315c9a286d4388b8da5d8311523c7372.png" class="math-inline" />

__問9（超限帰納法・超限再帰）__

(1) 順序数全体のクラス <img src="tmp/24df8a72caec0be7051202ec37ad58da.png" class="math-inline" /> 上の超限帰納法について、次の問いに答えよ。

- (a) 超限帰納法の主張を、「基底」「帰納段階」の2条件で述べよ。
- (b) 順序数を「0」「後続順序数」「極限順序数」の3種類に分けたときの、3段階の超限帰納法の形を書け。
- (c) なぜこの3段階で「すべての順序数」について命題が成り立つといえるのか、直感的に説明せよ。

(2) 次の命題を、超限帰納法を用いて証明する方針を述べよ（完全な証明でなくてよい）。

> すべての順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は推移的集合である。  
> （すなわち、<img src="tmp/d10ac4b712413caad40a38dd7f5415f6.png" class="math-inline" /> かつ <img src="tmp/b3ac70357fd167bc08c071acb37e59a8.png" class="math-inline" /> ならば <img src="tmp/6ae7746013b4453fc4b0cd84c8f2627b.png" class="math-inline" /> が成り立つ。）

(3) 超限再帰を用いて、順序数の和 <img src="tmp/523ee0a625bb9be4d02ba6cf448d12be.png" class="math-inline" /> を定義する際の「基底」「後続段階」「極限段階」を具体的に書け。

__問10（応用：整列集合の比較）__

2つの整列集合 <img src="tmp/76d737f0fd6643bedce44d8996c3db7f.png" class="math-inline" /> と <img src="tmp/3f366266c3d64d9ec26403b5741aede4.png" class="math-inline" /> が与えられているとする。

次の命題を、超限帰納法（または整列集合上の帰納法）を用いて証明せよ。

> <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は、次のいずれか一方が成り立つ。
> 1. <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> のある始片と順序同型である。
> 2. <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> のある始片と順序同型である。
> 3. <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は順序同型である。

（ヒント：<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の元を「同時に」比較しながら帰納法を適用する。実際の証明では、順序同型写像を超限再帰で構成し、その構成がうまくいくことを帰納法で示す。）


### 解答

以下、各問の解答です。

__問1（順序数の定義と基本性質）__

__(1) フォン・ノイマン順序数の定義と性質__

**定義（フォン・ノイマン順序数）**

集合 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が**順序数**であるとは、次の2条件を満たすことをいう。

1. **推移的（transitive）**：  
   <img src="tmp/4c61f40b06ad41a41b6377a7b51f0d5b.png" class="math-inline" />（<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> の元は <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> の部分集合でもある）。
2. **<img src="tmp/703433fde77b69514cee13f320ecf601.png" class="math-inline" /> に関して整列集合**：  
   <img src="tmp/7a1c574eadd58f5dcda2010e1ba7e2ec.png" class="math-inline" /> が整列集合である（任意の空でない部分集合が <img src="tmp/703433fde77b69514cee13f320ecf601.png" class="math-inline" /> に関する最小元を持つ）。

**(a) <img src="tmp/7d12786edc71c93b2272bc848a8f9933.png" class="math-inline" /> の証明**

- 順序数の大小は <img src="tmp/f949a7f5f90dcb9d545b0541ee0f85fe.png" class="math-inline" /> で定義される。
- よって <img src="tmp/6c3d84561c6b6c0936c07cc1d0ece0d2.png" class="math-inline" />。
- 右辺は <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> そのものなので、<img src="tmp/7d12786edc71c93b2272bc848a8f9933.png" class="math-inline" />。

**(b) <img src="tmp/ade1f69048f6124b4357341bee821f8e.png" class="math-inline" /> の証明**

- 定義により、<img src="tmp/29b7b525deaf992bcdd385ebe7253468.png" class="math-inline" /> は <img src="tmp/00ad5e80abd29770117b1b6dff1c56e3.png" class="math-inline" /> のこと。
- したがって同値である。

__(2) 順序数 <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> について__

**(a) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は後続順序数か、極限順序数か**

- 極限順序数である。

**(b) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の直前要素は存在するか**

- 存在しない。

**(c) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> が極限順序数である理由（直感的説明）**

- <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は自然数全体の順序型であり、任意の自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> に対して <img src="tmp/6263c987771690e784009fcdb7d0955c.png" class="math-inline" /> だが、<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は「ある自然数の次」として得られるものではない。
- すなわち、<img src="tmp/ec74ef74bfcb085813c6f86f1413cabf.png" class="math-inline" /> となる順序数 <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> は存在せず、それより小さいすべての有限順序数の「極限」として現れる。

__(3) 順序数 <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> について__

**(a) <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> は後続順序数か、極限順序数か**

- 後続順序数である（<img src="tmp/33f768f9b7a5f126aa364b9d4503fe7b.png" class="math-inline" /> と書けるため）。

**(b) <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> の直前要素は何か**

- <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" />

__問2（順序数の大小と整列集合）__

__(1) 順序数の全順序性と整列性__

**(a) 任意の2つの順序数は比較可能であること**

- 順序数 <img src="tmp/2eca3764c0b126d743b89c10781da605.png" class="math-inline" /> に対し、<img src="tmp/ae6ded84d46cd69814ecdc0763afc2b9.png" class="math-inline" /> はどちらかの始片になる。
- 整列集合の比較定理により、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> は順序同型か、一方が他方の真の始片になる。
- 順序数は「自分より小さい順序数全体」として定義されているので、順序同型なら等しい。
- したがって、<img src="tmp/24f830963b96201ad64b529a12de7d9c.png" class="math-inline" />（<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の始片）か、その逆か、あるいは等しいかのいずれかになる。
- すなわち、<img src="tmp/29b7b525deaf992bcdd385ebe7253468.png" class="math-inline" />、<img src="tmp/37692c5fcbf9c794c39abb122b1dbafd.png" class="math-inline" />、<img src="tmp/b4d58078d6159ee352ef89f89ce6e36c.png" class="math-inline" /> のいずれか1つだけが成り立つ。

**(b) 順序数の任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/703433fde77b69514cee13f320ecf601.png" class="math-inline" /> に関して整列集合であること**

- <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> を順序数の空でない集合とする。
- <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の最小元は、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の元の中で「どの元の要素にもなっていない順序数」として特徴づけられる。
- 実際、<img src="tmp/9cfdf06fb8eabd004dc51881a137adc8.png" class="math-inline" /> とおくと、<img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> のすべての元に含まれる最小の順序数であり、<img src="tmp/9507e0235f1b54262f27fa314e741f93.png" class="math-inline" /> となる（<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が集合であることと順序数の性質から）。
- よって <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/703433fde77b69514cee13f320ecf601.png" class="math-inline" /> に関して最小元を持つ。全順序性は (a) より従うので、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は整列集合。

__(2) 整列集合と順序同型な順序数の一意存在__

**命題**：任意の整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> に対し、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> と順序同型となる順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が唯一存在する。

**証明の概略**

**存在性**：

- 超限再帰により、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> から順序数への順序同型写像 <img src="tmp/ecf37bd7d1eed512ba2fdd3e0ad58109.png" class="math-inline" /> を構成する。
- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> の最小元 <img src="tmp/831a2636030c0a02e60887391a56cf5c.png" class="math-inline" /> に対し、<img src="tmp/119c4af3154cb480692d98115765ee73.png" class="math-inline" /> と定める。
- 帰納的に、<img src="tmp/510e7bf5a43891cb1005631c60476fff.png" class="math-inline" /> に対し、
  

<div class="math-display-container"><img src="tmp/3aadd207f30d42eba59df6be0e0a4003.png" class="math-display" /></div>


  と定義する（<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> が極限点の場合は上限、後続点の場合は <img src="tmp/264fed5f5ba6db26318929111b727110.png" class="math-inline" />）。
- この <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> は順序同型写像であり、その像はある順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> となる。

**一意性**：

- <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> が順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の両方と順序同型だと仮定する。
- すると <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> も順序同型になる。
- 順序数の比較定理により、2つの順序数が順序同型ならば等しい。
- よって <img src="tmp/37692c5fcbf9c794c39abb122b1dbafd.png" class="math-inline" />。

したがって、<img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> と順序同型な順序数は一意に定まる。

__問3（最小値原理）__

__(1) 整列集合の最小値原理__

**定義**：整列集合 <img src="tmp/4fd15d17444f119fd81db993a6c83eed.png" class="math-inline" /> の**最小値原理**とは、

> 任意の空でない部分集合 <img src="tmp/e6612a2b4cfad71807019d18e53b598f.png" class="math-inline" /> が最小元を持つ

という性質をいう。

- これは整列集合の定義そのものの一部であり、「整列性」と同義。

__(2) 自然数の最小値原理と数学的帰納法の同値性__

**自然数の最小値原理**：任意の空でない自然数の部分集合は最小元を持つ。

**数学的帰納法**：命題 <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> が

- <img src="tmp/b27dacc5676ded5b21aa21ada36bd16a.png" class="math-inline" /> が真
- 任意の <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> について <img src="tmp/87e470da05f50cdf7c4e5890d663a1c0.png" class="math-inline" />

ならば、すべての自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> について <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> が真。

**同値性の証明**

**(i) 最小値原理 ⇒ 数学的帰納法**

- 数学的帰納法の仮定を満たすが、ある <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> で <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> が偽になると仮定する。
- <img src="tmp/f5d5dd67bda666b97a41ee7890c05170.png" class="math-inline" /> は空でない。
- 最小値原理より <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> には最小元 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> が存在する。
- <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> が最小なので、<img src="tmp/0a217cd04e6e931ef3cfe5e26b84871d.png" class="math-inline" /> なるすべての <img src="tmp/1ebd8eed2a684f4edcbc6db4e42bfc27.png" class="math-inline" /> について <img src="tmp/ce4770d648c64b583ecb3b6daa2bd4b9.png" class="math-inline" /> は真。
- 帰納法の仮定より <img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> も真となるが、<img src="tmp/9507e0235f1b54262f27fa314e741f93.png" class="math-inline" /> より <img src="tmp/58b204b560f975705c8a675901d65715.png" class="math-inline" /> は偽 → 矛盾。
- よってそのような <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> は存在せず、すべての <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> で <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> が真。

**(ii) 数学的帰納法 ⇒ 最小値原理**

- 空でない自然数の部分集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が最小元を持たないと仮定する。
- <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" />：「<img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の元ではない」とおく。
- <img src="tmp/b27dacc5676ded5b21aa21ada36bd16a.png" class="math-inline" /> は真（<img src="tmp/ab27eee741a177e85b03f1baa00756d3.png" class="math-inline" /> なら <img src="tmp/b0782419bb8039fe8b6afc6b67925494.png" class="math-inline" /> が最小元になるが、最小元はないと仮定した）。
- <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> が真なら <img src="tmp/e4bc429c866bbde4b0f321d9b5139986.png" class="math-inline" /> も <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の元ではない（<img src="tmp/53573db474cbf9567707f04a1c3f34db.png" class="math-inline" /> なら <img src="tmp/e4bc429c866bbde4b0f321d9b5139986.png" class="math-inline" /> が最小元になるが、最小元はないと仮定した）。
- よって数学的帰納法により、すべての自然数 <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> について <img src="tmp/5ac279c7f9578d45a03e7376c98556af.png" class="math-inline" /> が真 → <img src="tmp/08bc0aa7b438e79f873552278da3d6b6.png" class="math-inline" /> となり矛盾。
- したがって <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は最小元を持つ。

__(3) 半順序集合における「極小元の存在」との違い__

**整列集合の最小値原理**：任意の空でない部分集合に**最小元**（その集合のすべての元より小さい元）が存在。

**半順序集合の「極小元の存在」**：任意の空でない部分集合に**極小元**（その集合の中でそれより真に小さい元が存在しない元）が存在。

**具体例**：

- <img src="tmp/8f9a8b06e58e7d3d84d7502c607b5b2f.png" class="math-inline" />、順序は <img src="tmp/5d82b024beeb88ca6e2143ceb0acc337.png" class="math-inline" /> のみ（<img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> と <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> は比較不能）。
- このとき <img src="tmp/bbe36f81377c21f5d6615f3b9ef1f3a7.png" class="math-inline" /> 自体は空でないが、最小元は存在しない（<img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> と <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> のどちらも他より小さいとはいえない）。
- しかし極小元は <img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> と <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> の両方存在する（それより小さい元はない）。
- よって「極小元の存在」は成り立つが、「最小値原理（最小元の存在）」は成り立たない。

__問4（順序数の和）__

__(1) 順序数の和の定義__

**直和による定義**：

- 整列集合 <img src="tmp/76d737f0fd6643bedce44d8996c3db7f.png" class="math-inline" />（順序型 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" />）と <img src="tmp/3f366266c3d64d9ec26403b5741aede4.png" class="math-inline" />（順序型 <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" />）の直和 <img src="tmp/cb335d6740d1bd11636fb3cc54d74a83.png" class="math-inline" /> に、次の順序を入れる：
  - <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の元同士は <img src="tmp/9f77b640f8c0c0ea881294ed31e895c4.png" class="math-inline" /> の順序
  - <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の元同士は <img src="tmp/25131f566c061324dc3663273d436727.png" class="math-inline" /> の順序
  - <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の元はすべて <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の元より小さい
- この整列集合の順序型を <img src="tmp/523ee0a625bb9be4d02ba6cf448d12be.png" class="math-inline" /> と定義する。

**超限再帰による定義**：

1. 基底：<img src="tmp/c241ee17471fbbd5d3fee092d46a3e52.png" class="math-inline" />
2. 後続：<img src="tmp/70de2424c935a6244f37f22c6107f26e.png" class="math-inline" />
3. 極限：<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> が極限順序数のとき、<img src="tmp/6796a8d94c4ee9366e9cba2943156eb5.png" class="math-inline" />

__(2) 等式の判定__

**(a) <img src="tmp/4f3680ce3f0315b0ffae3ca963d4bf8c.png" class="math-inline" />**

- 偽。<img src="tmp/f6f2da162c34be1440825ab199ea760e.png" class="math-inline" />（自然数全体と順序同型）、<img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" /> は自然数の列の後に1つ要素を追加した順序型で、最大元を持つため <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> とは順序同型でない。

**(b) <img src="tmp/3768e1be5fe83d9b27a16e0171e01478.png" class="math-inline" />**

- 偽。左辺は <img src="tmp/9597cb7a7e31588fd8ec0500cad9e2f3.png" class="math-inline" />、右辺は <img src="tmp/c61a7a5c21b8d91907d3c6d34fa675e8.png" class="math-inline" /> となり、最大元の有無が異なる。

**(c) <img src="tmp/139acda3c8e8512a6d259e5a9c39dcdf.png" class="math-inline" />**

- 真（結合律）。どちらも「<img src="tmp/e8604fb1e70a912aa41b59393ecde955.png" class="math-inline" /> をこの順に並べた」整列集合の順序型になる。

__(3) 和の非可換性の具体例__

- <img src="tmp/f6f2da162c34be1440825ab199ea760e.png" class="math-inline" />（自然数全体）
- <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" />：自然数の列の後に1つ要素を追加したもの（最大元あり）
- よって <img src="tmp/c00608e723e91fd621f5276656e00e92.png" class="math-inline" />。

__問5（順序数の積）__

__(1) 順序数の積の定義__

**直積の辞書式順序による定義**：

- 整列集合 <img src="tmp/76d737f0fd6643bedce44d8996c3db7f.png" class="math-inline" />（順序型 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" />）と <img src="tmp/3f366266c3d64d9ec26403b5741aede4.png" class="math-inline" />（順序型 <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" />）の直積 <img src="tmp/9aa33bf4c800bdd83164f41c1c878b1b.png" class="math-inline" /> に、辞書式順序
  

<div class="math-display-container"><img src="tmp/665367623c6b69dcf05070638efc3618.png" class="math-display" /></div>


  を入れる。
- この整列集合の順序型を <img src="tmp/42585268f08ccf21650310fe301440ba.png" class="math-inline" /> と定義する。

**超限再帰による定義**：

1. 基底：<img src="tmp/77990ba0a74ca7b072744000afdf449f.png" class="math-inline" />
2. 後続：<img src="tmp/7de7f9948f56ca8b090026d911144c48.png" class="math-inline" />
3. 極限：<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> が極限順序数のとき、<img src="tmp/689cb0de439b183cf9a42c3dc8772b59.png" class="math-inline" />

__(2) 等式の判定__

**(a) <img src="tmp/97c1ad8c4ce92f5498cd992ed8ef9927.png" class="math-inline" />**

- 偽。<img src="tmp/acdde78e2cffa42f98c5d9112efa5464.png" class="math-inline" />（長さ <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の0-1列の辞書式順序は自然数全体と順序同型）、<img src="tmp/06c09cb0fc78c022cfd655b4500164dc.png" class="math-inline" />（自然数の列を2つ並べたもの）。

**(b) <img src="tmp/224c99b38fc0ce87f05db0448b649dd3.png" class="math-inline" />**

- 真（結合律）。どちらも「自然数の列を6つ並べた」順序型 <img src="tmp/c959941fb7150e0ed6270ff33a55c221.png" class="math-inline" /> になる。

**(c) <img src="tmp/ce94bbf9893dcfa0fa975d24da97480f.png" class="math-inline" />**

- 真（左分配律）。どちらも「まず <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> の列、その後 <img src="tmp/38d840493750405a3139d8e7ab8dde68.png" class="math-inline" /> の列を並べ、各項を <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> で置き換えた」順序型になる。

__(3) 積の非可換性の具体例__

- <img src="tmp/acdde78e2cffa42f98c5d9112efa5464.png" class="math-inline" />（可算）
- <img src="tmp/06c09cb0fc78c022cfd655b4500164dc.png" class="math-inline" />（非可算の形を持つが、濃度は可算だが順序型は異なる）
- よって <img src="tmp/02f24cef1fbf3e1213f537d369cce4dd.png" class="math-inline" />。

__問6（順序数の巾）__

__(1) 順序数の巾の定義（超限再帰）__

1. 基底：<img src="tmp/26a85cd4f089c3fd0b626e0067b0782e.png" class="math-inline" />
2. 後続：<img src="tmp/5178bf81ca94eee115709933c1db75f9.png" class="math-inline" />
3. 極限：<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> が極限順序数のとき、<img src="tmp/3aa68ad4e1d705b192f4ff04155d58ce.png" class="math-inline" />

__(2) 値と対応する整列集合__

**(a) <img src="tmp/b47c72b7b01e3f09b09ce354c8b1c997.png" class="math-inline" />**

- 値：<img src="tmp/720faebaf35561abc302debf0f13e910.png" class="math-inline" />
- 対応する整列集合：自然数のペア <img src="tmp/37b8a07fcc40b9fa20f047fcc39dde7e.png" class="math-inline" /> を辞書式順序で並べたもの。

**(b) <img src="tmp/d5e3a396a6bab873425ff37ef4ff2668.png" class="math-inline" />**

- 値：<img src="tmp/544445452cf5d32b8f82ee8ae22788ae.png" class="math-inline" />
- 対応する整列集合：長さ <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の0-1列の辞書式順序（自然数全体と順序同型）。

**(c) <img src="tmp/fd008f3c43680f61586487c6a823220f.png" class="math-inline" />**

- 値：<img src="tmp/fd008f3c43680f61586487c6a823220f.png" class="math-inline" />
- 対応する整列集合：有限長の自然数列全体の辞書式順序。

__(3) 指数法則の判定__

**(a) <img src="tmp/78c719df47b8b259ca38798980b66d7c.png" class="math-inline" />**

- 真（積の巾）。定義より従う。

**(b) <img src="tmp/03a16a8ebab2e939a103bef56af67971.png" class="math-inline" />**

- 真（巾の巾）。定義より従う。

**(c) <img src="tmp/62b75e725de500adac913334ba789138.png" class="math-inline" />**

- 一般には偽。例：<img src="tmp/16e57c3dc41ea365e895b784c6f13bd2.png" class="math-inline" />、一方 <img src="tmp/02c55c115f8fe3f53463ae13d02809fe.png" class="math-inline" />。

__問7（置換公理）__

__(1) 置換公理の内容__

**置換公理（Axiom of Replacement）**：

任意の論理式 <img src="tmp/288fa8457940d6558bf3c4b50a6a1651.png" class="math-inline" /> について、次が成り立つとする。



<div class="math-display-container"><img src="tmp/ada3eb7785abfd993d0c099a6be29c3a.png" class="math-display" /></div>



（すなわち、<img src="tmp/86bea25b35a6ac9761eb775cc73398d4.png" class="math-inline" /> は「各 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に対して高々1つの <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> を定める」）

このとき、任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に対して、



<div class="math-display-container"><img src="tmp/4f5bbb9768005e3430e66f79496abe37.png" class="math-display" /></div>



が成り立つ。

__(2) 「関数の像は集合である」ことの直感的説明__

- <img src="tmp/288fa8457940d6558bf3c4b50a6a1651.png" class="math-inline" /> が「<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に対して <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> を一意に定める」なら、それは関数 <img src="tmp/ae8dfc9d7968799f14635213f636682f.png" class="math-inline" /> とみなせる。
- 置換公理は、この関数の像
  

<div class="math-display-container"><img src="tmp/26e707f68c2ed7fad9f28f96da39d497.png" class="math-display" /></div>


  が集合であることを保証する。

__(3) <img src="tmp/a31833572a4dcad5c9cf45fc25b90445.png" class="math-inline" /> が集合であることの証明__

- <img src="tmp/288fa8457940d6558bf3c4b50a6a1651.png" class="math-inline" /> を「<img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> は順序数で、<img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> 未満の順序数全体の集合である」と定義する。
- 各順序数 <img src="tmp/fefd88f33bff6924ab8dae7811b3cf3e.png" class="math-inline" /> に対して、そのような <img src="tmp/cebda1ac60ee1112b4abe181a82def7a.png" class="math-inline" /> は一意に定まる（順序数の性質より）。
- 置換公理を <img src="tmp/5fab90b00192da666f6a452df5d52b6b.png" class="math-inline" />（1点集合）に適用すると、
  

<div class="math-display-container"><img src="tmp/e0d0e4ccf75f29f2f531baf967437dd0.png" class="math-display" /></div>


  が集合であることがわかる。
- よって <img src="tmp/a31833572a4dcad5c9cf45fc25b90445.png" class="math-inline" /> も集合である。

__問8（Cantor標準形）__

__(1) Cantor標準形の条件と一意性__

**(a) 条件**

- <img src="tmp/0c981f8a25ba73f39eb99d26092dece7.png" class="math-inline" />（順序数）
- <img src="tmp/bb564be474656ead5d6d88f7eb7b521d.png" class="math-inline" /> は正の自然数

**(b) 一意性の直感的説明**

- 順序数は「<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> の巾」の重み付き和として一意に分解できる。
- 大きい巾から順に取り出すことで、係数 <img src="tmp/bb564be474656ead5d6d88f7eb7b521d.png" class="math-inline" /> と指数 <img src="tmp/4feb03feb7c732c65152165c52fde9ce.png" class="math-inline" /> が一意に決まる。

__(2) 具体例のCantor標準形__

**(a) <img src="tmp/d4c9e3a4805dfa3c9cfa82d709a25121.png" class="math-inline" />**

- <img src="tmp/7ab2b0361eb0c397b17d89861853e25c.png" class="math-inline" />

**(b) <img src="tmp/838230f39aa4893752cb4d57f76fdc99.png" class="math-inline" />**

- <img src="tmp/917d2ac41ff829e49bd00cb750a3fedf.png" class="math-inline" />

**(c) <img src="tmp/315c9a286d4388b8da5d8311523c7372.png" class="math-inline" />**

- <img src="tmp/6a6598390ee71c7038bbd24b037445e9.png" class="math-inline" />

__問9（超限帰納法・超限再帰）__

__(1) 超限帰納法__

**(a) 2条件での主張**

> 命題 <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> が次を満たすなら、すべての順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> が成り立つ。
>
> 1. 基底：<img src="tmp/b27dacc5676ded5b21aa21ada36bd16a.png" class="math-inline" /> が成り立つ。
> 2. 帰納段階：任意の <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について、「すべての <img src="tmp/8b343c5abba58f89bdc01954701dd97e.png" class="math-inline" /> で <img src="tmp/3be9b85cf782a26b88a0cfd353bdaf69.png" class="math-inline" /> が成り立つなら <img src="tmp/834a1c7533982756c9289e3338d3f04c.png" class="math-inline" /> も成り立つ」ことを示す。

**(b) 3段階の形**

> 1. 基底：<img src="tmp/b27dacc5676ded5b21aa21ada36bd16a.png" class="math-inline" /> を示す。
> 2. 後続段階：任意の <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> について、<img src="tmp/3be9b85cf782a26b88a0cfd353bdaf69.png" class="math-inline" /> ならば <img src="tmp/d3956bde88781497b0ed0243360459f1.png" class="math-inline" /> を示す。
> 3. 極限段階：任意の極限順序数 <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> について、「すべての <img src="tmp/054f7f0b6dbff27340a67da24adf06b9.png" class="math-inline" /> で <img src="tmp/3be9b85cf782a26b88a0cfd353bdaf69.png" class="math-inline" /> が成り立つなら <img src="tmp/d2d07d491ce308b6749dfc422f75d6b4.png" class="math-inline" /> も成り立つ」ことを示す。

**(c) 3段階で十分な理由（直感的説明）**

- 任意の 0 でない順序数は後続か極限のどちらか。
- 基底から出発し、後続段階で「+1」を繰り返し、極限段階で「より小さい順序数全体から極限順序数へ」のジャンプを保証することで、すべての順序数に命題が伝播する。

__(2) 「すべての順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は推移的集合である」の証明方針__

- <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> に関する超限帰納法を用いる。
- 基底：<img src="tmp/1697cff35554c385d83b81d62555f6cc.png" class="math-inline" />（空集合）は推移的。
- 帰納段階：任意の <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> について、「すべての <img src="tmp/8b343c5abba58f89bdc01954701dd97e.png" class="math-inline" /> で <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> は推移的」と仮定し、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が推移的であることを示す。
  - <img src="tmp/2da090afd11b5685ecd7b755d8b4e669.png" class="math-inline" /> とする。
  - <img src="tmp/b4d58078d6159ee352ef89f89ce6e36c.png" class="math-inline" /> より帰納仮定から <img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> は推移的：<img src="tmp/b3ac70357fd167bc08c071acb37e59a8.png" class="math-inline" /> かつ <img src="tmp/d10ac4b712413caad40a38dd7f5415f6.png" class="math-inline" /> なら <img src="tmp/6ae7746013b4453fc4b0cd84c8f2627b.png" class="math-inline" />。
- よってすべての <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> で <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は推移的。

__(3) 順序数の和 <img src="tmp/523ee0a625bb9be4d02ba6cf448d12be.png" class="math-inline" /> の超限再帰による定義__

- 基底：<img src="tmp/c241ee17471fbbd5d3fee092d46a3e52.png" class="math-inline" />
- 後続段階：<img src="tmp/70de2424c935a6244f37f22c6107f26e.png" class="math-inline" />
- 極限段階：<img src="tmp/c7ceb93e486df346ebfd4b0e77996ece.png" class="math-inline" /> が極限順序数のとき、<img src="tmp/6796a8d94c4ee9366e9cba2943156eb5.png" class="math-inline" />

__問10（整列集合の比較定理）__

**命題**：2つの整列集合 <img src="tmp/da0bb8eef72f9bf9e8c276487b64d8a8.png" class="math-inline" /> について、次のいずれか一方が成り立つ。

1. <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> のある始片と順序同型である。
2. <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> のある始片と順序同型である。
3. <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> は順序同型である。

**証明の概略（超限帰納法による）**

- 超限再帰により、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の元を「同時に」比較する順序同型写像を構成する。
- <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の最小元 <img src="tmp/fa90aad2c4990b7bc44434aef9fe632c.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の最小元 <img src="tmp/9a4fce870f8a707de5a0ca48723f8ba6.png" class="math-inline" /> を対応させる。
- 帰納的に、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の始片 <img src="tmp/6f043dd1382258dbd297c585121b9c88.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の始片 <img src="tmp/17138bd63b6eb0bf6297b1e10f556ea1.png" class="math-inline" /> が順序同型であると仮定し、次のように進める：
  - もし <img src="tmp/6f043dd1382258dbd297c585121b9c88.png" class="math-inline" /> と <img src="tmp/17138bd63b6eb0bf6297b1e10f556ea1.png" class="math-inline" /> が順序同型なら、<img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> と <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> を対応させる。
  - もし <img src="tmp/6f043dd1382258dbd297c585121b9c88.png" class="math-inline" /> が <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の真の始片と順序同型なら、<img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> 側にまだ対応していない最小元を <img src="tmp/c5b4a99c4189d1f1ec6e04ddc52abad8.png" class="math-inline" /> に対応させる。
  - もし <img src="tmp/17138bd63b6eb0bf6297b1e10f556ea1.png" class="math-inline" /> が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の真の始片と順序同型なら、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> 側にまだ対応していない最小元を <img src="tmp/857896a1c38d27a6d6f636afa1f05ce5.png" class="math-inline" /> に対応させる。
- この構成を続けると、次のいずれかが起こる：
  - <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> 全体が <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の始片と順序同型になる（場合1）
  - <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> 全体が <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の始片と順序同型になる（場合2）
  - <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> 全体が順序同型になる（場合3）
- いずれも起こらないと仮定すると、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> のどちらにも対応していない元が残り続けるが、整列性からそのような元の最小元が存在し、帰納法のステップで必ず対応が進むため矛盾。
- よって3つの場合のいずれかが成り立つ。



<div style="page-break-before:always"></div>




# 基数と濃度

## 基数と濃度

集合論における「基数（cardinal number）」と「濃度（cardinality）」は、集合の「大きさ」を表す概念です。多くの文脈ではほぼ同じ意味で使われますが、厳密には次のように整理できます。

### 1. 濃度（cardinality）の定義

__1.1 集合の「大きさ」の直観__
- 有限集合なら「要素の個数」で大きさを表せます。
- 無限集合でも、たとえば自然数全体の集合と偶数全体の集合は「同じ大きさ」とみなしたい（全単射が存在する）という直観があります。
- この「大きさ」を一般化したのが濃度です。

__1.2 濃度の形式的な定義（公理的集合論）__

公理的集合論（ZFCなど）では、次のように定義されることが多いです。

- 集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の濃度を <img src="tmp/efa403df45c1ea69e5464fa35c14815a.png" class="math-inline" /> と書きます。
- 集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> について、「<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の間に全単射が存在する」とき、<img src="tmp/46c4dd564b86222198c1bd3d0f7100f9.png" class="math-inline" /> と定義します。
- 濃度は「全単射が存在する」という同値関係による同値類として定義されます。

より厳密には、選択公理（AC）を仮定すると、任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に対して  
「<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と全単射で結びつく最小の順序数」を <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**基数（cardinal number）** と定義し、それを <img src="tmp/efa403df45c1ea69e5464fa35c14815a.png" class="math-inline" /> と書きます。

### 2. 基数（cardinal number）の定義

__2.1 基数とは__
- 基数とは、「濃度を表す順序数」のことです。
- 順序数は「整列順序の型」を表す数ですが、その中で「自分より小さい順序数との間に全単射が存在しない」ものを**基数**と呼びます。
- 例：
  - 有限順序数 <img src="tmp/18991571b007a883e99c3893b3740bdf.png" class="math-inline" /> はすべて基数です。
  - 最小の無限順序数 <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は基数であり、<img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> と書きます（可算無限集合の濃度）。

__2.2 基数の記号__
- 有限基数：<img src="tmp/18991571b007a883e99c3893b3740bdf.png" class="math-inline" />
- 可算無限基数：<img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />
- その次の無限基数：<img src="tmp/d706757d0f16c9f1c8d7fa027ccabc22.png" class="math-inline" />（アレフ数）

### 3. 濃度と基数の関係

- **濃度**：集合の「大きさ」そのもの（同値類としての概念）。
- **基数**：その大きさを表す「代表元」として選ばれた順序数。
- 多くの場合、同じ意味で使われますが、厳密には
  - 「濃度」は同値類
  - 「基数」はその同値類の代表として選ばれた順序数
  という関係です。

### 4. 濃度の比較（大小関係）

__4.1 定義__
- <img src="tmp/efdb956bee5d8baccd0fee7507e1ade5.png" class="math-inline" /> とは、「<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への単射が存在する」ことと定義します。
- <img src="tmp/cdf89d42f0059ea8c63230b802884b42.png" class="math-inline" /> とは、「<img src="tmp/efdb956bee5d8baccd0fee7507e1ade5.png" class="math-inline" /> かつ <img src="tmp/9e6bd43e02ea4b6ecd6932084ac781de.png" class="math-inline" />」と定義します。

__4.2 カントールの定理__
任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> について、


<div class="math-display-container"><img src="tmp/6f4ba8a962e85747bb740b4402a400da.png" class="math-display" /></div>


が成り立ちます（<img src="tmp/e1fa47c9075271d02cf8ead81516d053.png" class="math-inline" /> は <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の冪集合）。  
特に、無限集合には「より大きい無限集合」が必ず存在します。

### 5. 主な濃度の例

__5.1 有限濃度__
- <img src="tmp/0d25d178b7971239fb0d782eb9be6877.png" class="math-inline" />（自然数） ⇔ <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> 個の要素を持つ有限集合。

__5.2 可算無限濃度__
- <img src="tmp/05bedd17488afc9fbca66cd269648904.png" class="math-inline" />
- 自然数と全単射がつくれる集合を「可算無限集合」といいます。

__5.3 連続体濃度__
- <img src="tmp/d9da9ae90b562b96eb6102d2021c5205.png" class="math-inline" />
- これを連続体濃度と呼び、<img src="tmp/0a2f3f25776381173161ff6d7ffeb331.png" class="math-inline" /> と書くこともあります。

### 6. 濃度の演算

基数 <img src="tmp/dac3488f64320b5da038ecb547d823d5.png" class="math-inline" /> に対して、次のように定義します（無限基数の場合も含む）。

- **和**：<img src="tmp/9a03a4d7e1d4f4beb52d787432480c5a.png" class="math-inline" />（ただし <img src="tmp/d171287a19b00490e9b6a6a54cb247a5.png" class="math-inline" /> で <img src="tmp/e8712fbc84c6c78d3ae9e3499b3a8c0d.png" class="math-inline" />）
- **積**：<img src="tmp/561cec90a3aff9695a2fb9abb8e77cef.png" class="math-inline" />
- **冪**：<img src="tmp/e8c29e5a77ac22eb8a8e9f62e7a59179.png" class="math-inline" />（ただし <img src="tmp/e8712fbc84c6c78d3ae9e3499b3a8c0d.png" class="math-inline" />）

__6.1 無限基数の演算の性質__
- <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が無限基数のとき：
  - <img src="tmp/b73ae02fdff40cf0b76cf1ea71ed600c.png" class="math-inline" />
  - 特に <img src="tmp/6da79c35109481028b9fc81ae10d22d3.png" class="math-inline" />、<img src="tmp/98a18778764aec07402f1633cc912131.png" class="math-inline" /> など。
- 冪演算は一般に異なる値を取り得ます（例：<img src="tmp/962a228ac52da28bd482d9bc2a0db055.png" class="math-inline" />）。

### 7. 連続体仮説（CH）と一般連続体仮説（GCH）

- **連続体仮説**：<img src="tmp/0ed95fbd8a8f7e23c6c080c57a0da502.png" class="math-inline" /> かどうかという問題。
- **一般連続体仮説**：任意の基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> について <img src="tmp/631c186a3761449347f25a493dd7bf93.png" class="math-inline" />（<img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> の次の基数）かどうかという問題。
- これらは ZFC から独立であることが知られています（証明も反証もできない）。


## 基数の例

基数（cardinal number）の例を、有限・可算無限・非可算無限の順に挙げます。

### 1. 有限基数の例

有限集合の要素の個数として現れる基数です。

- <img src="tmp/7b35bc7c79f25379b8086822d9272cad.png" class="math-inline" />
- <img src="tmp/5d8ce2b90827c60262db122b99f2e6b0.png" class="math-inline" />
- <img src="tmp/38318a329852b740797c64198e02682b.png" class="math-inline" />
- 一般に、<img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> 個の要素からなる有限集合の濃度は <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" />（自然数）です。

### 2. 可算無限基数の例

自然数全体の集合 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と全単射が存在する集合の濃度を「可算無限濃度」といい、その基数を <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> と書きます。

- <img src="tmp/024d16c0e3fdfd485f9bd70c7df42ce1.png" class="math-inline" />
- <img src="tmp/16f357ade8b2efc3817912afdde5200d.png" class="math-inline" />（整数全体）
- <img src="tmp/b5baee36161f2ea853c4241a02cba91e.png" class="math-inline" />（有理数全体）
- <img src="tmp/1b5c7af1d4a97eec9144ed953a3748b2.png" class="math-inline" />（自然数の組全体）
- 偶数全体、素数全体、整数係数の多項式全体なども、いずれも <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />

これらはすべて「自然数と 1 対 1 に対応づけられる」という意味で同じ大きさです。

### 3. 非可算無限基数の例

自然数と全単射が作れない無限集合の濃度です。

__3.1 連続体濃度（<img src="tmp/107dd2dfd0c9f57f499e47a77ddaaeff.png" class="math-inline" />）__

- <img src="tmp/8c36b63e00b43179429583544530996a.png" class="math-inline" />（実数全体）
- <img src="tmp/1f450effe6d74b280f691a53dabe396e.png" class="math-inline" />（開区間）
- <img src="tmp/f55d0b50760106e9e76b5cdc08a5147d.png" class="math-inline" />（自然数の部分集合全体）
- 複素数全体 <img src="tmp/1d8ddd3aadaa5a36ac8588548645f46c.png" class="math-inline" /> も同じ濃度です。

__3.2 より大きな無限基数__

- <img src="tmp/3c78bb2d609d7cbd614f9e45d7054aed.png" class="math-inline" />（実数の部分集合全体）
- 一般に、集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に対して <img src="tmp/f418a8158959d25022ee3b4dea050b06.png" class="math-inline" /> であり、カントールの定理により
  

<div class="math-display-container"><img src="tmp/fe613872977f9bb9da4b6a0daa639b3c.png" class="math-display" /></div>


  が成り立ちます。

したがって、無限基数の列


<div class="math-display-container"><img src="tmp/b42e439a1b003ce6d9e1ceb7562c4b9c.png" class="math-display" /></div>


はどんどん大きくなっていきます。

### 4. アレフ数（<img src="tmp/b119e30f90d20c190b96e80e81269deb.png" class="math-inline" />）としての基数

順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> に対して、<img src="tmp/b119e30f90d20c190b96e80e81269deb.png" class="math-inline" /> という無限基数が定義されます。

- <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />：可算無限基数
- <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" />：<img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> の「次の」基数（最小の非可算基数）
- <img src="tmp/fc26b6e98f20e9c587c0356a460acde5.png" class="math-inline" />：その次の基数
- 一般に、<img src="tmp/b119e30f90d20c190b96e80e81269deb.png" class="math-inline" /> は「それより小さい基数との間に全単射が存在しない」最小の順序数として定義されます。

連続体仮説（CH）は


<div class="math-display-container"><img src="tmp/f191da77b76af753e9dab5e90d44d499.png" class="math-display" /></div>


が成り立つかどうかという問題ですが、これは ZFC から独立しています。

## 特異基数と正則基数

以下、「特異基数」と「正則基数」について、数学的な定義と特徴を説明します。

### 1. 正則基数（regular cardinal）

__1.1 定義__

基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が**正則（regular）** であるとは、次の条件を満たすことをいいます。

> 任意の集合族 <img src="tmp/da5f053c957a98144578e65e9806d498.png" class="math-inline" /> について、
> - 各 <img src="tmp/dbf44aed127618c2c4ff460c58fbe451.png" class="math-inline" /> の濃度が <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> 未満（<img src="tmp/16a2f93a69d2a258e0b0fcae008e598f.png" class="math-inline" />）
> - 添字集合の濃度も <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> 未満（<img src="tmp/9fe347a99d291aae5aa8abeb1ba57713.png" class="math-inline" />） 
>ならば、その和集合の濃度も <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> 未満である：


<div class="math-display-container"><img src="tmp/f3eaf61c9963afd37c20381fa95e1661.png" class="math-display" /></div>


>

言い換えると：

- 「小さな集合（濃度 <img src="tmp/2f15c14ca3eecf0850f2f62690a7db43.png" class="math-inline" />）を、それより小さい個数（<img src="tmp/2f15c14ca3eecf0850f2f62690a7db43.png" class="math-inline" /> 個）だけ集めても、全体の大きさは <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> に達しない」

という性質です。

__1.2 同値な特徴づけ（共終数を用いた定義）__

基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> の**共終数（cofinality）** を <img src="tmp/b9cf278894fc6c4ce31c542b04c5f35e.png" class="math-inline" /> と書きます。これは

> <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> に共終な（cofinal）増加列の最小の長さ

として定義されます。

このとき、次が成り立ちます。

- <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が正則 ⇔ <img src="tmp/16b320dcba1532ca2f10b404aeb538d0.png" class="math-inline" />
- <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が特異（後述） ⇔ <img src="tmp/b2bc86da093198b182f40a3bc677b0c7.png" class="math-inline" />

__1.3 正則基数の例__

- すべての**有限基数**は正則です。
- 最小の無限基数 <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> は正則です。
- 後続基数（successor cardinal）は常に正則です：
  - <img src="tmp/a3ab42c1c7f593acefbf4f4e224f1485.png" class="math-inline" /> はすべて正則。
- 到達不能基数（inaccessible cardinal）も正則です。

__1.4 正則基数の性質__

- 正則基数は「分解しにくい」基数です。
- 順序数として見ると、「自分より小さい順序数の列で、極限として自分自身に到達できない」という性質を持ちます。
- 正則基数は、多くの組合せ論的・モデル論的性質（木性質、分割性質など）を持つことが知られています。

### 2. 特異基数（singular cardinal）

__2.1 定義__

基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が**特異（singular）** であるとは、**正則でない**こと、すなわち次が成り立つことをいいます。

> ある集合族 <img src="tmp/da5f053c957a98144578e65e9806d498.png" class="math-inline" /> が存在して、
> - 各 <img src="tmp/dbf44aed127618c2c4ff460c58fbe451.png" class="math-inline" /> の濃度が <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> 未満（<img src="tmp/16a2f93a69d2a258e0b0fcae008e598f.png" class="math-inline" />）
> - 添字集合の濃度も <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> 未満（<img src="tmp/9fe347a99d291aae5aa8abeb1ba57713.png" class="math-inline" />）
> 
> であるにもかかわらず、和集合の濃度が <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> に等しい：


<div class="math-display-container"><img src="tmp/3e4efba861147e0ae27b546d7d1b91cd.png" class="math-display" /></div>


>

共終数を用いると：

- <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が特異 ⇔ <img src="tmp/b2bc86da093198b182f40a3bc677b0c7.png" class="math-inline" />

__2.2 特異基数の例__

- <img src="tmp/0a3158deee2d8f878a0ea1096c9c8133.png" class="math-inline" /> は特異基数です。
  - 各 <img src="tmp/62558454c80164b304386456e52ea11a.png" class="math-inline" /> は <img src="tmp/1741bcd4d659d9931bdd7fa89a4205cf.png" class="math-inline" /> より小さい基数。
  - 可算無限個（<img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> 個）の和集合として <img src="tmp/1741bcd4d659d9931bdd7fa89a4205cf.png" class="math-inline" /> に到達する。
  - よって <img src="tmp/8e2ffce12f50c5c716b8226cb89deca6.png" class="math-inline" />。
- 一般に、極限順序数 <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> に対して <img src="tmp/6b782f7612a189fdff1510a1f1aa30eb.png" class="math-inline" /> が特異になることがあります。
  - 例：<img src="tmp/2fcc6c5c6f24a67c100b82934f1821b3.png" class="math-inline" /> など。

__2.3 特異基数の性質__

- 特異基数は「小さな基数の集まりとして表現できる」基数です。
- 特異基数仮説（Singular Cardinals Hypothesis, SCH）など、特異基数に関する組合せ論的性質は集合論の重要な研究対象です。
- 特異基数は、正則基数に比べて「分解可能」であり、その振る舞いはしばしばより複雑です。

### 3. 正則基数と特異基数の関係

__3.1 基本的な関係__

- 任意の無限基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> について、
  - <img src="tmp/b9cf278894fc6c4ce31c542b04c5f35e.png" class="math-inline" /> は常に正則基数です。
  - 特に <img src="tmp/b9cf278894fc6c4ce31c542b04c5f35e.png" class="math-inline" /> は <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> 以下の正則基数です。
- したがって、
  - <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が正則 ⇔ <img src="tmp/16b320dcba1532ca2f10b404aeb538d0.png" class="math-inline" />
  - <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が特異 ⇔ <img src="tmp/b2bc86da093198b182f40a3bc677b0c7.png" class="math-inline" />

__3.2 後続基数と極限基数__

- **後続基数**：ある基数 <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> に対して <img src="tmp/b88abfe0306b23f7ecf66a17a634a375.png" class="math-inline" /> と書ける基数。
  - 例：<img src="tmp/a5551495649f9d588dd7cf8ac0a6c918.png" class="math-inline" />
  - 後続基数は常に正則です。
- **極限基数**：後続基数でない無限基数。
  - 例：<img src="tmp/d409e2ea9c6fb22d9fd7ddee9fdcf3e4.png" class="math-inline" /> など。
  - 極限基数は正則であることも特異であることもあります。

__3.3 正則・特異の判定の直感的イメージ__

- 正則基数：
  - 「階段を一段ずつ上がっていっても、有限ステップや可算ステップでは頂上に達しない」
  - 例：<img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" /> は可算個の可算集合の和では到達できない。
- 特異基数：
  - 「より小さい基数を、それより小さい個数だけ集めると、自分自身に到達してしまう」
  - 例：<img src="tmp/1741bcd4d659d9931bdd7fa89a4205cf.png" class="math-inline" /> は可算個の <img src="tmp/62558454c80164b304386456e52ea11a.png" class="math-inline" /> の和として得られる。

## 基数の演算

以下、基数の演算について説明します。

### 1. 基数の演算の定義

基数 <img src="tmp/dac3488f64320b5da038ecb547d823d5.png" class="math-inline" /> に対し、次の3つの演算が定義されます。

__1.1 和（加法）__

**定義**：  
<img src="tmp/35a9cf0bd63ee92eb268afcbe9a7a7e7.png" class="math-inline" /> を互いに素な集合とし、<img src="tmp/fde2b8286f959f48f85b32c26e7f4434.png" class="math-inline" /> とする。このとき、



<div class="math-display-container"><img src="tmp/a56dfa6f138ee2da1cba85a3ac73190c.png" class="math-display" /></div>



と定義する。

- 互いに素でない場合も、<img src="tmp/521bdc5757f82815854354b3e905daab.png" class="math-inline" /> のようにして互いに素な集合に取り替えればよい。
- 有限基数の場合は通常の自然数の和と一致します。

__1.2 積（乗法）__

**定義**：  
<img src="tmp/fde2b8286f959f48f85b32c26e7f4434.png" class="math-inline" /> とする。このとき、



<div class="math-display-container"><img src="tmp/3f949513d15e5144f75e5817403d819d.png" class="math-display" /></div>



と定義する。

- 直積集合の要素数として定義されます。
- 有限基数の場合は通常の自然数の積と一致します。

__1.3 冪（べき乗）__

**定義**：  
<img src="tmp/fde2b8286f959f48f85b32c26e7f4434.png" class="math-inline" /> とする。このとき、



<div class="math-display-container"><img src="tmp/f205624a3ce9f694ab7bba7120727672.png" class="math-display" /></div>



と定義する。

- すなわち、<img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> から <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> への写像全体の集合の濃度です。
- 有限基数の場合、<img src="tmp/a9698c8745845ba1cf83104d2e1759c4.png" class="math-inline" /> は通常の自然数のべき乗と一致します。

### 2. 有限基数の演算

有限基数（自然数）については、通常の算術と一致します。

- <img src="tmp/bc3611a55c75ae6fd35c3d8f34ceccf5.png" class="math-inline" />：要素数 <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> の集合と <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> の集合の和集合の要素数。
- <img src="tmp/5698b8836e9c16ffc13f910dd9649b5b.png" class="math-inline" />：直積集合の要素数。
- <img src="tmp/a5a61c9026ff18134f478acfb12ea6d9.png" class="math-inline" />：<img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> 元集合から <img src="tmp/0289e2224e153179243fd7c6715722f2.png" class="math-inline" /> 元集合への写像の個数。

### 3. 無限基数の演算の性質

__3.1 和と積の簡約法則__

<img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> を無限基数とすると、任意の基数 <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> に対し、次が成り立ちます。

1. **和**：
   

<div class="math-display-container"><img src="tmp/e4aa2b45f91704d3950c7acec8ac6209.png" class="math-display" /></div>



2. **積**：
   

<div class="math-display-container"><img src="tmp/f6e40209569b2a00a08e57beb45e61eb.png" class="math-display" /></div>



特に：

- <img src="tmp/6da79c35109481028b9fc81ae10d22d3.png" class="math-inline" />（<img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> は有限）
- <img src="tmp/98a18778764aec07402f1633cc912131.png" class="math-inline" />
- <img src="tmp/7b10aaad6c4308f1b07b55389d07f00a.png" class="math-inline" />
- <img src="tmp/edc51c4285f0025b04264fa864297ad9.png" class="math-inline" />
- <img src="tmp/4215f6e4170dbcde3112c01fb061b773.png" class="math-inline" />

**直感的な理由**：  
無限集合は「有限個や可算個の要素を加えたり、有限個や可算個のコピーを取ったりしても、濃度は変わらない」からです。

__3.2 冪演算の性質__

冪演算は一般に和・積とは異なり、単純な <img src="tmp/228f9fe07136d3e298ef23396170984a.png" class="math-inline" /> にはなりません。

- <img src="tmp/962a228ac52da28bd482d9bc2a0db055.png" class="math-inline" />（カントールの定理）
- 一般に、<img src="tmp/8010cd1166d99a382a95919d6536a86c.png" class="math-inline" />（カントールの定理）

また、次の指数法則が成り立ちます（基数としての演算）。

1. **積の冪**：
   

<div class="math-display-container"><img src="tmp/b9ecc155c32a7565f7cb860a2d6e598f.png" class="math-display" /></div>



2. **冪の冪**：
   

<div class="math-display-container"><img src="tmp/cfa223f0540899a9838f5d3a5f32d9b0.png" class="math-display" /></div>



3. **積の冪（一般には成り立たない）**：
   

<div class="math-display-container"><img src="tmp/86106883b7d9dbce433d2953d11727c6.png" class="math-display" /></div>


   は一般には成り立ちません（有限のときは成り立つが、無限では反例あり）。

### 4. 具体例

__4.1 可算無限基数 <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> の演算__

- <img src="tmp/6da79c35109481028b9fc81ae10d22d3.png" class="math-inline" />
- <img src="tmp/98a18778764aec07402f1633cc912131.png" class="math-inline" />
- <img src="tmp/911193af61960dfe0eedcc94bd0b506d.png" class="math-inline" />（<img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> 有限、<img src="tmp/d4382038c32cbadc914a8eb3de5cb029.png" class="math-inline" />）
- <img src="tmp/7b10aaad6c4308f1b07b55389d07f00a.png" class="math-inline" />
- <img src="tmp/b3bcfda33f3c369d48982b072bf2da0a.png" class="math-inline" />（<img src="tmp/62ad16c68b793ce369d8f8ac642f5f7a.png" class="math-inline" /> のとき）
- <img src="tmp/391ef2cd0f5dcbd1fbc2d00353d2bfa3.png" class="math-inline" />

__4.2 連続体濃度 <img src="tmp/107dd2dfd0c9f57f499e47a77ddaaeff.png" class="math-inline" /> の演算__

- <img src="tmp/f735987e74799fd20406818d2e3bd30c.png" class="math-inline" />
- <img src="tmp/09eaaa89c867609cf87528aed7029c9d.png" class="math-inline" />
- <img src="tmp/0bdf8a317862b23f75402cbbd0dd05e6.png" class="math-inline" />
- <img src="tmp/ac4aa03cbcacc50a87da61c07951f94e.png" class="math-inline" />

### 5. 基数演算と順序数演算の違い

- **順序数演算**：整列順序の「型」を表す演算（和は連結、積は辞書式順序など）。
- **基数演算**：集合の「大きさ（濃度）」のみを考える演算。

例：

- 順序数の和：<img src="tmp/505cae9ada1773332693d39441136fac.png" class="math-inline" />
- 基数の和：<img src="tmp/e6fe1b31bce9d4a0010e6b0abc3f2822.png" class="math-inline" />

基数演算では順序の情報を忘れるため、非可換性などは現れません（和・積は可換）。

## 連続体仮説

以下、「連続体仮説（Continuum Hypothesis, CH）」について、数学的な定義と特徴を説明します。

### 1. 連続体仮説の定義

__1.1 連続体濃度__

- 自然数全体の集合 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> の濃度を**可算無限濃度**といい、その基数を <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> と書きます。
- 実数全体の集合 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の濃度を**連続体濃度（continuum）** といい、<img src="tmp/8c36b63e00b43179429583544530996a.png" class="math-inline" /> と書きます。

__1.2 連続体仮説（CH）の主張__

**連続体仮説**とは、次の命題のことです。

> 連続体濃度 <img src="tmp/107dd2dfd0c9f57f499e47a77ddaaeff.png" class="math-inline" /> は、可算無限濃度 <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> の**直後の基数**である：


<div class="math-display-container"><img src="tmp/ded0c2085051e24cfeb2cda17c2734cb.png" class="math-display" /></div>




ここで <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" /> は、<img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> より大きい最小の基数（最小の非可算基数）を表します。

### 2. 一般連続体仮説（GCH）

**一般連続体仮説（Generalized Continuum Hypothesis, GCH）** は、CH を任意の基数に拡張したものです。

> 任意の無限基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> について、


<div class="math-display-container"><img src="tmp/d84154f59056f72e907e168c0077219b.png" class="math-display" /></div>


> が成り立つ。

ここで <img src="tmp/45b3b3edee9ec542013fb165f6941145.png" class="math-inline" /> は <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> の**次の基数（successor cardinal）** を表します。

- CH は GCH の <img src="tmp/68b2c0eab8af7a77c1683b4a35cacc3d.png" class="math-inline" /> の場合に相当します。

### 3. 連続体仮説の特徴

__3.1 歴史的背景__

- カントールは、実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> が自然数 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> より「真に大きい」ことを示しました（対角線論法）。
- さらに彼は、「<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の濃度は <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> の次の基数 <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" /> に等しいのではないか」と予想しました。これが連続体仮説です。
- 長い間、CH は真か偽かが未解決の問題でした。

__3.2 ZFC からの独立性__

- ゲーデル（1938）は、ZFC が無矛盾ならば **ZFC + CH** も無矛盾であることを示しました（構成可能宇宙 <img src="tmp/f89f6967a442874733ff0c94b75fc330.png" class="math-inline" /> において CH が成り立つ）。
- コーエン（1963）は、強制法（forcing）を用いて、ZFC が無矛盾ならば **ZFC + ¬CH** も無矛盾であることを示しました。
- したがって、CH は **ZFC から独立**（証明も反証もできない）であることがわかります。

同様に、GCH も ZFC から独立であることが知られています。

__3.3 集合論における位置づけ__

- CH は ZFC の**追加公理**として採用することも、採用しないこともできます。
- 集合論の多くの結果は「ZFC + CH」や「ZFC + ¬CH」のどちらを仮定するかによって異なります。
- 実解析・記述集合論・測度論など、実数に関わる分野では、CH の仮定の有無が定理の成立に影響することがあります。

### 4. CH が成り立つ世界と成り立たない世界

__4.1 CH が成り立つ世界（<img src="tmp/0ed95fbd8a8f7e23c6c080c57a0da502.png" class="math-inline" />）__

- 実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の濃度は最小の非可算基数 <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" /> に等しい。
- 実数の部分集合の「サイズ」の種類が比較的少ない：
  - 可算集合
  - 連続体濃度の集合
- 実数上の測度・カテゴリーに関するいくつかの命題が単純化されることがあります。

__4.2 CH が成り立たない世界（<img src="tmp/6a3b3e99e4f40db634853925ed7a0c52.png" class="math-inline" />）__

- 実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の濃度は <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" /> より大きい（例：<img src="tmp/7970b1c8baa9dce423cb2a19ca37a8c8.png" class="math-inline" /> など）。
- 実数の部分集合の「中間的なサイズ」が存在しうる：
  - 濃度が <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> と <img src="tmp/107dd2dfd0c9f57f499e47a77ddaaeff.png" class="math-inline" /> の間にある集合が存在しうる（ただし、これはさらに別の公理に依存）。
- 強制法を用いて、さまざまな <img src="tmp/107dd2dfd0c9f57f499e47a77ddaaeff.png" class="math-inline" /> の値を実現するモデルを構成できます。

### 5. CH と関連する概念

__5.1 ベール空間・完全集合性質__

- 実数上の「ベールのカテゴリー定理」や「完全集合性質」などは、CH の仮定の有無によって性質が変わることがあります。
- 例えば、CH を仮定すると「非可算な実数部分集合で、完全集合を含まないもの」が存在しうるが、CH を否定する公理系ではそのような集合が存在しないこともあります。

__5.2 巨大基数との関係__

- 巨大基数公理を仮定すると、実数集合論の構造がより「整った」ものになり、CH の否定と整合的なモデルが構成しやすくなることが知られています。
- しかし、巨大基数公理そのものは CH を決定しません。

## 演習

以下に、「基数と濃度」に関する演習問題を出題します。  


### 問題


__問1（濃度と基数の定義）__

(1) 集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**濃度** <img src="tmp/efa403df45c1ea69e5464fa35c14815a.png" class="math-inline" /> を、全単射を用いて定義せよ。

(2) 公理的集合論（ZFC）において、**基数（cardinal number）** を「順序数」を用いてどのように定義するか説明せよ。

(3) 濃度と基数の関係を、「同値類」と「代表元」という観点から説明せよ。

__問2（濃度の比較）__

(1) 濃度の大小関係 <img src="tmp/efdb956bee5d8baccd0fee7507e1ade5.png" class="math-inline" /> を、単射を用いて定義せよ。

(2) カントールの定理

> 任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> について <img src="tmp/d9b1e3181ce16e869107e5da64c576d3.png" class="math-inline" />

を証明せよ。

(3) 次の集合の濃度の大小関係を比較せよ（理由も簡潔に述べよ）。

- <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" />（自然数全体）
- <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />（整数全体）
- <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" />（有理数全体）
- <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" />（実数全体）
- <img src="tmp/ebf25b819f989f3f4adb855a4d1293c6.png" class="math-inline" />（自然数の部分集合全体）

__問3（可算無限集合）__

(1) 集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が**可算無限集合**であるとはどういうことか、定義せよ。

(2) 次の集合が可算無限集合であることを、全単射を具体的に構成するか、または可算集合の性質を用いて示せ。

- (a) 偶数全体の集合 <img src="tmp/2a9a74be1b88a507bca22ff741b1ee93.png" class="math-inline" />
- (b) 整数全体 <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />
- (c) 有理数全体 <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" />
- (d) 整数係数の多項式全体 <img src="tmp/61cf8b9404384d463cf04e8aaaa97c9d.png" class="math-inline" />

(3) 可算無限集合の部分集合は、有限集合または可算無限集合であることを示せ。

__問4（非可算無限集合）__

(1) 実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> が非可算無限集合であることを、カントールの対角線論法を用いて証明せよ。

(2) 開区間 <img src="tmp/039474a1a493fb3c2d8f4adbf24f971e.png" class="math-inline" /> の濃度が <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の濃度と等しいことを示せ。

(3) 次の集合の濃度が <img src="tmp/107dd2dfd0c9f57f499e47a77ddaaeff.png" class="math-inline" />（連続体濃度）であることを示せ。

- (a) 閉区間 <img src="tmp/171d58b8063a24dc65a33bacb46a5fb7.png" class="math-inline" />
- (b) 複素数全体 <img src="tmp/1d8ddd3aadaa5a36ac8588548645f46c.png" class="math-inline" />
- (c) 実数列全体 <img src="tmp/226899de81ec35f904a2efff77891b5c.png" class="math-inline" />

__問5（基数の演算）__

(1) 基数 <img src="tmp/78c7c4b30c2d9733b3cfbf415205f302.png" class="math-inline" /> に対し、次の演�を定義せよ。

- (a) 和 <img src="tmp/6de08965a1aed40bf5ce6b6833e21300.png" class="math-inline" />
- (b) 積 <img src="tmp/1a05b84869df7c0c4587f6b76b511b4d.png" class="math-inline" />
- (c) 冪 <img src="tmp/0166726d18abff61b5af44c5ecf01fa5.png" class="math-inline" />

(2) 次の等式が成り立つかどうかを判定し、理由を簡潔に述べよ。

- (a) <img src="tmp/457a57457068092bbbe36a32c9fbd018.png" class="math-inline" />
- (b) <img src="tmp/98a18778764aec07402f1633cc912131.png" class="math-inline" />
- (c) <img src="tmp/7b10aaad6c4308f1b07b55389d07f00a.png" class="math-inline" />
- (d) <img src="tmp/baa1dd28be7ebca28783c2bfe2cee036.png" class="math-inline" />

(3) 無限基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> に対し、<img src="tmp/2e66da77e1bc36cbdba2e0966a29a5b1.png" class="math-inline" /> が成り立つことを示せ（<img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> は任意の基数）。

__問6（アレフ数と連続体仮説）__

(1) アレフ数 <img src="tmp/b119e30f90d20c190b96e80e81269deb.png" class="math-inline" />（<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は順序数）の定義を述べよ。

(2) 次の問いに答えよ。

- (a) <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> はどのような集合の濃度か。
- (b) <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" /> はどのような性質を持つ基数か。
- (c) <img src="tmp/fc26b6e98f20e9c587c0356a460acde5.png" class="math-inline" /> は <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" /> とどのような関係にあるか。

(3) **連続体仮説（CH）** と **一般連続体仮説（GCH）** の内容を述べ、その独立性について知っていることを説明せよ。

__問7（濃度の同値類としての扱い）__

(1) 集合の族 <img src="tmp/9db31d506ef7050c54b936d05cf1b6b5.png" class="math-inline" /> が given されたとき、これらを「濃度が等しい」という同値関係で類別した商集合を考えたい。  
この同値関係が well-defined であることを示せ。

(2) 濃度の大小関係 <img src="tmp/63dbf201871a52e9ac6494f6c8badb96.png" class="math-inline" /> が半順序（反射的・反対称・推移的）であることを示せ。

(3) 濃度の大小関係 <img src="tmp/63dbf201871a52e9ac6494f6c8badb96.png" class="math-inline" /> が全順序（任意の2つの濃度が比較可能）であることは、どの公理に依存するか説明せよ。

__問8（基数の順序数としての性質）__

(1) 順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が**基数**であるとはどういうことか、定義せよ。

(2) 任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に対し、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と全単射で結びつく最小の順序数が存在することを、整列可能定理（または選択公理）を用いて説明せよ。

(3) 順序数 <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> が基数であることを示せ。

__問9（正則基数と特異基数）__

(1) 基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が**正則（regular）** であることの定義を述べよ。

(2) 基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> の**共終数** <img src="tmp/b9cf278894fc6c4ce31c542b04c5f35e.png" class="math-inline" /> を定義し、<img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が正則であることと <img src="tmp/16b320dcba1532ca2f10b404aeb538d0.png" class="math-inline" /> であることが同値であることを示せ。

(3) 次の基数が正則か特異かを判定し、理由を述べよ。

- (a) <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />
- (b) <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" />
- (c) <img src="tmp/1741bcd4d659d9931bdd7fa89a4205cf.png" class="math-inline" />
- (d) <img src="tmp/781d2e2dc78b06320faf19ae3e3c6142.png" class="math-inline" />

(4) 正則基数と特異基数の違いを、直感的なイメージで説明せよ。

__問10（基数演算と正則・特異基数）__

(1) 無限基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> に対し、<img src="tmp/a65bfb4a078256a0144295b32976502e.png" class="math-inline" /> が成り立つことを用いて、次のことを示せ。

- 正則基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> について、<img src="tmp/9fe347a99d291aae5aa8abeb1ba57713.png" class="math-inline" /> ならば <img src="tmp/bee8c1dab51befe3820d4dc75d0b662a.png" class="math-inline" />。

(2) 特異基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が「小さな基数の和として表せる」ことを、共終数を用いて説明せよ。

(3) 次の基数演算の結果を求め、その基数が正則か特異かも判定せよ。

- (a) <img src="tmp/c5aa021e5cd0de18b648f108807d8af8.png" class="math-inline" />
- (b) <img src="tmp/dcd8d96f015ded545181ea9b537d5a29.png" class="math-inline" />
- (c) <img src="tmp/29369a631b4b26503c8bd12c65987643.png" class="math-inline" />
- (d) <img src="tmp/8eb8bbfd65c481259951bda8993e064d.png" class="math-inline" />

__問11（連続体仮説の応用）__

(1) 連続体仮説（CH）が成り立つと仮定したとき、次の命題が真か偽かを判定し、理由を述べよ。

> 実数全体 <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の任意の非可算部分集合は、連続体濃度を持つ。

(2) CH を仮定しない ZFC の枠組みで、上記の命題が一般には成り立たない可能性があることを、直感的に説明せよ。

(3) CH が成り立つ世界と成り立たない世界の違いを、実数の部分集合の「サイズの種類」という観点から説明せよ。

__問12（巨大基数と連続体仮説）__

(1) **到達不能基数（inaccessible cardinal）** の定義を述べよ。

(2) 到達不能基数が正則基数であることを示せ。

(3) 到達不能基数の存在が ZFC から独立していることを知っている範囲で説明せよ。

(4) 巨大基数公理を仮定することが、連続体仮説（CH）の真偽にどのような影響を与えるか、知っていることを述べよ。

### 解答

以下、各問の解答です。

__問1（濃度と基数の定義）__

__(1) 濃度の定義__

集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**濃度** <img src="tmp/efa403df45c1ea69e5464fa35c14815a.png" class="math-inline" /> は、次の同値関係による同値類として定義されます。

- 集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の間に全単射が存在するとき、<img src="tmp/46c4dd564b86222198c1bd3d0f7100f9.png" class="math-inline" /> と定義する。
- この同値関係による同値類を「濃度」と呼ぶ。

__(2) 基数の定義（順序数を用いて）__

ZFC において、**基数（cardinal number）** は次のように定義されます。

- 順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が**基数**であるとは、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> より小さい任意の順序数 <img src="tmp/b4d58078d6159ee352ef89f89ce6e36c.png" class="math-inline" /> に対し、<img src="tmp/a7cc9391cfd704a47cf43e4548bb1e40.png" class="math-inline" /> が成り立つことをいう。
- すなわち、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は「自分より小さい順序数との間に全単射が存在しない」最小の順序数である。

また、任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に対し、選択公理（整列可能定理）により <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> を整列させ、その順序型として得られる順序数のうち、最小のものを <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の**基数**と定義し、それを <img src="tmp/efa403df45c1ea69e5464fa35c14815a.png" class="math-inline" /> と書きます。

__(3) 濃度と基数の関係__

- **濃度**：全単射が存在するという同値関係による**同値類**。
- **基数**：その同値類から「代表元」として選ばれた**順序数**。

多くの文脈では同じ記号 <img src="tmp/efa403df45c1ea69e5464fa35c14815a.png" class="math-inline" /> で表され、実質的に同じ意味で使われますが、厳密には
- 濃度は「大きさのクラス」
- 基数は「そのクラスの代表として選ばれた順序数」
という関係にあります。

__問2（濃度の比較）__

__(1) 濃度の大小関係の定義__

濃度の大小関係 <img src="tmp/efdb956bee5d8baccd0fee7507e1ade5.png" class="math-inline" /> は、次のように定義します。

- <img src="tmp/efdb956bee5d8baccd0fee7507e1ade5.png" class="math-inline" /> ⇔ <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> への単射が存在する。

また、<img src="tmp/cdf89d42f0059ea8c63230b802884b42.png" class="math-inline" /> は <img src="tmp/efdb956bee5d8baccd0fee7507e1ade5.png" class="math-inline" /> かつ <img src="tmp/e88505e6ac847fe2c568600a0a97b422.png" class="math-inline" /> と定義します。

__(2) カントールの定理の証明__

**定理**：任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> について <img src="tmp/d9b1e3181ce16e869107e5da64c576d3.png" class="math-inline" />。

**証明**：

1. <img src="tmp/3a95708d26ee44bcb51da0db370d224d.png" class="math-inline" /> を示す。  
   - 写像 <img src="tmp/f4b6558dca2a5d9167a67d73153b08c2.png" class="math-inline" /> を <img src="tmp/92550e7be5b4e6f3790b6b00c3f7e307.png" class="math-inline" /> と定義すると、これは単射である。
   - よって <img src="tmp/3a95708d26ee44bcb51da0db370d224d.png" class="math-inline" />。

2. <img src="tmp/9c850ce9c7e300ae5c43ef52f344297a.png" class="math-inline" />（すなわち全単射が存在しない）を示す。  
   - 背理法で、全単射 <img src="tmp/3de2ec7850b2c5e41109c6a4c3dd3a1d.png" class="math-inline" /> が存在すると仮定する。
   - <img src="tmp/5ba90ab39ecb32600137793f36794aeb.png" class="math-inline" /> とおく。<img src="tmp/42f1edc1db8a856d544a069291f86036.png" class="math-inline" /> なので <img src="tmp/538418b38e0f19010a07a81469e604d6.png" class="math-inline" />。
   - <img src="tmp/ce689b6baffcda72072471af44754299.png" class="math-inline" /> は全射なので、ある <img src="tmp/48b1b260aff3a033bcef457e4aecb474.png" class="math-inline" /> が存在して <img src="tmp/7813edc13cd9e81ed37268095aa64d15.png" class="math-inline" />。
   - このとき、
     - <img src="tmp/1c7b9d06bbb9ff1ebb4d49f09efce87f.png" class="math-inline" /> とすると <img src="tmp/5a0bf4e3f945fdc0e1fdf88716f3f5ed.png" class="math-inline" /> だが、<img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の定義より <img src="tmp/690981b4508fbad1e6ba38d91f2b2dca.png" class="math-inline" /> となり矛盾。
     - <img src="tmp/259cc234906a5e1625438ca49fb16028.png" class="math-inline" /> とすると <img src="tmp/690981b4508fbad1e6ba38d91f2b2dca.png" class="math-inline" /> なので <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の定義より <img src="tmp/1c7b9d06bbb9ff1ebb4d49f09efce87f.png" class="math-inline" /> となり矛盾。
   - いずれも矛盾するので、そのような全単射 <img src="tmp/ce689b6baffcda72072471af44754299.png" class="math-inline" /> は存在しない。

3. よって <img src="tmp/d9b1e3181ce16e869107e5da64c576d3.png" class="math-inline" />。

__(3) 濃度の大小比較__

- <img src="tmp/05bedd17488afc9fbca66cd269648904.png" class="math-inline" />
- <img src="tmp/8c36b63e00b43179429583544530996a.png" class="math-inline" />
- <img src="tmp/f55d0b50760106e9e76b5cdc08a5147d.png" class="math-inline" />

したがって、

- <img src="tmp/f9b0e0541abec6092dfe3f4d7491a866.png" class="math-inline" />

__問3（可算無限集合）__

__(1) 可算無限集合の定義__

集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が**可算無限集合**であるとは、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> が自然数全体の集合 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と全単射で結びつくこと、すなわち <img src="tmp/7d9ce8d0341d1192b344d69e64b6e685.png" class="math-inline" /> であることをいう。

__(2) 可算無限集合であることの証明__

**(a) 偶数全体 <img src="tmp/2a9a74be1b88a507bca22ff741b1ee93.png" class="math-inline" />**

- 写像 <img src="tmp/fd710a92fcf9c9b976b11768fb45a47b.png" class="math-inline" /> を <img src="tmp/dfb1a45f5591c9ed4c0296797512b7b8.png" class="math-inline" /> と定義すると、全単射。
- よって可算無限集合。

**(b) 整数全体 <img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" />**

- 写像 <img src="tmp/348f9544b22fbe3a546f607cfbf64200.png" class="math-inline" /> を
  

<div class="math-display-container"><img src="tmp/f9cdcddfb7a1b822e72e9211a039ec7b.png" class="math-display" /></div>


  と定義すると、全単射。
- よって可算無限集合。

**(c) 有理数全体 <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" />**

- 有理数は既約分数 <img src="tmp/f4f4ff9a31e576a624cc206bae56eadb.png" class="math-inline" />（<img src="tmp/6accb358b72f5972637b44290a8880ad.png" class="math-inline" />）として表せる。
- 写像 <img src="tmp/b03a49c7e02011612f5601160740db66.png" class="math-inline" /> は <img src="tmp/a4b98dd0dbb6478dbe4f3c63dda933a7.png" class="math-inline" /> から <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> への全射。
- <img src="tmp/a4b98dd0dbb6478dbe4f3c63dda933a7.png" class="math-inline" /> は可算無限集合（<img src="tmp/610eff1e93167f087c8e08f1325e1b6b.png" class="math-inline" /> と <img src="tmp/b67d8cea7201395d869d32523e602eba.png" class="math-inline" /> は可算）なので、その像である <img src="tmp/149c1fc7eab19005c4b98bc4c969cbcb.png" class="math-inline" /> も可算無限集合。

**(d) 整数係数の多項式全体 <img src="tmp/61cf8b9404384d463cf04e8aaaa97c9d.png" class="math-inline" />**

- 各多項式は有限個の係数 <img src="tmp/e7e8104369bddc18354437e8ab2b9209.png" class="math-inline" /> で表せる。
- 長さ <img src="tmp/e4bc429c866bbde4b0f321d9b5139986.png" class="math-inline" /> の整数列全体は <img src="tmp/2077620a503a4deee85a1d768502ea48.png" class="math-inline" /> と同相で、可算無限集合。
- 可算無限個の可算無限集合の和集合として表せるので、可算無限集合。

__(3) 可算無限集合の部分集合の性質__

<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> を可算無限集合とし、<img src="tmp/42f1edc1db8a856d544a069291f86036.png" class="math-inline" /> とする。

- <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> が有限なら有限集合。
- <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> が無限なら、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> から <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> への全単射 <img src="tmp/bf88d0d594906180503c2797ecf4f18d.png" class="math-inline" /> をとり、<img src="tmp/bef05e69c83546d0a77ed294fc99a2a0.png" class="math-inline" /> は無限部分集合。
- 自然数の無限部分集合は <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> と順序同型（最小元から順に番号を振れる）なので可算無限。
- よって <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> も可算無限集合。

__問4（非可算無限集合）__

__(1) <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> が非可算であること（対角線論法）__

**証明**（背理法）：

- <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> が可算無限であると仮定する。すると、<img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の元を <img src="tmp/2ce9795a49acef17b7714cae8bc86a0b.png" class="math-inline" /> と列挙できる。
- 各実数 <img src="tmp/3c2c52fab4bf95b0c6610f35afa374bc.png" class="math-inline" /> を10進小数展開（一意性のために、有限小数は <img src="tmp/a236b6bf7936396063dd076315ea55e1.png" class="math-inline" /> 型を避ける）で
  

<div class="math-display-container"><img src="tmp/322750edaa1c7382b6971b47fef96069.png" class="math-display" /></div>


  と表す。
- 新しい実数 <img src="tmp/add5bb4fc3bfe26587c101125d12ae33.png" class="math-inline" /> を
  

<div class="math-display-container"><img src="tmp/a668078d83ab547e7d1984e0885328ed.png" class="math-display" /></div>


  と定義する。
- この <img src="tmp/add5bb4fc3bfe26587c101125d12ae33.png" class="math-inline" /> はどの <img src="tmp/3c2c52fab4bf95b0c6610f35afa374bc.png" class="math-inline" /> とも <img src="tmp/1595ee4e856afd6d12858bd9b47ac2fd.png" class="math-inline" /> 桁目が異なるので、列挙に含まれない。
- よって矛盾。したがって <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> は非可算無限集合。

__(2) <img src="tmp/039474a1a493fb3c2d8f4adbf24f971e.png" class="math-inline" /> と <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> の濃度が等しいこと__

- 関数 <img src="tmp/4805c410196d30385807d4848624c085.png" class="math-inline" /> を
  

<div class="math-display-container"><img src="tmp/afc61b5b1b2331ccc76f91047ba6bcd0.png" class="math-display" /></div>


  と定義すると、これは全単射（<img src="tmp/039474a1a493fb3c2d8f4adbf24f971e.png" class="math-inline" /> を <img src="tmp/3d4c0178ea4508c181bdcef23de99208.png" class="math-inline" /> に線形写像し、tan で <img src="tmp/a38f21982e54bae61d1551d37b212ff2.png" class="math-inline" /> 全体に写す）。
- よって <img src="tmp/08347ab74b2236de05f4973e74814d87.png" class="math-inline" />。

__(3) 連続体濃度 <img src="tmp/107dd2dfd0c9f57f499e47a77ddaaeff.png" class="math-inline" /> の例__

**(a) 閉区間 <img src="tmp/171d58b8063a24dc65a33bacb46a5fb7.png" class="math-inline" />**

- <img src="tmp/c989955ebe63d4bdf6edd795395e3520.png" class="math-inline" /> かつ <img src="tmp/21dc5407784c4a67680620b1948867a0.png" class="math-inline" /> なので、<img src="tmp/8b3850d5eadd8095c43bb3850704e6cc.png" class="math-inline" />。

**(b) 複素数全体 <img src="tmp/1d8ddd3aadaa5a36ac8588548645f46c.png" class="math-inline" />**

- <img src="tmp/27eecf62430ced7a6a07f2066f250d0e.png" class="math-inline" /> なので <img src="tmp/d4f9a9a63e2879c2605dbe2be47ffdf9.png" class="math-inline" />。

**(c) 実数列全体 <img src="tmp/226899de81ec35f904a2efff77891b5c.png" class="math-inline" />**

- <img src="tmp/db0b446afb5e4c7ada855d004fc4e73c.png" class="math-inline" />。

__問5（基数の演算）__

__(1) 演算の定義__

基数 <img src="tmp/78c7c4b30c2d9733b3cfbf415205f302.png" class="math-inline" /> に対し、<img src="tmp/49646265ce40ee2c7522063f557d60cb.png" class="math-inline" /> となる集合 <img src="tmp/f94c95c5bdfef2b6f0a176a1a91c74fc.png" class="math-inline" /> をとる。

- **(a) 和**：<img src="tmp/9a03a4d7e1d4f4beb52d787432480c5a.png" class="math-inline" />（ただし <img src="tmp/d171287a19b00490e9b6a6a54cb247a5.png" class="math-inline" />）
- **(b) 積**：<img src="tmp/561cec90a3aff9695a2fb9abb8e77cef.png" class="math-inline" />
- **(c) 冪**：<img src="tmp/0f7fcbce597562c10007e2e86b21b65f.png" class="math-inline" />

__(2) 等式の判定__

**(a) <img src="tmp/457a57457068092bbbe36a32c9fbd018.png" class="math-inline" />**

- 真。無限集合に有限個の元を加えても濃度は変わらない。

**(b) <img src="tmp/98a18778764aec07402f1633cc912131.png" class="math-inline" />**

- 真。可算無限集合2つの非交和は可算無限。

**(c) <img src="tmp/7b10aaad6c4308f1b07b55389d07f00a.png" class="math-inline" />**

- 真。<img src="tmp/7f1d134f574dd32cd3f99fe5c6229948.png" class="math-inline" /> は可算無限。

**(d) <img src="tmp/baa1dd28be7ebca28783c2bfe2cee036.png" class="math-inline" />**

- 偽。カントールの定理より <img src="tmp/962a228ac52da28bd482d9bc2a0db055.png" class="math-inline" />。

__(3) <img src="tmp/2e66da77e1bc36cbdba2e0966a29a5b1.png" class="math-inline" /> の証明__

<img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> を無限基数、<img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> を任意の基数とする。

- <img src="tmp/8639157a6bc47562d88847ce7a8c91e0.png" class="math-inline" /> のとき：<img src="tmp/50a301a88821bbe14945643c90d04dcf.png" class="math-inline" />。
  - <img src="tmp/2968507ccf9b05f55179f95a51740bc6.png" class="math-inline" />（無限基数の性質）。
  - 一方 <img src="tmp/d346402f15ab344e20ac511efc8e53ef.png" class="math-inline" /> は明らか。
  - よって <img src="tmp/5945041b3bb74413f479f29277314576.png" class="math-inline" />。
  - 同様に <img src="tmp/a34dd31bc90d17a361a2a9a1b01486d3.png" class="math-inline" />。
- <img src="tmp/6a7fa5c4b0cc6441bc979670b6785c5c.png" class="math-inline" /> のとき：<img src="tmp/8ac8c8e9d85ba644c9749c93cef32460.png" class="math-inline" />。
  - <img src="tmp/8b3beb8c38bff3cc03765d1e81036930.png" class="math-inline" />。
  - 一方 <img src="tmp/5b76b8810dc2cad61cab24d1f90d0680.png" class="math-inline" /> は明らか。
  - よって <img src="tmp/bee8c1dab51befe3820d4dc75d0b662a.png" class="math-inline" />。
  - 同様に <img src="tmp/210312b3da1b2eceb53915d400bebecb.png" class="math-inline" />。

したがって、いずれの場合も <img src="tmp/2e66da77e1bc36cbdba2e0966a29a5b1.png" class="math-inline" />。

__問6（アレフ数と連続体仮説）__

__(1) アレフ数の定義__

順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> に対して、**アレフ数** <img src="tmp/b119e30f90d20c190b96e80e81269deb.png" class="math-inline" /> は次のように定義される。

- <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />：最小の無限基数（可算無限濃度）。
- 順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が与えられたとき、<img src="tmp/b119e30f90d20c190b96e80e81269deb.png" class="math-inline" /> は「それより小さいすべての基数 <img src="tmp/020d9315ee35c89582b08191399cbc96.png" class="math-inline" />（<img src="tmp/8b343c5abba58f89bdc01954701dd97e.png" class="math-inline" />）と全単射を持たない」最小の順序数。

__(2) アレフ数の性質__

**(a) <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />**

- 自然数全体 <img src="tmp/4b2bf2a0d6267d2290afd93ed01fd102.png" class="math-inline" /> の濃度。可算無限集合の濃度。

**(b) <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" />**

- <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" /> の次の基数。最小の非可算基数。

**(c) <img src="tmp/fc26b6e98f20e9c587c0356a460acde5.png" class="math-inline" />**

- <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" /> の次の基数。<img src="tmp/40d0358dcca1e3faecc2f4d0ba7ed6ee.png" class="math-inline" />。

__(3) 連続体仮説（CH）と一般連続体仮説（GCH）__

- **CH**：<img src="tmp/0ed95fbd8a8f7e23c6c080c57a0da502.png" class="math-inline" />
- **GCH**：任意の無限基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> について <img src="tmp/e5fa7124ce55e5e21e1d3d86ba733dfd.png" class="math-inline" />

これらは ZFC から独立（証明も反証もできない）であることが知られています。

__問7（濃度の同値類）__

__(1) 同値関係の well-defined 性__

「濃度が等しい」という関係を

- <img src="tmp/b33ff4cffc07d6cb89d704a3c2bc1b9f.png" class="math-inline" /> <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と <img src="tmp/c0ad0b98746c42733a7cdd4a47a9dc59.png" class="math-inline" /> の間に全単射が存在する

と定義する。

- 反射律：恒等写像により <img src="tmp/ebb9d9215fab7eeabf252e732c67aa76.png" class="math-inline" />。
- 対称律：<img src="tmp/d29b665c21e6c60b86868dc565af865c.png" class="math-inline" /> なら全単射 <img src="tmp/18766859ca7c11d2de922641bfec7ae1.png" class="math-inline" /> が存在し、その逆写像 <img src="tmp/3d7e9ec55e33f769e417c11c5ab449b3.png" class="math-inline" /> も全単射なので <img src="tmp/3d8e3db090e04716ed8137b989128e5a.png" class="math-inline" />。
- 推移律：<img src="tmp/d29b665c21e6c60b86868dc565af865c.png" class="math-inline" /> かつ <img src="tmp/d4f165a00576c18773298b33f27e4bdb.png" class="math-inline" /> なら、全単射の合成により <img src="tmp/52b61e77ef81a6079d1439e9b8c0e19c.png" class="math-inline" />。

よって同値関係であり、商集合が well-defined。

__(2) 大小関係 <img src="tmp/63dbf201871a52e9ac6494f6c8badb96.png" class="math-inline" /> が半順序であること__

- 反射的：恒等写像により <img src="tmp/76cb1278ae31f17134ca21eed7882272.png" class="math-inline" />。
- 反対称：（ベルンシュタインの定理）<img src="tmp/efdb956bee5d8baccd0fee7507e1ade5.png" class="math-inline" /> かつ <img src="tmp/70b6be17f300615cddff13ce3092a2fe.png" class="math-inline" /> なら <img src="tmp/46c4dd564b86222198c1bd3d0f7100f9.png" class="math-inline" />。
- 推移的：単射の合成により <img src="tmp/efdb956bee5d8baccd0fee7507e1ade5.png" class="math-inline" /> かつ <img src="tmp/da55aa0146baa07741cb1334735f9f0b.png" class="math-inline" /> なら <img src="tmp/25d6b4e8e2a3185b1bb3f6de295de40b.png" class="math-inline" />。

__(3) 全順序性に依存する公理__

濃度の大小関係が全順序（任意の2つの濃度が比較可能）であることは、**選択公理（AC）** に依存します。

- AC を仮定すると、任意の集合は整列可能であり、順序数と比較できるため濃度も比較可能。
- AC を仮定しない場合、比較不能な濃度が存在しうることが知られています。

__問8（基数の順序数としての性質）__

__(1) 基数の定義（順序数として）__

順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> が**基数**であるとは、

- 任意の <img src="tmp/b4d58078d6159ee352ef89f89ce6e36c.png" class="math-inline" /> について <img src="tmp/a7cc9391cfd704a47cf43e4548bb1e40.png" class="math-inline" />

が成り立つことをいう。すなわち、<img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> は「自分より小さい順序数との間に全単射が存在しない」最小の順序数。

__(2) 最小の順序数の存在__

任意の集合 <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> に対し、整列可能定理（AC と同値）により <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> を整列できる。

- 整列順序 <img src="tmp/74e9979e422e8d2011129b4f8f6e0d9f.png" class="math-inline" /> の順序型を <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> とすると、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> は順序数 <img src="tmp/b6728823d99de89134691e64effef116.png" class="math-inline" /> と順序同型。
- 順序数全体は整列クラスなので、<img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> と順序同型な順序数のうち最小のものが存在する。
- これを <img src="tmp/bc50d4bc86728307fe84984a7b1a875d.png" class="math-inline" /> の基数と定義する。

__(3) <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> が基数であること__

- <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は有限順序数 <img src="tmp/18991571b007a883e99c3893b3740bdf.png" class="math-inline" /> より大きいが、有限順序数とは全単射を持たない（有限と無限は濃度が異なる）。
- よって <img src="tmp/19403e8d53013064d94b69e9b5d3b1d0.png" class="math-inline" /> は基数である（<img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />）。

__問9（正則基数と特異基数）__

__(1) 正則基数の定義__

基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が**正則**であるとは、

- 任意の集合族 <img src="tmp/da5f053c957a98144578e65e9806d498.png" class="math-inline" /> について、
  - <img src="tmp/16a2f93a69d2a258e0b0fcae008e598f.png" class="math-inline" />（各 <img src="tmp/a3dcd668e156e5eab97d6545beb2a0d8.png" class="math-inline" />）
  - <img src="tmp/9fe347a99d291aae5aa8abeb1ba57713.png" class="math-inline" />
- ならば
  - <img src="tmp/e517e08d65349292de14450f023b216e.png" class="math-inline" />

が成り立つことをいう。

__(2) 共終数との同値性__

基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> の**共終数** <img src="tmp/b9cf278894fc6c4ce31c542b04c5f35e.png" class="math-inline" /> は、

- <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> に共終な増加列の最小の長さ

として定義される。

このとき、

- <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が正則 ⇔ <img src="tmp/16b320dcba1532ca2f10b404aeb538d0.png" class="math-inline" />
- <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が特異 ⇔ <img src="tmp/b2bc86da093198b182f40a3bc677b0c7.png" class="math-inline" />

が成り立つ。

__(3) 正則・特異の判定__

**(a) <img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />**

- 正則。有限個または可算個の有限集合の和は高々可算。

**(b) <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" />**

- 正則。後続基数は常に正則。

**(c) <img src="tmp/1741bcd4d659d9931bdd7fa89a4205cf.png" class="math-inline" />**

- 特異。<img src="tmp/8e2ffce12f50c5c716b8226cb89deca6.png" class="math-inline" />。

**(d) <img src="tmp/781d2e2dc78b06320faf19ae3e3c6142.png" class="math-inline" />**

- 正則。後続基数。

__(4) 直感的な違い__

- 正則基数：小さな集合を小さな個数だけ集めても、全体はその基数に達しない。
- 特異基数：小さな集合を小さな個数だけ集めると、全体がその基数になってしまう。

__問10（基数演算と正則・特異基数）__

__(1) 正則基数における和__

<img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> を正則基数、<img src="tmp/9fe347a99d291aae5aa8abeb1ba57713.png" class="math-inline" /> とする。

- <img src="tmp/af4fcd1a80d774242806984cee32326c.png" class="math-inline" />（無限基数の和の性質）。
- 正則性から、<img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> 未満の濃度を <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> 未満個集めても <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> に達しないが、ここでは演算の定義により直接 <img src="tmp/bee8c1dab51befe3820d4dc75d0b662a.png" class="math-inline" /> が得られる。

__(2) 特異基数の「和としての表現」__

<img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が特異なら <img src="tmp/246a0f425bdf49b0f60b495e390b971b.png" class="math-inline" />。

- 共終な増加列 <img src="tmp/cad8a75060c904cbfd41425e2d81f1d5.png" class="math-inline" />（各 <img src="tmp/19ee46c273a8e49b1c6cb972728f8cbc.png" class="math-inline" />）が存在し、
  

<div class="math-display-container"><img src="tmp/873710673bbedb19545a367c4a41aa19.png" class="math-display" /></div>


- この列を用いると、<img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> は「より小さい基数 <img src="tmp/37c382c96693867fbd0b138408bfbf90.png" class="math-inline" /> を <img src="tmp/d52e776be328464a0576b7332294ff5f.png" class="math-inline" /> 個集めたもの」として表現できる。

__(3) 演算結果と正則・特異__

**(a) <img src="tmp/c5aa021e5cd0de18b648f108807d8af8.png" class="math-inline" />**

- <img src="tmp/2789ecd469fb44091194e2be1786cefa.png" class="math-inline" />（特異）。

**(b) <img src="tmp/dcd8d96f015ded545181ea9b537d5a29.png" class="math-inline" />**

- <img src="tmp/2789ecd469fb44091194e2be1786cefa.png" class="math-inline" />（特異）。

**(c) <img src="tmp/29369a631b4b26503c8bd12c65987643.png" class="math-inline" />**

- <img src="tmp/49a0d171df951b8186e5b73e95cb2d8c.png" class="math-inline" />（特異）。

**(d) <img src="tmp/8eb8bbfd65c481259951bda8993e064d.png" class="math-inline" />**

- <img src="tmp/49a0d171df951b8186e5b73e95cb2d8c.png" class="math-inline" />（特異）。

__問11（連続体仮説の応用）__

__(1) CH 下での命題の真偽__

CH を仮定すると <img src="tmp/0ed95fbd8a8f7e23c6c080c57a0da502.png" class="math-inline" />。

- 実数の非可算部分集合 <img src="tmp/9a291854af6a974cbb198bfb3a26fcea.png" class="math-inline" /> は <img src="tmp/a6b8a5e42f66086fa8cc13a1a1aeb1f0.png" class="math-inline" />。
- 一方 <img src="tmp/2044d0bf8295b8d438dafd1503f6524e.png" class="math-inline" />。
- よって <img src="tmp/f1ea280d6295b21190d24e52e3fff0e1.png" class="math-inline" />。

したがって命題は真。

__(2) CH を仮定しない場合の可能性__

CH を仮定しない ZFC では、<img src="tmp/107dd2dfd0c9f57f499e47a77ddaaeff.png" class="math-inline" /> は <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" /> より大きくなり得る。

- このとき、濃度が <img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" /> で <img src="tmp/51c7f4b26fc735e3919bc230f6404114.png" class="math-inline" /> となる実数部分集合が存在しうる。
- よって「非可算なら連続体濃度」とは限らない。

__(3) CH の世界と非 CH の世界の違い__

- CH が成り立つ世界：実数の部分集合の濃度は
  - 可算（<img src="tmp/bf2d2a54bd4e59e4f8f0ec768e5cf804.png" class="math-inline" />）
  - 連続体濃度（<img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" />）
  の2種類のみ。
- CH が成り立たない世界：中間的な濃度（例：<img src="tmp/feca73683b78968c579aaf6c2edd0c91.png" class="math-inline" />）の部分集合が存在しうる。

__問12（巨大基数と連続体仮説）__

__(1) 到達不能基数の定義__

基数 <img src="tmp/99821c710bdd6e5c9698cb46197e4945.png" class="math-inline" /> が**到達不能（inaccessible）** であるとは、次の3条件を満たすことをいう。

1. **非可算**：<img src="tmp/c452c38a882b6c413881bdb79c3b28ba.png" class="math-inline" />
2. **正則**：<img src="tmp/16b320dcba1532ca2f10b404aeb538d0.png" class="math-inline" />
3. **強極限**：任意の <img src="tmp/9fe347a99d291aae5aa8abeb1ba57713.png" class="math-inline" /> について <img src="tmp/175f026ed9f92a6faee8be22facb0849.png" class="math-inline" />

__(2) 到達不能基数が正則であること__

定義により正則性（<img src="tmp/16b320dcba1532ca2f10b404aeb538d0.png" class="math-inline" />）は明示的に要求されている。

__(3) 到達不能基数の存在の独立性__

- 到達不能基数の存在は ZFC から証明できない（ZFC が無矛盾なら、ZFC +「到達不能基数は存在しない」も無矛盾）。
- また、ZFC +「到達不能基数が存在する」も無矛盾（巨大基数公理として採用可能）。
- よってその存在は ZFC から独立。

__(4) 巨大基数公理と CH の関係__

- 巨大基数公理（到達不能基数以上の存在）を仮定しても、CH の真偽は決定されない。
- 巨大基数公理は強制法によるモデル構成を豊かにし、CH が成り立たないモデルを構成しやすくするが、CH を「必然的に偽」にするわけではない。
- CH の真偽は、巨大基数公理とは独立に選択できる追加公理として扱われる。


<div style="page-break-before:always"></div>
