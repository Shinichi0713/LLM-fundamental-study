LLMの内部レイヤ（中間表現・活性化）を参照・操作する技術は、解釈可能性（Interpretability）研究とモデル制御（Steering/Editing）研究の両面から活発に発展しています。以下に主要な技術を分類してまとめます。

---

## 1. 中間表現の「読み取り」技術

### Logit Lens
各層の hidden state をそのまま語彙空間に投影（language modeling head を経由）し、「この層の時点でモデルは何を予測しようとしているか」を可視化する技術です。層が深くなるにつれて予測が収束していく様子を観察できます。

### Probing（プロービング）
中間表現に対して線形分類器や回帰器を学習させ、**特定の概念（品詞、文法関係、真偽、感情など）がどの層に符号化されているか**を特定します。線形性が確保できれば、その概念は表現空間に「読み出し可能」であると判断されます。

### Mechanistic Analysis（因果追跡）
特定の入力パッチ（activation patching）を別の入力のものに差し替えて、**どの層・どのトークン位置が出力に因果的に影響しているか**を特定します。因果関係を追跡することで、モデル内部の「回路（circuit）」を特定できます。

---

## 2. 中間表現の「操作・制御」技術

### Activation Addition / Steering Vectors
対照的な概念ペア（例：「肯定的な文」vs「否定的な文」）から**方向ベクトル（steering vector）**を抽出し、推論時に任意の層の hidden state に加算・減算することで、生成文のトーンや性質を連続的に制御します。数値的な「ノブ」のような操作が可能です。

### Representation Engineering (RepE)
PCA などで中間表現から「真偽」「安全性」「同意/拒否」などの主成分を抽出し、その方向に沿って表現を調整することで、モデルの振る舞いを高レベルに制御する枠組みです。単一方向の加算にとどまらず、表現空間全体の構造を操作する点が特徴です。

### Function Vectors / Task Vectors
特定のタスク（例：文法修正、翻訳、要約）を実行する際の層間の変換パターンを抽出し、それを別の入力に**注入（grafting）**することで、プロンプトなしでタスク実行能力を引き出す技術です。In-context Learning が層間でどう実現されているかを操作レベルで示しています。

---

## 3. 知識・パラメータの「編集」技術

### Knowledge Editing（ROME / MEMIT / R-Llama など）
特定の事実知識（例：「Aの首都はB」）を、**中間層の特定の行列（FFNの重み）**を直接書き換えることで、モデル全体を再学習せずに更新・削除する技術群です。層レベルでの「局所的な記憶書き換え」を実現します。

### Layer Swapping / Model Surgery
モデル間（例：ベースモデルとファインチューニング済みモデル、あるいは異なるサイズのモデル）で特定の層やブロックを入れ替え・複製・削除し、**どの層がどの能力に寄与しているか**を実験的に特定する手法です。能力の「局在化」を調べるために使われます。

---

## 4. Attention 機構の直接操作

### Attention Patching / Head Ablation
特定の attention head や層の attention パターンを別の入力のものに置き換えたり、特定の head を無効化（ablation）したりして、**どの head が文法解析・照応・事実想起などに使われているか**を特定します。

### Attention Steering
推論時に attention スコア自体を直接操作（特定トークンへの注目を強化・抑制）することで、長文脈の追跡や特定情報の無視を制御する手法も研究されています。

---

## 5. 層間の「情報伝達」操作

### Latent Space Interpolation / Mixture of Depths
層と層の間の情報伝達を操作する技術です。例えば、Mixture of Depths（MoD）では、トークンごとに「どの層まで計算するか」を動的に決定し、不要な層計算をスキップします。これは層への「参照」を動的に制御する構造的操作です。

### Skip Connection Gating
Residual connection の通過情報をゲーティングすることで、層間の情報伝達量を調整し、浅い層の情報と深い層の情報のバランスを制御する手法も提案されています。

---

## まとめ：技術の分類図

```
レイヤ参照・操作技術
├── 「読み取り」系
│   ├── Logit Lens（層ごとの予測可視化）
│   ├── Probing（概念の符号化位置特定）
│   └── Causal Tracing（因果的な影響の追跡）
├── 「制御」系
│   ├── Steering Vectors（方向ベクトル加算）
│   ├── RepE（表現空間の構造的操作）
│   └── Function Vectors（タスク能力の注入）
├── 「編集」系
│   ├── ROME / MEMIT（知識の層レベル書き換え）
│   └── Layer Swapping（層の入れ替え実験）
└── 「Attention 操作」系
    ├── Head Ablation（head の無効化実験）
    └── Attention Patching（attention パターンの差し替え）
```

これらの技術は、モデルを「ブラックボックス」として使うのではなく、**「どの層が何を担当し、どの方向を操作すれば振る舞いが変わるか」**を科学的に扱うための基盤となっています。特に、Representation Engineering と Knowledge Editing は、モデルの安全性調整や事実更新の新たなパラダイムとして注目されています。


## レイヤ選択のオリジン

はい、現在のレイヤー選択研究には明確な「起源」となる論文が存在し、大きく**3つの系統**に分類できます。それぞれの系譜をご説明いたします。

### 系統1：層の確率的ドロップ（プルーニング・圧縮の起源）

__Stochastic Depth (2016)__
- **著者**: Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Q. Weinberger（Cornell University, Tsinghua University）
- **発表**: ECCV 2016
- **概要**: ResNetなどの非常に深い畳み込みネットワークにおいて、**学習時に層（残差ブロック）を確率的にドロップする**手法を提案しました。これにより、非常に深いネットワークの学習を安定化させ、正則化効果も得られます。被引用数は約2,000回に達し、現在の層スキップ・プルーニング研究の最も直接的な起源です。<source-chip title="arXiv" url="https://arxiv.org/abs/1603.09382" />

**発展**: Stochastic Depth → **LayerDrop (Fan et al., 2019/2020)** → 現在のLLM層プルーニング（LaCo, FinerCut, DLP など）

__LayerDrop (2019/2020)__
- **著者**: Angela Fan, Edouard Grave, Armand Joulin（Facebook AI Research）
- **発表**: ICLR 2020
- **概要**: Stochastic DepthをTransformerに拡張し、**学習時に層をランダムにドロップし、推論時に任意の深度のサブモデルを抽出できる**手法（LayerDrop）を提案。これはTransformer時代における動的深度選択の直接的な先駆けとなりました。<source-chip title="arXiv" url="https://arxiv.org/pdf/1909.11556" />

### 系統2：適応的計算時間（動的深度・計算量調整の起源）

__Adaptive Computation Time (ACT) (2016)__
- **著者**: Alex Graves（Google DeepMind）
- **発表**: 2016年
- **概要**: RNNが**入力を受けてから出力を出すまでの計算ステップ数を、入力の難易度に応じて学習する**アルゴリズムを提案。これは「すべての入力に同じ計算量を使う必要はない」という核心的な洞察を初めて形式化したものです。被引用数は360回以上。<source-chip title="arXiv" url="https://arxiv.org/abs/1603.08983" />

**発展**: ACT → **Depth-Adaptive Transformer (Elbayad et al., 2019)** → **PonderNet (2021)** → 現在の動的深度調整・トークンルーティング（Mixture-of-Depths, Token-Select, BUDDY など）

__Depth-Adaptive Transformer (2019)__
- **著者**: Maha Elbayad, Jiatao Gu, Edouard Grave, Michael Auli（Facebook AI）
- **発表**: 2019年
- **概要**: ACTの考え方をTransformerに初めて本格的に応用し、**入力シーケンスの難易度に応じて異なる層深さで予測を行う**モデルを提案。現在の動的深度ルーティング研究の直接的な先駆けです。<source-chip title="arXiv" url="https://arxiv.org/abs/1910.10073" />

### 系統3：動的ルーティング（経路選択の起源）

__Deciding How to Decide: Dynamic Routing in Artificial Neural Networks (2017)__
- **著者**: Mason McGill, Pietro Perona（Caltech）
- **発表**: ICML 2017
- **概要**: ニューラルネットワーク内で**入力信号に応じて異なる経路（パス）を動的に選択する**手法を体系的に評価した基盤的な研究。現在のトークンレベル動的ルーティングの理論的基盤となっています。<source-chip title="PMLR" url="https://proceedings.mlr.press/v70/mcgill17a/mcgill17a.pdf" />

### 系譜のまとめ

| 系統 | 起源論文（年） | 核心的アイデア | 現在の代表的研究 |
|------|-------------|--------------|---------------|
| **層の確率的ドロップ** | Stochastic Depth (2016) | 学習時に層をランダムにドロップ | LayerDrop, LaCo, FinerCut, DLP |
| **適応的計算時間** | Adaptive Computation Time (2016) | 入力の難易度に応じて計算ステップを変える | Mixture-of-Depths, Token-Select, BUDDY |
| **動的ルーティング** | McGill & Perona (2017) | 入力に応じて異なる経路を選択 | Radial Networks, Dr.LLM, HeRo |

### 特に注目すべき「原点」

もっとも影響力が大きいのは、**2016年にほぼ同時期に発表された2本の論文**です：

1. **Stochastic Depth (Huang et al., 2016)**: 「層をスキップする」という物理的な操作の起源
2. **Adaptive Computation Time (Graves, 2016)**: 「入力に応じて計算量を変える」という概念的な起源

この2本の論文が、現在のレイヤー選択研究の**両輪**を形成しています。前者が「どの層を使うか」の**構造**を、後者が「いくら計算するか」の**適応性**をそれぞれ開拓しました。

もし特定の系統についてさらに深く知りたい論文がございましたら、お知らせください。
