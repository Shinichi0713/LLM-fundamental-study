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


## LLM向け

はい、Stochastic Depth（ランダムな層ドロップ）とは異なり、**入力や問題の難易度に応じてレイヤーを選択する**研究は、別の系統として発展してきました。以下に主要な論文を系統別にご紹介いたします。

---

## 系統1：入力の難易度に応じて計算ステップ数を変える

### Adaptive Computation Time (ACT) (2016)
- **著者**: Alex Graves（Google DeepMind）
- **発表**: 2016年
- **概要**: RNNが**入力を受けてから出力を出すまでの計算ステップ数を、入力の複雑さに応じて学習する**手法を提案しました。シンプルな入力には少ないステップ、複雑な入力には多いステップを使います。これは「すべての入力に同じ計算量を使う必要はない」という核心的洞察を、**初めて学習可能な形で実現**した論文です。<source-chip title="arXiv" url="https://arxiv.org/abs/1603.08983" />

### Spatially Adaptive Computation Time (SACT) (2017)
- **著者**: Michael Figurnov ら（Google, HSE, CMU）
- **発表**: [CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Figurnov_Spatially_Adaptive_Computation_CVPR_2017_paper.html)
- **概要**: ACTをResNetに拡張し、**画像の「領域」ごとに計算ステップ数を変える**手法を提案しました。画像内の簡単な部分（空など）では少ない層で処理し、複雑な部分（物体の境界など）では多くの層を使います。画像分類・物体検出・セマンティックセグメンテーションなど、問題に依存せず適用可能です。<source-chip title="arXiv" url="https://arxiv.org/abs/1612.02297" />

### Depth-Adaptive Transformer (2020)
- **著者**: Maha Elbayad ら（Facebook AI Research）
- **発表**: [ICLR 2020](https://arxiv.org/abs/1910.10073)
- **概要**: Transformerで**入力シーケンスの難易度に応じて異なる層深さで予測を行う**モデルを提案。例えば、簡単な翻訳文（"Merci." → "Thank you."）は浅い層で出力し、難しい文は深い層まで計算します。複数の「出口」を設け、どこで停止するかを学習します。<source-chip title="arXiv" url="https://arxiv.org/abs/1910.10073" />

---

## 系統2：トークンごとに層を選択する（現在の主流）

### Mixture-of-Depths (MoD) (2024)
- **著者**: David Raposo ら（Google DeepMind）
- **発表**: 2024年
- **概要**: Transformerの各層で、**トークンごとにその層を通過させるかスキップするかをルーターが動的に決定**します。重要なトークンには計算を集中させ、不要なトークンは恒等写像でバイパスします。これはStochastic Depthの「ランダムドロップ」を、**入力に応じた学習可能な選択へと進化させた**ものです。<source-chip title="arXiv" url="https://arxiv.org/abs/2404.02258" />

### Token-Selective Attention (2025)
- **著者**: Ahmed Abdelmuniem Abdalla
- **発表**: 2026年（arXiv先行公開）
- **概要**: 標準的なTransformerがすべてのトークンに同じ数の層を適用するのに対し、**トークンの文脈的脈的難易度に応じて層数を変える**手法を提案しています。<source-chip title="arXiv" url="https://arxiv.org/html/2605.05222v1" />

### Router-Tuning (2024)
- **著者**: Shwai He ら（University of Maryland, Tencent AI Lab）
- **発表**: 2024年
- **概要**: Mixture of Depths（MoD）を発展させ、**ルーターのみを微調整することで動的深度調整**を実現する手法です。既存の事前学習済みモデルに対しても適用可能です。<source-chip title="arXiv" url="https://arxiv.org/html/2410.13184v2" />

---

## 系統3：条件付き計算・専門家選択（MoEの系統）

### Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017)
- **著者**: Noam Shazeer ら（Google Brain, Jeff Dean, Geoffrey Hinton など）
- **発表**: [ICLR 2017](https://arxiv.org/abs/1701.06538)
- **概要**: **入力に応じて一部の「専門家（層）」のみを活性化する**条件付き計算手法を提案。ネットワークの容量を大幅に増やしつつ、計算コストは増やさないというアイデアを実現しました。現在のMixture-of-Experts（MoE）モデルの直接的な起源です。<source-chip title="arXiv" url="https://arxiv.org/abs/1701.06538" />

---

## 系統4：早期終了（Early Exit）

### BranchyNet / Shallow-Deep Networks など
- ネットワークの途中に複数の「出口」を設け、**モデル自身が予測の自信度に応じて早めに出力する**手法群です。簡単な入力は浅い層で、難しい入力は深い層で出力します。

---

## 系譜のまとめ

「入力に応じてレイヤーを選択する」研究の発展を時系列で整理すると：

| 年 | 論文 | 選択の単位 | 核心 |
|---|------|----------|------|
| 2016 | [ACT (Graves)](https://arxiv.org/abs/1603.08983) | 計算ステップ数 | 入力の複雑さに応じて計算回数を変える |
| 2017 | [SACT (Figurnov)](https://arxiv.org/abs/1612.02297) | 画像の領域ごと | 画像内で難易度の異なる領域に異なる計算量 |
| 2017 | [MoE (Shazeer)](https://arxiv.org/abs/1701.06538) | 専門家（層） | 入力に応じて一部の層のみ活性化 |
| 2020 | [Depth-Adaptive Transformer](https://arxiv.org/abs/1910.10073) | シーケンスごと | 簡単な文は浅い層で、難しい文は深い層で |
| 2024 | [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) | トークン・層ごと | 各層でトークンごとに通過/スキップを学習 |

Stochastic Depth（ランダム）から、ACT・SACT（難易度に応じた計算量調整）を経て、現在のMoD（トークンごとの動的ルーティング）へと発展してきたことがお分かりいただけるかと思います。特に [SACT](https://arxiv.org/abs/1612.02297) は、Stochastic Depthと同年（2016年）にarXivに投稿され、**「どの位置にどれだけ計算を使うか」を学習する**という点で、Stochastic Depthの「ランダムドロップ」と対をなす重要な研究です。


LLMのレイヤ選択に関する研究は、大きく**2つの方向性**に分かれています。代表的な論文を紹介します。

---

### 方向1：MoE（Mixture of Experts）におけるエキスパート選択

各Transformerレイヤの内部に複数のサブネットワーク（エキスパート）を配置し、入力に応じてどのエキスパートを使うかを選択する研究です。

#### 1. Switch Transformer（2021）
GoogleによるMoEの基盤論文です。Top-1ゲーティング方式を採用し、各トークンを1つのエキスパートに振り分けることで、パラメータ数を兆規模に拡張しながら計算コストを抑えました。<source-chip title="arXiv" url="https://arxiv.org/html/2101.03961v2" />

#### 2. Expert Choice Routing（2022）
GoogleによるMoEの負荷分散改善版です。従来の「トークンがエキスパートを選ぶ」方式を逆転し、「エキスパートがトークンを選ぶ」ことで、特定エキスパートへの偏り（collapse）を防ぎます。<source-chip title="arXiv" url="https://arxiv.org/abs/2202.09368" />

#### 3. MoEfication（2021）
清華大学などによる研究です。既存のTransformerのFFN層を学習後にクラスタリングして複数のエキスパートに分割する手法です。大規模な事前学習から既存モデルをMoE化できる点が特徴です。<source-chip title="arXiv" url="https://arxiv.org/html/2110.01786v3" />

#### 4. Layerwise Recurrent Router for MoE（2024）
レイヤー間でルーターの状態を再帰的に伝播させる手法です。各レイヤー独立にルーティングする従来手法より、レイヤー間の文脈を考慮したエキスパート選択が可能になります。<source-chip title="arXiv" url="https://arxiv.org/pdf/2408.06793v2.pdf" />

#### 5. LExI（2025）
レイヤーごとにアクティブなエキスパート数を適応的に調整する手法です。推論時の計算効率をさらに高める方向の研究です。<source-chip title="arXiv" url="https://arxiv.org/html/2509.02753" />

---

### 方向2：Transformerレイヤそのものの動的選択・スキップ

MoEのようにレイヤー内を分岐するのではなく、**どのレイヤーを実行するか、何回繰り返すか**を入力に応じて動的に決定する研究です。

#### 6. Dynamic Layer Selection in Decoder-Only Transformers（2024）
McGill大学などによる研究です。デコーダー型LLMにおいて「レイヤースキップ」と「早期終了（early exit）」を比較検証しました。重要な発見として、**全レイヤーの平均23.3%だけを使えば、フルモデルと同等の性能が達成できる**構成が存在することを示しました。<source-chip title="arXiv" url="https://arxiv.org/abs/2410.20022" />

#### 7. Dr.LLM: Dynamic Layer Routing in LLMs（2025）
Parameter Lab・MBZUAIなどによる研究です。既存の学習済みLLMに**軽量なルーターを後付け**し、各レイヤーに対して「スキップ」「実行」「繰り返し」の3択を動的に決定します。MCTS（モンテカルロ木探索）で最適なレイヤー構成を探索し、それを教師データとしてルーターを学習します。ARCや数学タスクで**精度向上+平均5レイヤー削減**を実現しています。<source-chip title="arXiv" url="https://arxiv.org/abs/2510.12773" />

#### 8. Read-ME: Router-Decoupled Mixture of Experts（2024）
テキサス大学オースティン校などによる研究です。MoEにおけるルーターとエキスパートを分離し、システムレベルでの共設計を行うフレームワークです。<source-chip title="arXiv" url="https://arxiv.org/html/2410.19123" />

---

### まとめ

| 方向性 | 代表論文 | 選択対象 |
|---|---|---|
| **MoEエキスパート選択** | Switch Transformer, Expert Choice, MoEfication | レイヤー内のサブネットワーク |
| **レイヤー動的選択** | Dynamic Layer Selection, Dr.LLM | Transformerレイヤーそのもの（スキップ/繰り返し） |

あなたの強化学習実験で扱っている「MoEのエキスパートcollapse」問題は、**方向1（MoE）**の文脈に属します。特にSwitch Transformerの負荷分散損失（load-balancing loss）やExpert Choice Routingの「エキスパートがトークンを選ぶ」方式は、あなたの実装と直接関連する研究です。一方、**方向2（レイヤー動的選択）**は、MoEとは別に「どの深さまで計算すればよいか」を学習する新しい研究領域として注目されています。


方向2（レイヤーそのものの動的選択・スキップ）の中で、現在のLLMに応用されるような最も著名な研究成果は、以下の2つです。

---

### 1. LayerSkip（Meta / FAIR, ACL 2024）

**最も実用的で広く知られた研究**です。MetaのFAIRチームによるもので、**学習時にレイヤードロップアウト（layer dropout）を適用し、推論時に早期終了（early exit）と自己投機的デコーディング（self-speculative decoding）を組み合わせる**エンドツーエンドの推論加速手法です。

**なぜ著名か：**
- **実際にモデルが公開されている**：Hugging Faceで `facebook/layerskip-llama` などの学習済みモデルが公開され、誰でも試せる
- **GitHubスター379**、引用数も300超えと、方向2の中で最も実装・検証が進んでいる
- **「学習時の工夫（layer dropout）」と「推論時の工夫（early exit + speculative decoding）」を組み合わせた**点が画期的で、単なる推論時トリックではなく、モデル自体を「浅い層でも使えるように」育てるアプローチ

**核心アイデア：**
学習中にランダムにレイヤーをドロップさせることで、モデルが「途中の層の出力でも正解を出せるように」頑張るようになります。推論時には、浅い層で自信があればそこで出力し、自信がなければ深い層まで計算して自己補正する、という**動的深度**を実現します。<source-chip title="arXiv" url="https://arxiv.org/abs/2404.16710" />

---

### 2. Mixture of Depths（Google DeepMind, 2024）

**「MoEの対となる概念」として大きな注目を集めた**研究です。Google DeepMindによるもので、従来のTransformerが「すべてのトークンに同じ計算量を使う」という非効率を解消します。

**なぜ著名か：**
- **MoE（幅方向のスパース化）に対する「深度方向のスパース化」**という新しいパラダイムを提示した
- **FLOPsを最大50%削減**しつつ、通常のTransformerと同等以上の性能を達成
- 「簡単なトークンはレイヤーをスキップ、難しいトークンはフル計算」という**トークン単位の動的計算配分**を学習で実現

**核心アイデア：**
各Transformerブロックの前にルーターを置き、「このトークンをこのブロックで処理するか／スキップするか」をtop-k選択で決定します。これにより、同じモデル内で**トークンごとに異なる深度**の計算が行われます。MoEが「どのエキスパートを使うか」を選ぶのに対し、MoDは「どのトークンに計算を割くか」を選びます。<source-chip title="arXiv" url="https://arxiv.org/abs/2404.02258" />

---

### どちらが「最も」著名か？

| 観点 | LayerSkip | Mixture of Depths |
|---|---|---|
| **実用性・実装の広まり** | **高**（Metaがモデル公開、GitHub活発） | 中（概念の新規性が高い） |
| **学術的インパクト** | 高（ACL 2024採択） | **高**（新パラダイム提示） |
| **実際の推論加速** | **確認済み**（自己投機的デコーディングと統合） | 理論的FLOPs削減は大きい |
| **既存モデルへの適用** | **後付け可能**（学習済みモデルを再学習） | 学習からやり直しが必要 |

**結論として、現在「最も著名で実用的」なのは LayerSkip です。** Metaが実際にモデルを公開し、推論パイプラインに組み込める形で整備しているため、研究コミュニティ・業界双方で最も参照されています。一方、Mixture of Depthsは**「次の標準アーキテクチャになりうる概念」**として、長期的な影響力で注目されています。

先ほど紹介した **Dr.LLM（2025）** は、この2つの流れを受けつつ「既存の学習済みLLMに後付けで動的ルーティングを追加する」という点で新しい方向性を示していますが、著名度・実用化の面ではまだLayerSkipやMoDに及んでいません。
