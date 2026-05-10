
画像を認識する[VLM](https://yoshishinnze.hatenablog.com/entry/2026/01/24/000000)（Vision-Language Model）は**フィジカルAI（Physical AI）** や**エンボディドAI（Embodied AI）** の分野で、**「ロボットの頭脳」として使われている**という話が2025〜2026年頃から増えています。

>__1. 「フィジカルAI」と「エンボディドAI」でのVLMの役割__
>フィジカルAI／エンボディドAIは、**実世界の環境で行動するエージェント（ロボットなど）** を対象としたAI分野です。ここでは、VLMは単なる「画像を見て説明するモデル」ではなく、**視覚と言語を統合して行動を生成するモデル**として使われます。
>- サーベイ論文『A Survey on Vision-Language-Action Models for Embodied AI』では、  
>  - LLMやVLMの成功を受けて、**Vision-Language-Action (VLA) モデル**という新しいカテゴリが登場し、  
>  - VLMをベースに**行動（action）モダリティを追加**することで、ロボット制御に応用していると説明されています[arXiv](https://arxiv.org/html/2405.14093v8)。
>このように、VLMは **「視覚と言語の理解」を担当し、その上に「行動生成」を乗せる** 形でフィジカルAIに組み込まれています。
>__2. VLA（Vision-Language-Action）モデルとしての展開__
>VLMをベースに**行動モダリティを追加したVLAモデル**が、フィジカルAIの主流になりつつあります。
>- 先述のサーベイ論文では、VLAモデルを  
>  - **個別コンポーネント（VLM＋行動予測器）の組み合わせ**  
>  - **低レベル行動を直接予測するVLAベースの制御ポリシー**  
>  に分類し、ロボットタスク（ナビゲーション、マニピュレーションなど）への応用を整理しています[arXiv](https://arxiv.org/html/2405.14093v8)。

そんなフィジカルAIに必要なVLMですが、画像で理解した内容と全然違う文章を生成する**視覚的幻覚（visual hallucination）** という問題があります。
そこで現実世界で起こるイベントを視覚でとらえる汎用視覚モデルとしての課題と、対策について掘り下げていきます。

## 視覚的幻覚とは

VLMが画像で理解した内容と**全く異なる**文章を生成する現象は、一般に視覚的幻覚と呼ばれ、近年の研究でその原因がかなり詳細に分析されています。以下、論文などの客観情報に基づいて主な要因を整理します。


### 1. 学習データのノイズ・バイアスとアライメント不足

**サーベイ論文**では、幻覚の主な原因として「**学習データ要因**」が挙げられています。

- Liu et al. (2024) のサーベイ『Understanding Hallucinations in Large Visual and Language Models』では、  
  - 事前学習データセットにおける**テキストと画像の対応の不完全さ**  
  - **誤ったキャプション**や**画像と無関係なテキスト**が混入していること  
  が幻覚の主要因であると指摘されています[ACM Digital Library](https://dl.acm.org/doi/pdf/10.1145/3811409?download=true)。

- GitHubリポジトリ『Awesome-LVLM-Hallucination』でも、  
  - **言語事前分布（language prior）**  
  - **視覚コンテキストの不足**  
  - **訓練データのバイアスや誤情報**  
  が幻覚の原因として挙げられています[GitHub](https://github.com/NishilBalar/Awesome-LVLM-Hallucination)。

これらは、**画像とテキストのアライメントが不完全**な状態で学習されるため、モデルが「画像を見なくても出せる典型的な文章」を優先してしまうことを示しています。

### 2. マルチモーダル融合の不十分さと「モダリティ・ショートカット」

VLMは画像エンコーダとLLMをプロジェクタで接続しますが、**両モダリティがきちんと融合されず、どちらか一方に依存してしまう**ことが幻覚の原因とされています。

- Wang et al. (2025) の『Treble Counterfactual VLMs: A Causal Approach to Hallucination』では、  
  - VLMの出力に対する**視覚モダリティの直接効果（NDE: Natural Direct Effect）** と**テキストモダリティの直接効果**を因果グラフで分析し、  
  - 幻覚は「**意図しないモダリティ・ショートカット**」、すなわち**画像を経由せずにテキスト側だけで出力を決めてしまう経路**から生じると仮説を立てています[arXiv](https://arxiv.org/html/2503.06169v2)。

この研究は、VLMの内部で**画像情報が十分に活用されず、LLM側の言語事前分布に過度に依存**していることを因果的に示唆しています。

### 3. 時空間・マルチ画像推論における「表面的ショートカット」

静止画だけでなく、**複数画像や動画を用いた推論**では、時系列の因果関係を理解せずに「見た目の類似性」などの**表面的ショートカット**に依存し、幻覚が顕著になることが報告されています。

- Zhang et al. (2026) の『A Progressive Training Strategy for Vision-Language Models to Counteract Spatio-Temporal Hallucinations in Embodied Reasoning』では、  
  - 「**マルチ画像推論幻覚（multi-image reasoning hallucination）** 」という現象を指摘し、  
  - 順方向クエリと逆方向クエリで性能が大きく乖離する（70%以上のギャップ）ことを示し、  
  - モデルが**真の因果理解ではなく、見た目の類似性などの表面的ショートカット**に依存していると分析しています[arXiv](https://arxiv.org/html/2604.10506v1)。

このように、**動的な空間関係や因果関係の理解が不十分**な場合、画像と無関係な推論結果（幻覚）が生じやすくなります。


### 4. アライメント手法の限界と「細粒度セマンティクス」の不足

最近の研究では、**DPO（Direct Preference Optimization）などのアライメント手法**が、MLLM（Multimodal LLM）において**細粒度のセマンティクスを十分に捉えられず、ショートカット学習を助長する**ことが指摘されています。

- Li et al. (2025) の『Mitigating Visual Hallucinations via Semantic Curriculum Preference Optimization in MLLMs』では、  
  - 既存のDPOベースのアライメントが**微細な意味の違い**を捉えきれず、  
  - 画像と矛盾する応答を生成する「視覚的幻覚」を十分に抑制できないと報告しています[OpenReview](https://openreview.net/forum?id=DztuuQCn13)。

この論文では、**意味的カリキュラムに基づくPreference Optimization（SCPO）** を提案し、  
- 容易な例から難しい例へと段階的に学習させることで、  
- 幻覚率を最大62.9%削減できることを示しています[OpenReview](https://openreview.net/forum?id=DztuuQCn13)。

これは、**単に「正解／不正解」を学習させるだけでは不十分**で、**画像とテキストの細かい意味対応**をきちんと学習させないと幻覚が残ることを示しています。

### 5. 注意機構と幻覚検出の観点からの分析

**注意（attention）機構**に着目した研究では、幻覚が生じる際に**画像領域への注意が不十分**であることが示されています。

- 2025年のACL Findings論文『Visual Attention Guided Hallucination Detection and Elimination』では、  
  - 画像とテキストの**注意マップ**を用いて幻覚を検出・除去する手法（VADE）を提案し、  
  - 幻覚が生じる応答では、**画像の関連領域への注意が低い**ことを示しています[ACL Anthology](https://aclanthology.org/2025.findings-acl.773.pdf)。

この結果は、**画像情報が十分に“見られていない”状態で文章が生成されている**ことを裏付けています。


### 6. 体系的幻覚の検出と評価

最近の研究では、幻覚が**ランダムな誤りではなく、特定のパターンで体系的に発生する**ことも明らかになっています。

- ICCV 2025の論文『DASH: Detection and Assessment of Systematic Hallucinations of VLMs』では、  
  - VLMの**体系的幻覚（systematic hallucinations）**を検出・評価する枠組みを提案し、  
  - 特定の入力パターンやデータバイアスに起因する幻覚を定量的に分析しています[CVF OpenAccess](https://openaccess.thecvf.com/content/ICCV2025/papers/Augustin_DASH_Detection_and_Assessment_of_Systematic_Hallucinations_of_VLMs_ICCV_2025_paper.pdf)。

これにより、**特定の物体カテゴリや文脈で繰り返し幻覚が生じる**ことが確認されており、単なるノイズではなく**モデル内部のバイアス・構造的問題**であることが示されています。

## 視覚的幻覚は評価できるか
見たはずの画像から、あんまり関係のない思考がされる訳なので大問題です。
当然、視覚的幻覚がどの程度起こるかは把握しておきたいという面があります。そこで評価指標が重要となります。
VLMの視覚的幻覚は、**専用のベンチマーク・指標・データセット**を用いて評価されています。2024〜2025年頃のサーベイや論文では、主に以下のような枠組みで評価が行われています。

### 1. 代表的なベンチマークと指標

__(1) HallusionBench__
- **HallusionBench**は、MLLM（Multimodal LLM）の**視覚的幻覚**を評価するためのベンチマークです。
- 画像＋テキスト入力に対して、モデルが生成した回答が**画像内容と矛盾していないか**を評価します。
- サーベイ『Hallucination of Multimodal Large Language Models: A Survey』では、HallusionBenchを**LLMベースの幻覚評価指標**の一つとして紹介しています[arXiv](https://arxiv.org/html/2404.18930v2)。

__(2) POPE（Polling-based Object Probing Evaluation）__
- **POPE**は、画像内の**オブジェクトの有無**を質問し、モデルが「存在する／しない」と正しく答えられるかを評価するベンチマークです。
- 画像に**存在しないオブジェクト**について質問し、モデルが「存在する」と答える割合を**幻覚率**として測定します。
- 同じく上記サーベイで、**判別タスク型の幻覚ベンチマーク**として分類されています[arXiv](https://arxiv.org/html/2404.18930v2)。

__(3) M-HalDetect / ViGoR__
- **M-HalDetect**は、**マルチモーダル幻覚検出データセット**として提案されており、  
  - 画像＋キャプションに対して「幻覚を含む／含まない」をラベル付けし、  
  - モデルや検出器の**Precision / Recall / F1**などを評価します[ResearchGate](https://www.researchgate.net/publication/397200029_Calibrated_Self-Rewarding_Vision_Language_Models)。
- **ViGoR**は、視覚的幻覚検出のターゲットデータセットとして使われ、  
  - 注意マップベースの検出器（VADEなど）が**PR-AUC**で評価されています[ACL Anthology](https://aclanthology.org/2025.findings-acl.773.pdf)。

__(4) MMHal-Bench, AMBER など__
- **MMHal-Bench**や**AMBER**は、**生成タスク型の幻覚ベンチマーク**として分類されており、  
  - 画像説明やVQAタスクで、生成文が画像と矛盾する割合を評価します[arXiv](https://arxiv.org/html/2404.18930v2)。

### 2. 評価タスクの種類（サーベイによる分類）

サーベイ『Hallucination of Multimodal Large Language Models: A Survey』では、幻覚評価を以下のように整理しています[arXiv](https://arxiv.org/html/2404.18930v2)。

__(1) 判別タスク（Discriminative Task）__
- 画像＋質問に対して、**選択式（Yes/No, 多肢選択）** で回答させる。
- 例：POPE, MME, MMBench など。
- 指標：**Accuracy, F1**など。

__(2) 生成タスク（Generative Task）__
- 画像説明や自由回答VQAで、**生成文が画像と矛盾するか**を評価。
- 例：MMHal-Bench, AMBER, HallusionBench（一部）など。
- 指標：  
  - **CHAIR**（Caption Hallucination Assessment with Image Relevance）  
  - **LLMベースの自動評価**（GAVIE, HaELM, HallusionBenchなど）  
  - **人手評価（Manual Accuracy / F1）**

__(3) 幻覚検出タスク（Hallucination Detection）__
- 与えられた画像＋キャプション（またはモデル出力）が**幻覚を含むかどうか**を判定するタスク。
- 例：M-HalDetect, ViGoR。
- 指標：**Precision, Recall, F1, PR-AUC**など。

### 3. 評価指標の具体例

__(1) CHAIR（Caption Hallucination Assessment with Image Relevance）__
- 画像説明文に含まれる**オブジェクトが画像中に存在するか**を自動判定し、  
  - **Hallucination Rate**（幻覚率）を計算します。
- サーベイでは、**CHAIR**を代表的な幻覚指標として紹介しています[arXiv](https://arxiv.org/html/2404.18930v2)。

__(2) LLMベースの自動評価__
- HallusionBenchやGAVIE、HaELMなどでは、**評価用LLM**を用いて、  
  - 生成文が画像内容と矛盾していないか  
  - 画像に基づいているか  
  を自動判定します。
- これにより、大規模なベンチマークを**自動でスコアリング**できます[arXiv](https://arxiv.org/html/2404.18930v2)。

__(3) 注意マップベースの検出（VADEなど）__
- 『Visual Attention Guided Hallucination Detection and Elimination』では、  
  - 画像とテキストの**注意マップ**を用いて幻覚を検出する手法（VADE）を提案し、  
  - M-HalDetect / ViGoRで**PR-AUC**を指標に評価しています[ACL Anthology](https://aclanthology.org/2025.findings-acl.773.pdf)。

## ベンチマークの課題
ここまで幻覚を測定する方法について説明しました。
しかし、VLMの幻覚ベンチマーク（HallusionBench, POPE, M-HalDetect, ViGoRなど）には、**再現性や信頼性に関する課題**が複数指摘されています。主なポイントは以下の通りです。

![1778407705724](image/tech/1778407705724.png)

### 1. 評価指標の「LLM依存性」と再現性の問題

多くの幻覚ベンチマークでは、**評価用LLM（例：GPT-4など）** を使って「生成文が画像と矛盾していないか」を自動判定しています。

- HallusionBenchやGAVIE、HaELMなどは、**LLMベースの自動評価**を採用しています[arXiv](https://arxiv.org/html/2404.18930v2)。
- しかし、評価用LLM自体が**幻覚を起こす可能性**があり、  
  - 評価モデルのバージョン違い  
  - プロンプト設計の違い  
  - サンプリング設定（temperatureなど）  
  によってスコアが変動し、**再現性が低下する**という指摘があります。

サーベイ『Hallucination of Multimodal Large Language Models: A Survey』でも、  
- LLMベース評価は**スケーラブル**である一方、  
- **評価モデル自体のバイアスや幻覚**が問題になり得ると述べられています[arXiv](https://arxiv.org/html/2404.18930v2)。


### 2. ベンチマークの「過度な最適化」と汎化性能の低下

- 特定のベンチマーク（例：POPE, HallusionBench）で高スコアを出すようにモデルを訓練すると、  
  - **そのベンチマーク特有のパターン**に過適合し、  
  - 実世界の多様な入力に対しては幻覚が減らない、という「**ベンチマーク・ゲーミング**」が起こり得ます。

サーベイでは、  
- ベンチマークが**限定的なタスクやデータ分布**に基づいているため、  
- **実アプリケーションでの幻覚挙動を必ずしも反映しない**可能性が指摘されています[arXiv](https://arxiv.org/html/2404.18930v2)。


### 3. 根本的な課題

「評価指標のLLM依存性」と「ベンチマークの過度な最適化」という2つの課題は、見た目は別々ですが、**共通の根本的な問題**に起因しています。

__共通課題1：Ground Truth（正解）の定義が曖昧／不完全__

VLMの幻覚評価では、本来「**画像内容と生成文が一致しているか**」を評価したいはずですが、

- 画像内容そのものが**曖昧**（例：影と実物の区別、部分的な隠れ、抽象的な絵）
- キャプションや説明文が**推測を含む**（例：「多分〜だろう」「〜のように見える」）
- 人間の解釈にも**個人差**がある

といった理由で、**絶対的なGround Truth（正解）が存在しない**ケースが多くあります。

その結果、

- **LLMベース評価**では、  
  - 「正解」の代わりに**評価用LLMの判断**を頼ることになり、  
  - 評価モデル自体のバイアスや幻覚が問題になります[arXiv](https://arxiv.org/html/2404.18930v2)。

- **ベンチマーク・ゲーミング**では、  
  - 「正解」が**ベンチマーク内の限定的な分布**に縛られているため、  
  - モデルがその分布に過適合し、実世界の多様な入力では通用しなくなります[arXiv](https://arxiv.org/html/2404.18930v2)。

つまり、**「何が正しい回答か」を厳密に定義できない**ことが、両方の課題の根っこにあります。

__共通課題2：評価が「プロキシ（代理指標）」に依存している__

もう一つの根本問題は、**真に評価したいもの（実世界での信頼性）を直接測れず、代理指標に頼らざるを得ない**ことです。

- LLMベース評価は、  
  - 「人間がどう判断するか」の**代理**としてLLMを使っていますが、  
  - LLM自体が不完全であり、**評価モデルの挙動が変わるたびにスコアが変動**します。

- ベンチマーク・ゲーミングは、  
  - 「実世界での幻覚率」の代理として**特定ベンチマークのスコア**を使っていますが、  
  - ベンチマークが実世界分布の**ごく一部**しかカバーしていないため、**代理指標と真の目的が乖離**します。

サーベイでも、  
- LLMベース評価は**スケーラブル**である一方、**評価モデル自体のバイアスや幻覚**が問題になり得る[arXiv](https://arxiv.org/html/2404.18930v2)、  
- ベンチマークが**限定的なタスクやデータ分布**に基づいているため、**実アプリケーションでの挙動を必ずしも反映しない**[arXiv](https://arxiv.org/html/2404.18930v2)  
と指摘されており、いずれも **「代理指標の限界」** が根本にあります。

## 対応策

ここまで問題視してきた視覚幻覚に関する性能を正確に把握する決め手となる方法について検討してみます。
「Ground Truthの曖昧さ」と「代理指標依存」という2つの根本課題に**正確に手を打つ対策**は、主に次の2つです。

### 1. 実タスク・エージェント環境での評価

**何を評価したいか**  
- 本来評価したいのは「**実世界での信頼性**」（例：ロボットが正しく行動できるか、ユーザーが誤解しないか）です。

**なぜこれが根本課題に効くか**  
- Ground Truthの曖昧さ：  
  - 実タスクでは、「**行動の成否**」や「**タスク達成度**」が比較的明確なGround Truthになります。  
    - 例：ロボットが「ドアを開ける」タスクで、実際にドアが開いたかどうか。  
  - 画像内容の解釈の曖昧さを**行動結果で吸収**できるため、Ground Truthの定義がしやすくなります。

- 代理指標依存：  
  - ベンチマークスコアのような代理指標ではなく、**実際のタスク成功率**を直接測るため、  
  - 「ベンチマーク・ゲーミング」に依存しない評価が可能になります。

サーベイ『A Survey on Vision-Language-Action Models for Embodied AI』でも、  
- VLAモデルを**実タスク（ナビゲーション、マニピュレーション）**で評価する重要性が強調されており、  
- ベンチマーク上のスコアと実運用のギャップを埋める方向性が示されています[arXiv](https://arxiv.org/html/2405.14093v8)。


### 2. 因果的・内部表現ベースの評価

**何を評価したいか**  
- モデル内部で**画像とテキストがきちんと融合されているか**、  
- あるいは**どちらか一方のモダリティに過度に依存していないか**を評価します。

**なぜこれが根本課題に効くか**  
- Ground Truthの曖昧さ：  
  - 出力テキストの「正しさ」を直接定義する代わりに、  
  - **「画像情報がきちんと使われているか」という内部状態**を評価対象にします。  
  - 例：Treble Counterfactual VLMsでは、視覚モダリティの**直接効果（NDE）**を因果グラフで分析し、  
    - 「画像を経由せずテキスト側だけで出力を決めている経路」を特定します[arXiv](https://arxiv.org/html/2503.06169v2)。

- 代理指標依存：  
  - LLMベース評価のような**外部モデルに依存せず**、  
  - モデル内部の因果関係・注意マップを直接観測するため、**評価プロキシの不完全さ**を回避できます。  
  - 例：VADEでは、画像とテキストの**注意マップ**を用いて幻覚を検出し、PR-AUCで評価しています[ACL Anthology](https://aclanthology.org/2025.findings-acl.773.pdf)。

## 総括

- **VLMはフィジカルAI／エンボディドAIの「頭脳」**として使われ、**Vision-Language-Action (VLA)** モデルに発展し、ロボット制御に応用されている[arXiv](https://arxiv.org/html/2405.14093v8)。

- **視覚的幻覚**（画像と無関係な文章生成）の主因は、  
  - 学習データのノイズ・バイアスとアライメント不足[ACM Digital Library](https://dl.acm.org/doi/pdf/10.1145/3811409?download=true)、  
  - マルチモーダル融合の不十分さ（モダリティ・ショートカット）[arXiv](https://arxiv.org/html/2503.06169v2)、  
  - 時空間推論での表面的ショートカット[arXiv](https://arxiv.org/html/2604.10506v1)、  
  - アライメント手法の限界（細粒度セマンティクス不足）[OpenReview](https://openreview.net/forum?id=DztuuQCn13)、  
  - 注意機構の偏り[ACL Anthology](https://aclanthology.org/2025.findings-acl.773.pdf)、  
  - 体系的幻覚（特定パターンでの繰り返し）[CVF OpenAccess](https://openaccess.thecvf.com/content/ICCV2025/papers/Augustin_DASH_Detection_and_Assessment_of_Systematic_Hallucinations_of_VLMs_ICCV_2025_paper.pdf)。

- **評価**は、HallusionBench, POPE, M-HalDetect, ViGoR, MMHal-Benchなどで、  
  - 判別タスク（Yes/No QA）、生成タスク（キャプション・VQA）、幻覚検出タスクに分類され、  
  - CHAIR, LLMベース評価、注意マップベース検出（VADE）などで定量化される[arXiv](https://arxiv.org/html/2404.18930v2)[ACL Anthology](https://aclanthology.org/2025.findings-acl.773.pdf)。

- **ベンチマークの課題**は、  
  - 評価指標のLLM依存性（再現性低下）[arXiv](https://arxiv.org/html/2404.18930v2)、  
  - ベンチマーク・ゲーミング（過度な最適化で汎化性能低下）[arXiv](https://arxiv.org/html/2404.18930v2)。  
  根本的には、**Ground Truthの曖昧さ**と**代理指標依存**が原因。

- **対策**として、  
  - **実タスク・エージェント環境での評価**（行動の成否を直接測る）[arXiv](https://arxiv.org/html/2405.14093v8)、  
  - **因果的・内部表現ベースの評価**（画像とテキストの融合状態を内部で評価）[arXiv](https://arxiv.org/html/2503.06169v2)[ACL Anthology](https://aclanthology.org/2025.findings-acl.773.pdf)  
  が、根本課題に最も直接的に効くアプローチとして注目されている。

