
[LLMのレイヤを選択する手法として(レイヤの打ち切りを選択する手法として)DeeBERT](https://yoshishinnze.hatenablog.com/entry/2026/09/22/043000)を説明しました。
その後も問題に応じてレイヤを使い分けるという研究は続けられています。
本日説明するのはそんな後続手法で、現在につながるレイヤ選択の手法であるDepth-Adaptive Transformerです。

## 論文明細

[Depth-Adaptive Transformer](https://arxiv.org/abs/1910.10073) の論文明細を以下にまとめます。

### 論文情報

本論文はLLM関係のトップカンファレンスの一つICLRに採択されています。

| 項目 | 内容 |
|------|------|
| **タイトル** | Depth-Adaptive Transformer |
| **著者** | Maha Elbayad, Jiatao Gu, Édouard Grave, Michael Auli |
| **発表会議** | [ICLR 2020](https://iclr.cc/virtual/2020/poster/1770) (International Conference on Learning Representations) |
| **発表年** | 2020年 |
| **arXiv** | [1910.10073](https://arxiv.org/abs/1910.10073) |
| **OpenReview** | [forum?id=SJg7KhVKPH](https://openreview.net/forum?id=SJg7KhVKPH) |
| **著者所属** | Maha Elbayad: Univ. Grenoble Alpes（研究はFacebook AI Researchインターン中に実施）; Jiatao Gu, Édouard Grave, Michael Auli: Facebook AI Research |
| **Meta Research** | [research.facebook.com/publications/depth-adaptive-transformer](https://research.facebook.com/publications/depth-adaptive-transformer/) |
| **HAL（フランス学術アーカイブ）** | [inria.hal.science/hal-02422914](https://inria.hal.science/hal-02422914) |


### 補足

- **ICLR (International Conference on Learning Representations)** は、深層学習・表現学習分野における世界トップクラスの国際会議です。OpenReviewによるダブルブラインド査読を採用しており、本論文も査読を経て採択されています。[ICML](https://iclr.cc/virtual/2020/poster/1770)
- 本論文は2019年10月22日に [arXiv](https://arxiv.org/abs/1910.10073) に先行公開された後、ICLR 2020に採択されました。
- 第一著者の Maha Elbayad 氏は、当時 Univ. Grenoble Alpes（フランス・グルノーブル大学）の博士課程学生で、Facebook AI Research（FAIR）でのインターンシップ中に本研究を行いました。
- 共著者の Michael Auli 氏、Édouard Grave 氏、Jiatao Gu 氏はいずれも FAIR（現 Meta AI）の研究者です。特に Michael Auli 氏は音声認識・機械翻訳分野で広く知られる研究者です。
- ICLRの論文は通常 proceedings として出版されますが、DOIの付与は論文によって異なります。本論文の主要な参照先は [arXiv:1910.10073](https://arxiv.org/abs/1910.10073) および [OpenReview](https://openreview.net/forum?id=SJg7KhVKPH) となります。<source-chip title="arXiv" url="https://arxiv.org/abs/1910.10073" />





## 実験

今回実施した実験は、入力されたデータの難易度に応じて推論に使用するトランスフォーマーの層数（深さ）を動的に変更する「Depth-Adaptive Transformer（動的層数適応型トランスフォーマー）」の構築と挙動検証です。

実験の内容、アーキテクチャの構成、および使用したデータセットの詳細は以下の通りです。

### 1. 実験の内容（学習メカニズム）

モデルが「高い正解率の維持」と「計算コスト（通過層数）の削減」を自律的に両立できるかを検証しました。

* **微分可能な動的ルーティング**: 各層で推論を「打ち切る（Exit）」か「継続する（Continue）」かの二値判定を行うにあたり、Gumbel-Softmax法を使用しました。これにより、離散的な分岐判定を維持したまま、ネットワーク全体をエンドツーエンドで誤差逆伝播（学習）できるようにしています。
* **2つの損失関数の同時最適化**: 正解を導くための「タスク損失（Cross Entropy）」と、通過した層数の期待値を最小化する「深度ペナルティ（Depth Loss）」を加算して最適化しました。
* **ウォームアップ・スケジューリング**: 学習初期からペナルティを与えると、モデルが「精度を諦めて全データを1層目で退出させる」という局所最適解に陥ることが判明したため、最初の12エポックはペナルティを0にして全8層で問題を解かせ、その後に徐々にペナルティを課す仕組み（Warmup）を導入して解決しました。

### 2. レイヤの構成（アーキテクチャ）

PyTorchベースの最大8層からなるカスタムTransformerエンコーダを構築しました。

* **基本仕様**: 隠れ層の次元数 `d_model=128`、アテンションヘッド数 `nhead=4`、フィードフォワード層次元数 `dim_ff=256`。
* **Depth-Adaptive Block（各層の内部構造）**:
通常のTransformerブロック（Multi-Head Attention + FFN + LayerNorm）に加え、以下の2つの専用モジュールを各層に内包しています。
* **Exit Gate（退出ゲート）**: 先頭トークンの表現を受け取り、線形層で「Exit（0）」か「Continue（1）」のロジットを計算します。
* **Classifier（分類器）**: その層で推論を打ち切った場合に、最終的なクラス分類（0か1か）を出力します。


* **推論フロー**: 入力データが各層を通過するたびにゲートが判定を行い、Exitの確率が高まった時点で実質的に計算が打ち切られ、その層の予測結果が最終出力として採用されます。

### 3. 使用したデータセット

モデルに「問題の難易度に応じて層の深さを使い分ける」ことを強制するため、長さ16のシーケンスデータ（語彙サイズ50）に対して、以下3つの難易度（Tier）を均等に混在させたアルゴリズム的合成データを生成して使用しました。

* **Tier 1: 局所判定タスク（Easy）**
* **ルール**: シーケンス内の特定の1要素（インデックス1の値）が閾値（25）より大きいかを判定します。
* **想定層数**: 1〜2層（局所的な参照のみで即答可能）。


* **Tier 2: 広域比較タスク（Medium）**
* **ルール**: シーケンスの前半部分の最大値と、後半部分の最小値を抽出して大小を比較します。
* **想定層数**: 3〜5層（シーケンス全体へのAttentionと、中間状態の比較が必要）。


* **Tier 3: 長距離状態走査タスク（Ultra-Hard）**
* **ルール**: 先頭から末尾まで要素を1つずつ読み込み、「奇数・偶数」によって次の状態（State）に対する計算ルール（加算や乗算のモジュロ演算）が切り替わる連鎖的なパリティ問題です。
* **想定層数**: 7〜8層（1回のAttentionホップでは原理的に解けず、層を重ねて状態を逐次更新し続けることが必須）。



このデータと学習手法を組み合わせたことで、学習終盤には簡単な問題は浅い層で、難しい問題は最深層まで使って回答する「3峰性の動的なレイヤー退出分布」が確認できる実験となりました。
