RAG（Retrieval-Augmented Generation）は、**LLMに外部知識を検索させてから回答を生成する仕組み**です。  
最近は「単純なベクトル検索＋LLM」から、**マルチモーダル化・リランキング・自己改善・GraphRAG**など、かなり高度な仕組みに進化しています。

---

## 1. 基本フロー（Naive RAG）

まず、古典的なRAGの流れは以下の通りです[PromptingGuide.ai](https://www.promptingguide.ai/research/rag)。

1. **データ準備**
   - ドキュメントをチャンクに分割
   - ベクトルDBに埋め込み（embedding）して格納

2. **検索（Retrieval）**
   - ユーザーの質問を埋め込みに変換
   - ベクトル類似度で上位k件を取得

3. **生成（Generation）**
   - 上位k件をプロンプトに詰め込んでLLMに渡す
   - LLMがそれらを参照して回答を生成

→ 問題点：  
- 類似度だけでは**本当に必要な文書が上位に来ない**  
- ノイズが多いと**幻覚（hallucination）**が増える  
- マルチモーダル（画像・音声など）には対応していない

---

## 2. 最近のRAGの主な進化ポイント

### (1) マルチモーダルRAG（mRAG）
- **テキストだけでなく、画像・音声・動画・構造化データ**も検索対象に含める枠組み[Emergent Mind](https://www.emergentmind.com/topics/multimodal-retrieval-augmented-generation-mrag)。
- 例：
  - **RagVL**：MLLM（マルチモーダルLLM）を**リランカー（reranker）**として使い、  
    テキスト＋画像の候補を再ランキングして精度を上げる[OpenReview](https://openreview.net/forum?id=TPtzZQyiFm)。
  - **SAM-RAG**：自己適応型のマルチモーダルRAGで、  
    クエリの種類に応じて**どのモダリティをどれだけ検索するか**を動的に決める[ResearchGate](https://www.researchgate.net/publication/397633360_SAM-RAG_An_Self-adaptive_Framework_for_Multimodal_Retrieval-Augmented_Generation)。

### (2) 高度な検索・リランキング
- **ハイブリッド検索**：
  - ベクトル検索（semantic）＋キーワード検索（BM25）を組み合わせ、  
    意味的類似度と表層的な一致の両方を考慮[RAGFlow Blog](https://ragflow.io/blog/the-rise-and-evolution-of-rag-in-2024-a-year-in-review)。
- **リランキング（reranking）**：
  - 1段目で大量に候補を取得し、2段目で**より強力なモデル（MLLMやColBERT系）**で再スコアリング[Aman.ai](https://aman.ai/primers/ai/RAG/)。
  - RagVLでは、MLLMをリランカーとして使い、  
    **ノイズが多いマルチモーダル対応**を改善[OpenReview](https://openreview.net/forum?id=TPtzZQyiFm)。

### (3) 自己改善・自己適応（Self-RAG系）
- **Self-RAG**のような枠組みでは、LLM自身が
  - 「この質問には検索が必要か？」
  - 「何件検索すべきか？」
  - 「この検索結果は信頼できるか？」
  を**自己評価**し、必要に応じて**再検索・再生成**を行う[Medium Survey](https://medium.com/@sahin.samia/advancements-in-rag-a-comprehensive-survey-of-techniques-and-applications-b6160b035199)。
- SAM-RAGなども、**クエリの難易度やモダリティに応じて検索戦略を動的に変える**自己適応型の仕組みを採用。

### (4) GraphRAG
- ドキュメント間の関係を**グラフ構造**で表現し、
  - エンティティ（人物・組織・概念）のつながり
  - イベントの時系列・因果関係
  を考慮して検索する[RAGFlow Blog](https://ragflow.io/blog/the-rise-and-evolution-of-rag-in-2024-a-year-in-review)。
- これにより、
  - 「単語が似ているが文脈が違う」文書を区別
  - マルチホップ推論（複数文書をまたぐ推論）がしやすくなる

### (5) クエリプランニング・エージェント的RAG
- **クエリプランニング**：
  - ユーザーの質問を**サブクエリに分解**したり、
  - 「テキスト検索か？画像検索か？構造化データか？」を**ルーティング**するモジュールを追加[Emergent Mind](https://www.emergentmind.com/topics/multimodal-retrieval-augmented-generation-mrag)。
- **エージェント的RAG**：
  - RAGを**ツール呼び出し可能なエージェント**の一部として組み込み、
  - 必要に応じて複数回の検索・推論を繰り返す。

### (6) プライバシー・セキュリティ評価
- マルチモーダルRAGでは、**プライベートな画像・メタデータ**が漏洩するリスクが指摘されています[arXiv](https://arxiv.org/html/2601.17644v1)。
- 最近の研究では、
  - 「ある画像がRAGのデータベースに含まれているか」を推論する攻撃
  - メタデータ（キャプションなど）の漏洩
  を評価し、**プライバシー保護の必要性**を強調しています。

---

## 3. 最近のRAGの典型的なパイプライン（2024–2025）

最近の実用的なRAGシステムは、以下のような多段構成が主流です[PromptingGuide.ai](https://www.promptingguide.ai/research/rag)[Aman.ai](https://aman.ai/primers/ai/RAG/)。

1. **データ前処理**
   - マルチモーダル文書（PDF＋画像＋表など）をパース
   - チャンク分割＋メタデータ付与
   - グラフ構造（GraphRAG）や構造化DBも併用

2. **クエリ理解・プランニング**
   - 質問の意図を分類（テキスト／画像／複合）
   - 必要ならサブクエリ分解・マルチホップ計画

3. **第1段検索**
   - ベクトル検索＋BM25のハイブリッドで広く候補を取得

4. **第2段リランキング**
   - MLLMやColBERT系モデルで**関連度・信頼度**を再評価
   - ノイズを除去し、上位k件を絞り込む

5. **自己評価・再検索（Self-RAG的）**
   - LLMが「この結果で十分か？」を自己評価
   - 不十分ならクエリを修正して再検索

6. **生成**
   - 絞り込んだ文書をプロンプトに組み込み、LLMが回答生成
   - 必要に応じて**出典表示（citation）**も行う

---

## 4. まとめ

最近のRAGは、

- **マルチモーダル対応（mRAG）**で画像・音声・動画も扱える
- **ハイブリッド検索＋リランキング**で精度を大幅に向上
- **Self-RAG・GraphRAG・クエリプランニング**で、  
  より賢く・自己適応的に検索を行う
- **プライバシー・セキュリティ評価**も進み、  
  実運用でのリスク管理が重視されている

という方向に進化しています[RAGFlow Blog](https://ragflow.io/blog/the-rise-and-evolution-of-rag-in-2024-a-year-in-review)[Emergent Mind](https://www.emergentmind.com/topics/multimodal-retrieval-augmented-generation-mrag)。  
これにより、**「最新の外部知識を正確に参照するAIシステム」**としてのRAGの役割が一段と強まっています。

## マルチモーダル埋め込み

テキスト埋め込みと画像埋め込みを**両方使う**場合、大きく分けて以下の3つのアプローチがあります。

1. **共通埋め込み空間（Common Embedding Space）**  
   → テキストと画像を**同じベクトル空間**に埋め込む（CLIP系）

2. **マルチモーダル融合（Multimodal Fusion）**  
   → テキスト埋め込みと画像埋め込みを**別々に計算し、後で統合**する

3. **マルチモーダルLLM（MLLM）を埋め込み器として使う**  
   → GPT-4VやLLaVAなど、**テキスト＋画像を同時に理解できるモデル**で埋め込みを取る

以下、それぞれの手法と特徴を説明します。

---

## 1. 共通埋め込み空間（CLIP系）

### 概要
- **CLIP（Contrastive Language–Image Pre-training）**のようなモデルを使い、  
  テキストと画像を**同じベクトル空間**に埋め込みます[OpenAI CLIP](https://openai.com/research/clip)。
- テキストエンコーダと画像エンコーダを**対照学習（contrastive learning）**で訓練し、  
  「対応するテキストと画像の埋め込みが近くなる」ようにします。

### メリット
- **テキストクエリで画像を直接検索できる**（逆も可能）
- 埋め込み空間が統一されているため、**ハイブリッド検索がシンプル**
- 実装が比較的簡単（Hugging FaceにCLIPモデルが多数公開）

### 実装例
- `openai/clip-vit-base-patch32` など  
  → `CLIPModel`＋`CLIPProcessor`でテキスト・画像の埋め込みを取得[Hugging Face CLIP](https://huggingface.co/openai/clip-vit-base-patch32)。

---

## 2. マルチモーダル融合（別々の埋め込みを統合）

### 概要
- テキスト埋め込み（例：Sentence-BERT）と  
  画像埋め込み（例：ResNet, ViT）を**別々に計算**し、  
  後から**連結・重み付け・アテンション**などで統合します。

### 代表的な統合方法

#### (1) 単純連結（Concatenation）
- テキスト埋め込み \(v_t\) と画像埋め込み \(v_i\) を連結：
  \[
  v = [v_t; v_i]
  \]
- その後、線形層やMLPで次元を調整。

#### (2) 重み付き和（Weighted Sum）
- テキストと画像の重要度に応じて重み付け：
  \[
  v = \alpha v_t + (1 - \alpha) v_i
  \]
- \(\alpha\) は固定値、またはクエリに応じて動的に決定。

#### (3) クロスアテンション（Cross-Attention）
- テキストをQuery、画像をKey/Valueとして**アテンション**をかける（またはその逆）。
- これにより、**テキストと画像の関連部分を強調**した埋め込みを得る。

#### (4) 後段リランキング（Reranking）
- テキスト・画像それぞれで**別々に検索**し、  
  上位候補をMLLMに渡して**関連度を再評価**する方法もあります[OpenReview RagVL](https://openreview.net/forum?id=TPtzZQyiFm)。

### メリット
- 既存のテキスト埋め込みモデル・画像埋め込みモデルを**そのまま流用**できる
- モダリティごとに**最適なモデルを選べる**（例：専門ドメインのテキストモデル＋高精度画像モデル）

### デメリット
- 埋め込み空間が異なるため、**正規化やスケーリング**が必要
- 統合方法の設計（重み・アテンション構造）が**ハイパーパラメータチューニング**の対象になる

---

## 3. マルチモーダルLLM（MLLM）を埋め込み器として使う

### 概要
- GPT-4V, LLaVA, Qwen-VL などの**マルチモーダルLLM**を使い、  
  「テキスト＋画像」を入力とした**最終層の隠れ状態**を埋め込みとして利用します。
- 最近の研究では、MLLMを**リランカー（reranker）**として使う手法も提案されています[OpenReview RagVL](https://openreview.net/forum?id=TPtzZQyiFm)。

### 使い方の例
- **RagVL**：
  - 1段目でCLIPなどで候補を取得
  - 2段目でMLLMに「質問＋候補（テキスト or 画像）」を渡し、  
    **関連度スコア**を出力させて再ランキング[OpenReview RagVL](https://openreview.net/forum?id=TPtzZQyiFm)。
- **埋め込みとして使う場合**：
  - MLLMの最終層の隠れ状態（または特定トークンの表現）を**埋め込みベクトル**として扱い、  
    ベクトルDBに格納。

### メリット
- MLLMは**テキストと画像の意味的関係**を深く理解できるため、  
  高精度な関連度評価が可能。
- 既存のMLLMを**そのままRAGの一部として再利用**できる。

### デメリット
- 推論コストが高い（特に大規模MLLM）
- 埋め込み次元が大きくなりがちで、**インデックス構築・検索コスト**が増える

---

## 4. 実装上の選択肢（まとめ）

| 手法 | 特徴 | 実装のしやすさ | 精度 | コスト |
|------|------|----------------|------|--------|
| **CLIP系（共通空間）** | テキストと画像を同じ空間に埋め込む | ◎（Hugging Faceで簡単） | 高（対照学習済み） | 中 |
| **マルチモーダル融合** | 別々の埋め込みを後で統合 | ○（設計が必要） | 中〜高（設計次第） | 低〜中 |
| **MLLM埋め込み** | MLLMの隠れ状態を埋め込みとして使う | △（モデル依存） | 非常に高 | 高 |
| **MLLMリランキング** | MLLMで関連度を再評価 | ○（RAGの上に載せる） | 非常に高 | 高（2段目のみ） |

---

## 5. 実用的な組み合わせ例

実務では、以下のような**多段構成**がよく使われます[PromptingGuide.ai](https://www.promptingguide.ai/research/rag)。

1. **第1段：CLIP系で広く候補を取得**
   - テキスト・画像を**共通埋め込み空間**で検索
   - 高速で広範囲をカバー

2. **第2段：MLLMでリランキング**
   - 上位候補をMLLMに渡し、**関連度スコア**を再計算
   - ノイズを除去し、精度を向上

3. **最終段：MLLMで生成**
   - 選ばれたテキスト＋画像をプロンプトに詰め、回答生成

このように、**共通埋め込み空間（CLIP）＋MLLMリランキング**の組み合わせが、  
現在のマルチモーダルRAGで**バランスの良い選択肢**としてよく採用されています[Emergent Mind](https://www.emergentmind.com/topics/multimodal-retrieval-augmented-generation-mrag)。

