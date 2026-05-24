ハルシネーション評価用のデータセットは、ここ数年でかなり増えています。  
代表的なものを「**一般タスク向け**」「**医療・専門ドメイン向け**」「**RAG（検索＋生成）向け**」「**マルチモーダル・VLM向け**」に分けて紹介します。

---

## 1. 一般タスク向け（汎用LLMのハルシネーション評価）

### HalluVerse25
- **概要**：多言語・細粒度のLLMハルシネーション評価ベンチマーク[arXiv: HalluVerse25](https://arxiv.org/abs/2503.07833)。
- **特徴**：
  - 英語・中国語・日本語など複数言語に対応
  - ハルシネーションの種類（事実誤り、矛盾、捏造など）を細かくラベル付け
  - モデルがどのタイプのハルシネーションを起こしやすいかを分析可能

### HaluEval
- **概要**：LLMのハルシネーション検出能力を評価する大規模ベンチマーク[GitHub: HaluEval](https://github.com/RUCAIBox/HaluEval)。
- **特徴**：
  - QA・対話・要約など複数タスクを含む
  - 既存タスクデータ（HotpotQAなど）を元に、ChatGPTでハルシネーションサンプルを自動生成
  - 「正解回答」と「ハルシネーション回答」が混在する形式で、モデルに「これはハルシネーションか？」を判定させる

### HalluLens
- **概要**：Wikipedia QAなどに基づくハルシネーション評価ベンチマーク[ACL Anthology: HalluLens](https://aclanthology.org/2025.acl-long.1176.pdf)。
- **特徴**：
  - PreciseWikiQA, LongWiki, NonExistentEntities など複数タスク
  - 「存在しないエンティティ」に関する質問を混ぜ、モデルがどれだけ「知らないことを知らない」かを評価
  - 誤った拒否（False Refusal）や誤った受容（False Accept）も計測

### Hallucinations Leaderboard（Vectara / Hugging Face）
- **概要**：複数LLMのハルシネーション傾向を比較するリーダーボード＋評価データセット[Hugging Face Blog](https://huggingface.co/blog/leaderboard-hallucinations)[GitHub: hallucination-leaderboard](https://github.com/vectara/hallucination-leaderboard)。
- **特徴**：
  - 公開LLM（GPT-4系, Llama系, Gemini系など）を共通ベンチマークで比較
  - コンテキストあり／なし、指示追従、事実性など複数観点で評価
  - データセット自体も公開されており、自前評価にも利用可能

---

## 2. 医療・専門ドメイン向け

### MedHallu
- **概要**：医療LLMのハルシネーション検出に特化したベンチマーク[MedHallu 公式サイト](https://medhallu.github.io)。
- **特徴**：
  - PubMedQA から派生した 10,000 の医療QAペア
  - 「ハルシネーションあり／なし」のバイナリ判定タスク
  - GPT-4o, Llama-3.1, UltraMedical など最新モデルでもF1が0.6台と苦戦する難易度
  - 「わからない」カテゴリを導入することで精度向上を確認

---

## 3. RAG（検索＋生成）向け

### LibreEval
- **概要**：RAGにおけるハルシネーション評価のための大規模オープンソースデータセット[Arize: LibreEval](https://arize.com/llm-hallucination-dataset)。
- **特徴**：
  - LibreEval1.0 として、最大級のRAGハルシネーションデータセットを公開
  - Qwen2-1.5B-Instruct をファインチューニングしたハルシネーション検出モデルも提供
  - 合成データだけでなく、実データ由来のハルシネーションも含む
  - ドメイン・質問タイプ・データソースを横断的に評価可能

### RAGハルシネーション評価（Cleanlab など）
- **概要**：複数の公開RAGデータセット上で、RAGAS, G-eval, LLM-as-a-judge などのハルシネーション検出手法を比較[Cleanlab Blog](https://cleanlab.ai/blog/rag-tlm-hallucination-benchmarking)。
- **特徴**：
  - 既存RAGデータセット（例：Natural Questions, HotpotQA, TriviaQA など）を再利用
  - Precision/Recall で「どの検出手法がどのデータセットで強いか」を分析
  - 自前のRAGシステム評価にも応用しやすい

---

## 4. マルチモーダル・VLM向け

VLM（Vision-Language Model）のハルシネーション評価は、まだ「専用ベンチマーク」が少ないですが、以下のようなアプローチが一般的です。

- **既存VQAデータセットの拡張**  
  - VQAv2, GQA, TextVQA などに、「画像と矛盾する回答」や「画像に存在しない物体を言及する回答」を追加
  - LLMで合成したハルシネーション回答をフィルタリングして混ぜる（前回回答で紹介したCLIP＋VLM as a judge など）

- **合成データ＋人手チェック**  
  - 画像とテキストの不整合サンプルをLLM/VLMで自動生成し、人間がラベル付け
  - 日本語VLM向けでは、合成データのフィルタリング手法（CLIP＋VLM as a judge）が提案されています[ANLP 2025 論文](https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/Q8-16.pdf)。

---

## 5. どのデータセットを選ぶべきか

- **汎用LLMの事実性・ハルシネーション傾向を評価**  
  → HalluVerse25, HaluEval, HalluLens, Vectara Hallucinations Leaderboard が代表的です。

- **医療・法律・金融など専門ドメイン**  
  → MedHallu のようなドメイン特化ベンチマークをまず参照し、必要なら自前データで拡張します。

- **RAGシステム（検索＋生成）のハルシネーション**  
  → LibreEval や Cleanlab のRAGベンチマークが参考になります。

- **VLM（画像＋テキスト）**  
  → 現状は専用ベンチマークが少ないため、既存VQAデータセット＋合成ハルシネーションサンプルを組み合わせて自前評価セットを作るケースが多いです。

---

## 6. まとめ

ハルシネーション評価用データセットは、
- 一般タスク向け：HalluVerse25, HaluEval, HalluLens, Vectara Hallucinations Leaderboard
- 医療向け：MedHallu
- RAG向け：LibreEval, Cleanlab RAGベンチマーク
- VLM向け：既存VQA＋合成データ

といった選択肢があります。  
用途に応じて「どのドメイン・タスクで」「どの種類のハルシネーション」を評価したいかを明確にし、それに合ったデータセットを選ぶとよいでしょう。



結論から言うと、**HalluVerse25 と HaluEval は、Google Colab 上で「データセットのダウンロードから評価まで」を比較的スムーズに行える環境が整っています**。  
一方、MedHallu や LibreEval などはデータセット自体は公開されていますが、Colab 向けの公式ノートブックは（少なくとも現時点では）見当たらないため、自分で評価コードを書く必要があります。

以下、代表的なものについて「Colab でそのまま使えるか」という観点で整理します。

---

## 1. HalluVerse25（汎用・多言語ハルシネーション評価）

- **Colab 対応**：**公式に「再現可能な Colab パイプライン」が提供されている**と明記されています[arXiv: HalluVerse25](https://arxiv.org/abs/2503.07833)。
- **特徴**：
  - Hugging Face 上にデータセットが公開されており、`datasets` ライブラリで簡単にロード可能
  - 論文側で Colab ノートブックへのリンクや再現手順が示されている
  - 多言語（英語・中国語・日本語など）かつ細粒度のハルシネーションラベル付き
- **Colab での使い方イメージ**：
  1. `!pip install datasets transformers` などで環境構築
  2. `load_dataset("sabdalja/HalluVerse-M3")` でデータ取得
  3. 論文の Colab ノートブックに従って評価スクリプトを実行

**→ 現時点で「Colab からそのまま評価まで」を最もスムーズに試せるベンチマークのひとつです。**

---

## 2. HaluEval（汎用LLMのハルシネーション検出）

- **Colab 対応**：**GitHub に評価ノートブック（.ipynb）が公開されており、Colab から直接開いて実行可能**です[GitHub: cleanlab-tools](https://github.com/cleanlab/cleanlab-tools/blob/main/benchmarking_hallucination_metrics/benchmark_hallucination_metrics.ipynb)。
- **特徴**：
  - QA・対話・要約など複数タスクを含む大規模ハルシネーション評価データセット
  - Cleanlab の GitHub に、HaluEval サブセットを使ったハルシネーション検出手法の比較ノートブックが公開
- **Colab での使い方イメージ**：
  1. Colab で `https://github.com/cleanlab/cleanlab-tools/blob/main/benchmarking_hallucination_metrics/benchmark_hallucination_metrics.ipynb` を開く
  2. セルを順に実行（依存ライブラリのインストール → データロード → 評価）
  3. 必要に応じてモデルや評価指標をカスタマイズ

**→ 既存の .ipynb をそのまま Colab で動かせるため、実務・研究で最も手軽に試せる選択肢のひとつです。**

---

## 3. MedHallu（医療ハルシネーション検出）

- **Colab 対応**：**データセット自体は MedHELM 経由で非ゲート（公開）として提供されていますが、Colab 専用ノートブックは見当たりません**[MedHELM 公式サイト](https://medhelm.org)。
- **特徴**：
  - PubMedQA 由来の 10,000 QAペア＋意図的にハルシネーションを混ぜた回答
  - MedHELM フレームワーク内で評価する前提の設計
- **Colab での使い方**：
  - MedHELM のインストール手順に従い、Colab 上で環境構築 → 評価スクリプトを実行することは理論上可能ですが、依存関係が多く、**公式の「Colab からワンクリックで評価」という形ではありません**。
  - 自分で `uv` や `medhelm` を Colab にインストールし、`--suite medhallu` 相当のコマンドを実行する必要があります。

**→ データは公開されているが、Colab で「すぐ評価」というより、やや手間がかかるタイプです。**

---

## 4. LibreEval（RAGハルシネーション評価）

- **Colab 対応**：**Hugging Face 上にデータセットとファインチューニング済みモデルが公開されていますが、Colab 専用ノートブックは確認できません**[Arize: LibreEval](https://arize.com/llm-hallucination-dataset)。
- **特徴**：
  - RAG におけるハルシネーション評価用の大規模オープンソースデータセット
  - Qwen2-1.5B-Instruct をファインチューニングしたハルシネーション検出モデルも提供
- **Colab での使い方**：
  - 自分で `datasets` からデータをロードし、評価スクリプトを書く必要があります。
  - 公式サイトには「Colab で使える」と明記されていないため、**Colab 前提のノートブックは自分で作る必要があります。**

---

## 5. Hallucinations Leaderboard（Vectara / Hugging Face）

- **Colab 対応**：**リーダーボード自体は Web UI ですが、評価用データセットとコードは GitHub で公開されており、Colab からクローンして評価可能**です[GitHub: hallucination-leaderboard](https://github.com/vectara/hallucination-leaderboard)。
- **特徴**：
  - GPT-4系, Llama系, Gemini系など多数のLLMを共通ベンチマークで比較
  - 評価スクリプトとデータセットが GitHub で公開
- **Colab での使い方**：
  - `!git clone https://github.com/vectara/hallucination-leaderboard.git` でリポジトリを取得
  - 依存ライブラリをインストールし、評価スクリプトを実行
  - ただし、**公式の「Colab ノートブック」という形ではなく、自分でセルを組む必要があります。**

---

## まとめ：Colab で「すぐ評価」できるもの

- **HalluVerse25**：公式に Colab パイプラインが提供されており、Hugging Face からデータをロードして評価まで行える[arXiv: HalluVerse25](https://arxiv.org/abs/2503.07833)。
- **HaluEval**：Cleanlab の GitHub に評価ノートブック（.ipynb）があり、Colab から直接開いて実行できる[GitHub: cleanlab-tools](https://github.com/cleanlab/cleanlab-tools/blob/main/benchmarking_hallucination_metrics/benchmark_hallucination_metrics.ipynb)。

この2つが、現時点で「Google Colab 上で、データセットのダウンロードから実際の評価までを比較的スムーズに行える」代表的な選択肢です。  
MedHallu や LibreEval もデータは公開されていますが、Colab 向けの公式ノートブックはないため、自分で評価コードを書く必要があります。


