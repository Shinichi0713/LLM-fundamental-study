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


# VLM

はい、VLM（Vision-Language Model）向けの**ハルシネーション評価専用ベンチマーク**は複数存在します。  
代表的なものを用途別に整理すると、以下のようになります。

---

## 1. 一般画像キャプション・VQA向けベンチマーク

### CHAIR / CHAIR-MSCOCO
- **タスク**：画像キャプション（MSCOCO 画像＋キャプション）
- **指標**：CHAIR（Caption Hallucination Index）  
  - CHAIR-i：オブジェクトレベルのハルシネーション率  
  - CHAIR-s：センテンスレベルのハルシネーション率
- **特徴**：  
  - 画像中の**存在しないオブジェクトを言い過ぎる**タイプのハルシネーションを評価  
  - REVERSE などの論文でも標準ベンチマークとして利用されている[REVERSE](https://reverse-vlm.github.io)

### POPE（Polling-based Object Probing Evaluation）
- **タスク**：画像中のオブジェクト存在判定（Yes/No 質問）
- **指標**：Accuracy, Precision, Recall
- **特徴**：  
  - 「画像に○○はありますか？」という質問に対して、モデルが**存在しないものを「ある」と答える**頻度を測る  
  - 実装が比較的シンプルで、VLM のハルシネーション傾向を簡単に評価できる[REVERSE](https://reverse-vlm.github.io)

### HaloQuest / mmHal / AMBER / MME-Hall
- **タスク**：多様な視覚質問応答（VQA）・マルチモーダル評価
- **指標**：ハルシネーション率、Accuracy など
- **特徴**：  
  - HaloQuest：多様な質問形式でハルシネーションを評価  
  - mmHal：マルチモーダル・ハルシネーション評価ベンチマーク  
  - AMBER, MME-Hall：マルチモーダル評価セットの一部としてハルシネーション指標を含む[REVERSE](https://reverse-vlm.github.io)

### FREAK
- **タスク**：Fine-grained Hallucination Evaluation Benchmark（細粒度ハルシネーション評価）
- **特徴**：  
  - 既存ベンチマークがタスクを単純化しすぎている問題を指摘し、より**細かい粒度**でハルシネーションを評価するベンチマークを提案[OpenReview](https://openreview.net/forum?id=YeagC09j2K)

---

## 2. 医療画像・専門ドメイン向けベンチマーク

### Gut-VLM（Hallucination-Aware Multimodal Benchmark for GI Image Analysis）
- **タスク**：消化器内視鏡画像（Kvasir-v2）＋テキストレポート
- **特徴**：  
  - ChatGPT で生成した医療レポートに**意図的にハルシネーションを混入**し、それを検出・修正するタスク  
  - VLM を「レポート生成」だけでなく「ハルシネーション検出・修正」にファインチューニングするためのベンチマーク[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13007979)

### Localizing Before Answering（医療 MLLM のグラウンディング評価）
- **タスク**：医療画像 VQA＋グラウンディング（どこを見て答えたか）
- **特徴**：  
  - 回答だけでなく、**根拠となる画像領域**を正しく指せているかを評価  
  - 「根拠のない主張」＝ハルシネーションを検出するのに有効[GitHub](https://github.com/NishilBalar/Awesome-LVLM-Hallucination)

---

## 3. ハルシネーション検出・自己修正手法の評価セット

### REVERSE（Retrospective Verification and Self-correction）
- **タスク**：画像キャプション・VQA など
- **特徴**：  
  - VLM 自身に「生成途中で自分の出力を検証し、ハルシネーションを修正させる」推論パラダイム  
  - CHAIR-MSCOCO, HaloQuest, mmHal, POPE, MME-Hall, AMBER など複数ベンチマークで評価[REVERSE](https://reverse-vlm.github.io)

### DASH（Detection and Assessment of Systematic Hallucinations of VLMs）
- **タスク**：VLM の**系統的ハルシネーション**の検出・評価
- **特徴**：  
  - 単発のミスではなく、モデルが**特定のパターンで繰り返しハルシネーションを起こす**現象を体系的に評価するフレームワーク[CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Augustin_DASH_Detection_and_Assessment_of_Systematic_Hallucinations_of_VLMs_ICCV_2025_paper.pdf)

---

## 4. サーベイ・まとめリソース

### Awesome-LVLM-Hallucination（GitHub）
- VLM ハルシネーション関連の**論文・データセット・コード**をまとめたリポジトリ  
- Localizing Before Answering など、医療 MLLM 向けベンチマークも含む[GitHub](https://github.com/NishilBalar/Awesome-LVLM-Hallucination)

### A Survey of Multimodal Hallucination Evaluation and Detection
- I2T（画像→テキスト）・T2I（テキスト→画像）両方のハルシネーション評価ベンチマークを整理したサーベイ  
- 各ベンチマークの構築プロセス・評価目的・指標を俯瞰できる[arXiv](https://arxiv.org/html/2507.19024v1)

---

## 5. まとめ

- **一般画像キャプション・VQA**：CHAIR-MSCOCO, POPE, HaloQuest, mmHal, AMBER, MME-Hall, FREAK  
- **医療画像・専門ドメイン**：Gut-VLM, Localizing Before Answering  
- **ハルシネーション検出・自己修正**：REVERSE, DASH  
- **サーベイ・まとめ**：Awesome-LVLM-Hallucination, Multimodal Hallucination Survey

これらを使うことで、VLM の**「見えていないものを言い過ぎる」「根拠なく推測する」**といったハルシネーションを、タスク別・指標別に評価できます。  
特に CHAIR-MSCOCO と POPE は実装が比較的簡単で、VLM のハルシネーション傾向を把握する第一歩としてよく使われています。



はい、**Google Colab でも利用可能な VLM ハルシネーション評価ベンチマーク**は複数存在します。  
代表的なものを「Hugging Face Datasets でそのまま使えるか」「追加の画像ダウンロードが必要か」という観点で整理します。

---

## 1. Hugging Face Datasets でそのまま使えるもの（Colab 推奨）

### POPE（Polling-based Object Probing Evaluation）
- **用途**：画像中のオブジェクト存在判定（Yes/No 質問）によるハルシネーション評価
- **データ形式**：`id`, `question_id`, `question`, `answer`, `image_source`, `image`, `category`  
  - `image` カラムに画像データが直接含まれている（COCO 画像も HF に格納済み）[Hugging Face](https://huggingface.co/datasets/lmms-lab/POPE)
- **Colab でのロード例**：
  ```python
  from datasets import load_dataset
  dataset = load_dataset("lmms-lab/POPE")
  ```
- **特徴**：
  - COCO 画像を別途ダウンロードする必要がなく、`datasets` だけで完結
  - 実装がシンプルで、VLM の「存在しないものをあると言う」頻度を簡単に評価できる

### HaloQuest（Visual Hallucination Dataset）
- **用途**：マルチモーダル推論における視覚ハルシネーション評価
- **データ形式**：`image_name`, `url`, `image type`, `hallucination type`, `question`, `groundtruth responses`, `split` など  
  - 画像は `url` カラムに S3 リンクとして格納（外部ホスト）[Hugging Face](https://huggingface.co/datasets/johko/HaloQuest)
- **Colab でのロード例**：
  ```python
  dataset = load_dataset("johko/HaloQuest")
  ```
- **特徴**：
  - 画像は URL 経由で取得する必要があるが、`datasets` でメタデータはすぐにロード可能
  - 「false premises」「insufficient context」などハルシネーションタイプが細かくラベル付けされている

### MMHal-Bench
- **用途**：LMM（Large Multimodal Model）のハルシネーション評価ベンチマーク
- **データ形式**：OpenImages 由来の画像＋質問＋正解（ground-truth）  
  - Hugging Face 上ではカスタムロードスクリプト形式（`loading script` が必要）[Hugging Face](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench)
- **Colab でのロード例**：
  ```python
  dataset = load_dataset("Shengcao1006/MMHal-Bench")
  ```
- **特徴**：
  - 画像は OpenImages 由来で、HF 上にまとめて格納されている
  - 評価には GPT-4 による自動評価スクリプト（`eval_gpt4.py`）が提供されている（API コストがかかる点に注意）

---

## 2. GitHub からダウンロードが必要なもの（Colab でも利用可能）

### AMBER（An LLM-free Multi-dimensional Benchmark）
- **用途**：生成タスク・識別タスク両方のハルシネーションを多角的に評価
- **データ形式**：`data/` ディレクトリに JSON 形式で質問・画像情報・正解が格納  
  - 画像は別途ダウンロードが必要な場合がある（リポジトリの README で案内）[GitHub](https://github.com/junyangwang0410/AMBER)
- **Colab での利用例**：
  ```bash
  !git clone https://github.com/junyangwang0410/AMBER.git
  ```
- **特徴**：
  - LLM を使わない自動評価パイプライン（LLM-free）を提供
  - 存在・属性・関係といった細粒度のハルシネーションを評価

### CHAIR / CHAIR-MSCOCO
- **用途**：MSCOCO 画像キャプションのオブジェクトハルシネーション評価
- **データ形式**：MSCOCO 2014 画像＋キャプション＋オブジェクトアノテーション  
  - MSCOCO 画像は別途ダウンロードが必要（`cocodataset.org`）[GitHub](https://github.com/yejipark-m/ConVis)
- **Colab での利用例**：
  - MSCOCO 画像をダウンロードし、CHAIR のアノテーション（GitHub で配布）と組み合わせて評価
- **特徴**：
  - CHAIR-i（インスタンスレベル）、CHAIR-s（センテンスレベル）のハルシネーション率を算出
  - 画像ダウンロードに時間がかかるため、Colab ではサブセット評価が現実的

---

## 3. Colab で使う際の実践的な選択肢

**「すぐに試したい」「画像ダウンロードを最小限にしたい」場合は以下の順がおすすめです。**

1. **POPE**（`lmms-lab/POPE`）  
   - 画像が HF に内蔵されているため、`load_dataset` だけで完結  
   - Yes/No 質問でハルシネーション率を簡単に評価できる

2. **HaloQuest**（`johko/HaloQuest`）  
   - 画像は URL 経由だが、メタデータはすぐにロード可能  
   - ハルシネーションタイプが細かく分類されており、分析に適している

3. **MMHal-Bench**（`Shengcao1006/MMHal-Bench`）  
   - OpenImages 画像が HF にまとまっている  
   - GPT-4 評価スクリプト付き（API コストがかかる点に注意）

**「画像ダウンロードも含めて本格的に評価したい」場合は以下も選択肢になります。**

- **AMBER**（GitHub からクローン）  
- **CHAIR-MSCOCO**（MSCOCO 画像＋CHAIR アノテーション）

---

## 4. まとめ

- **Colab で最も手軽に使える**：POPE（`lmms-lab/POPE`）  
- **ハルシネーションタイプを細かく分析したい**：HaloQuest（`johko/HaloQuest`）  
- **LMM のハルシネーションを GPT-4 で自動評価したい**：MMHal-Bench（`Shengcao1006/MMHal-Bench`）  
- **LLM-free で多角的に評価したい**：AMBER（GitHub からクローン）

これらを組み合わせることで、Google Colab の無料枠でも VLM のハルシネーション性能を多面的に評価できます。