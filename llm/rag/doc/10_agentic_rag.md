

## エージェンティックRAGとは

**Agentic RAG（エージェンティックRAG）** は、**AIエージェントの考え方をRAG（Retrieval-Augmented Generation）に組み込んだ、自律的で動的な検索・生成パイプライン**です。

### 1. Agentic RAGとは何か

- **定義**  
  Agentic RAGは、LLMが**自律的に次のステップを計画しながら外部ソースから情報を取得する**AIパラダイムです[Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners/blob/main/05-agentic-rag/README.md)。
- **特徴**
  - **自律的な計画・推論**：LLM自身が「何を検索すべきか」「どのツールを使うか」を判断する。
  - **反復的な検索・生成ループ**：1回の検索で終わらず、結果を評価し、クエリを修正し、必要に応じて再検索する。
  - **メモリと状態管理**：過去の行動や検索結果を保持し、次のステップに活かす。
  - **自己修正（Self-Correction）**：不適切なクエリや無関係な結果を検出し、再クエリや別ツールの利用で修正する[Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners/blob/main/05-agentic-rag/README.md)。

IBMはこれを「**AIエージェントを使ってRAGを促進する**」アプローチと説明し、従来のRAGよりも**柔軟性・適応性・精度・スケーラビリティ**が高いとしています[IBM Think](https://www.ibm.com/think/topics/agentic-rag)。

### 2. 従来のRAG（Traditional RAG）との違い

NVIDIAのブログでは、従来RAGとAgentic RAGを以下のように対比しています[NVIDIA Developer Blog](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter)。

| 項目 | 従来のRAG（Traditional RAG） | Agentic RAG |
|------|-----------------------------|-------------|
| **動作の性質** | 静的な「ルックアップ」 | 動的な「推論プロセス」 |
| **ワークフロー** | 単純な「クエリ → 検索 → 生成」の直線パス | エージェントが自律的に計画・反復・修正するループ |
| **データソース** | 単一のベクトルDBや知識ベースに接続することが多い | 複数の外部知識ベースやツールを動的に使い分ける[IBM Think](https://www.ibm.com/think/topics/agentic-rag) |
| **適したタスク** | コスト重視・高速な単純問い合わせ | 複雑な調査・要約・コード修正など、非同期で複雑なタスク[NVIDIA Developer Blog](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter) |
| **精度・信頼性** | プロンプト設計と検索結果に依存（人間が品質を判断） | エージェントが結果を検証し、必要に応じて再検索・修正する |

IBMも、従来RAGが「**反応的でルールベース**」なのに対し、Agentic RAGは「**能動的で知的な問題解決**」を行うと説明しています[IBM Think](https://www.ibm.com/think/topics/agentic-rag)。

### 3. Agentic RAGの典型的なワークフロー（NVIDIAの整理）

NVIDIAはAgentic RAGのワークフローを以下のようにまとめています[NVIDIA Developer Blog](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter)。

1. **データの必要性を特定**（エージェントが「何が足りないか」を判断）
2. **クエリ生成**（エージェントが適切な検索クエリを自ら作成）
3. **動的な知識取得**（クエリエンジンや複数ソースから情報を取得）
4. **プロンプトへのコンテキスト追加**（取得した情報をLLMに渡す）
5. **LLMによる意思決定・生成**（必要なら再検索・修正を繰り返す）

この「**エージェントが能動的に検索戦略を変えながら繰り返す**」点が、従来RAGとの大きな違いです。

### 4. Agentic RAGのメリットとトレードオフ

__メリット__
- **精度向上**：反復的な検索・検証により、ハルシネーションを減らし、より正確な回答が可能[NVIDIA Developer Blog](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter)。
- **複雑な問い合わせへの対応**：複数の知識ベースやツールを跨いだ調査ができる[IBM Think](https://www.ibm.com/think/topics/agentic-rag)。
- **動的な知識への対応**：リアルタイムで変化するデータソースにも適応可能。

Agentic RAGは、 事実性データセット（factuality datasets）において、標準RAGに対して[最大34%の精度向上を達成](https://www.marktechpost.com/2026/06/08/google-research-adds-agentic-rag-to-gemini-enterprise-agent-platform-with-a-sufficient-context-agent-for-multi-hop-queries)。
と評価されています。


![1781125100256](image/10_agentic_rag/1781125100256.png)

__精度向上の理由__

Google Researchのブログでは、Agentic RAGが精度を向上させる理由として、以下の点を挙げています[Google Research Blog](https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag)。

1. **マルチエージェントによる計画・再検索**  
   - クエリを分解し、複数のエージェントが協調して検索・再検索を行う。
   - 情報が複数の「データの島」に分散している場合でも、**十分なコンテキストが見つかるまで検索を続ける**。

2. **Sufficient Context Agent（十分性エージェント）**  
   - 取得した情報が「質問に答えるのに十分か」を評価する専用エージェントを導入。
   - 不足があれば、**追加の検索クエリを生成して再検索**する。

3. **「推測」や「情報不足」を減らす**  
   - 従来RAGでは「最初の検索で情報が見つからない → 推測 or 『情報不足』と回答」になりがち。
   - Agentic RAGでは、**情報があるのに見つけられていないケースを掘り下げる**ことで、事実に基づいた回答率を高める。

このような設計により、**複雑なエンタープライズ問い合わせにおいて、事実性の高い回答を生成できる確率が大幅に向上した**と報告されています[MarkTechPost](https://www.marktechpost.com/2026/06/08/google-research-adds-agentic-rag-to-gemini-enterprise-agent-platform-with-a-sufficient-context-agent-for-multi-hop-queries)。

__トレードオフ__
- **コスト・レイテンシ**：反復的なLLM呼び出しとツール利用により、トークンコストと時間が増加[IBM Think](https://www.ibm.com/think/topics/agentic-rag)。
- **設計の複雑さ**：エージェントの状態管理・エラー処理・マルチエージェント協調など、設計・実装がより高度になる。

## どんなところに使われている？

Google AI（特にGemini Enterprise Agent Platform）では、Agentic RAGの考え方を取り入れたフレームワークが実際に使われています。


### 1. Google ResearchによるAgentic RAGフレームワークの紹介

Google Researchの公式ブログでは、**Gemini Enterprise Agent Platform向けの「Agentic RAGフレームワーク」** を導入したと明記されています[Google Research Blog](https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag)。

- このフレームワークは、**マルチエージェントのワークフロー**で構成され、
- 複雑なエンタープライズ問い合わせを**分解し、反復的に検索して十分なコンテキストを集めてから回答を生成**する仕組みです。
- 従来のRAGでは「情報が足りない」とすぐに諦めてしまうようなケースでも、**エージェントが再検索・再計画を行うことで、より信頼性の高い回答を目指す**設計になっています。

このブログでは、Agentic RAGを「**単一の検索エンジンではなく、組織化された研究部門のようなマルチエージェントRAG**」と表現しています[Google Research Blog](https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag)。

### 2. Gemini Enterprise Agent PlatformのRAG Engineとの関係

Google Cloudの公式ドキュメントでは、**Gemini Enterprise Agent Platformに「RAG Engine」というコンポーネント**が用意されており、LLMのコンテキストをプライベートデータで拡張するためのフレームワークとして位置づけられています[Google Cloud Docs](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/rag-overview)。

- RAG Engineは、**ベクトルDBや外部データソースから情報を取得し、LLMのプロンプトにコンテキストを追加する**役割を持ちます。
- Google ResearchのAgentic RAGフレームワークは、このRAG Engineを**エージェント的に制御するレイヤー**として機能し、
  - クエリの分解
  - 複数ソースへのルーティング
  - 検索結果の評価と再検索
  といった「Agenticな挙動」を実現しています。

つまり、**Gemini Enterprise Agent Platform上で構築されるエージェントは、RAG EngineをAgentic RAG的に利用できる**ようになっている、と言えます。

### 3. 具体的に何が「Agentic」なのか

Googleの説明によると、Agentic RAGフレームワークでは以下のような挙動が特徴的です[Google Research Blog](https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag)。

1. **クエリの分解と計画**  
   複雑な問い合わせを、複数のサブクエリに分解し、どのデータソースをどの順番で検索するかを計画する。
2. **反復的な検索**  
   一度の検索で十分な情報が得られない場合、**エージェントが自らクエリを修正し、再検索**する。
3. **十分性のチェック**  
   取得した情報が「質問に答えるのに十分か」を評価し、不足があれば追加検索を行う。
4. **マルチエージェント協調**  
   クエリ分解エージェント、検索エージェント、評価エージェントなどが連携し、**組織的な研究チームのように動く**。

これらは、前回説明した「Agentic RAG」の定義（自律的な計画・反復・自己修正）と完全に一致しています。





