**RAG・GraphRAGにおけるEntity Resolution（名寄せ・エンティティ解決）に関する研究・報告は複数存在します**。

以下、代表的なものを整理します。

---

## 1. RAG向けの動的エンティティ解決

### DynamicER: Resolving Emerging Mentions to Dynamic Entities for RAG
- **会議**: EMNLP 2024
- **著者**: Jinyoung Kim ら（Seoul National University）
- **内容**: 言語が急速に進化する中で、新しい言及（mentions）を動的なエンティティに解決する手法を提案。RAGにおいて、新しく現れた固有名詞や略称を既存エンティティに正しくリンクする問題に対処しています。<source-chip title="ACL Anthology" url="https://aclanthology.org/anthology-files/pdf/emnlp/2024.emnlp-main.762.pdf" />

気になる：〇

## 2. 知識グラフのノイズ除去（重複エンティティを含む）

### Less is More: Denoising Knowledge Graphs For Retrieval Augmented Generation
- **発表**: arXiv 2025
- **著者**: Yilun Zheng ら（Nanyang Technological University など）
- **内容**: RAG用の知識グラフからノイズを除去する研究で、**冗長なエンティティや誤った関係の除去**を含む。KGの品質を上げることでRAGの検索・回答精度を改善することを目指しています。<source-chip title="arXiv" url="https://arxiv.org/abs/2510.14271" />

---

## 3. マルチエージェントRAGによるEntity Resolution

### Multi-Agent RAG Framework for Entity Resolution
- **発表**: MDPI Computers 2025
- **内容**: 単一LLMではなく、**複数の専門エージェントを協調させるマルチエージェントRAGフレームワーク**でEntity Resolutionを行う手法。異なるエージェントが候補生成・類似度計算・最終判定などを分担することで、単一LLMより高精度な名寄せを実現しています。<source-chip title="MDPI" url="https://www.mdpi.com/2073-431X/14/12/525" />

---

## 4. LLM + グラフ精緻化によるコスト効率の良いEntity Resolution

### Adaptive Graph Refinement and Label Propagation with LLMs for Cost-Effective Entity Resolution
- **発表**: arXiv 2026
- **著者**: Hongtao Wang ら
- **内容**: LLMを使った**適応的グラフ精緻化とラベル伝播**による、コスト効率の良いEntity Resolution手法。すべてのペアにLLMを適用するのではなく、グラフ構造を活用して効率的にエンティティを統合するアプローチです。<source-chip title="arXiv" url="https://arxiv.org/pdf/2605.25814" />

---

## 5. 知識グラフ上のEnd-to-End Entity ResolutionとQA

### End-to-End Entity Resolution and Question Answering Using Differentiable Knowledge Graphs
- **会議**: EMNLP 2021
- **著者**: Armin Oliya ら（Amazon Alexa AI）
- **内容**: **微分可能な知識グラフ**を用いて、Entity Resolutionと質問応答をEnd-to-Endで学習する手法。Amazon Alexa AIの研究で、エンティティの曖昧性解消とQAを統合的に扱っています。<source-chip title="ACL Anthology" url="https://aclanthology.org/2021.emnlp-main.345.pdf" />

気になる：〇

## 6. 時間的・因果的一貫性を考慮したEntity-Event KG

### Respecting Temporal-Causal Consistency: Entity-Event Knowledge Graph for Retrieval-Augmented Generation
- **会議**: EACL 2026
- **著者**: Ze Yu Zhang ら（NUS / Alibaba Group）
- **内容**: 時間的・因果的な一貫性を保つEntity-Event知識グラフをRAGに活用する研究。エンティティとイベントの関係を正しく構造化し、検索時の整合性を高めることを目指しています。<source-chip title="ACL Anthology" url="https://aclanthology.org/2026.eacl-long.90.pdf" />

---

## 7. GraphRAGにおける情報ける情報損失の軽減（トリプル文脈復元）

### How to Mitigate Information Loss in Knowledge Graphs for GraphRAG
- **会議**: IJCAI 2025
- **著者**: Manzong Huang ら（Hefei University of Technology）
- **内容**: GraphRAGにおける知識グラフの情報損失問題に対し、**トリプル文脈復元とクエリ駆動フィードバック**で対処する手法。エンティティの文脈情報を失わずに保持することで、名寄せや検索精度の向上に寄与します。<source-chip title="IJCAI" url="https://www.ijcai.org/proceedings/2025/0901.pdf" />

---

## 8. マルチソース知識グラフ補完のための増分Entity Resolution

### Incremental Multi-source Entity Resolution for Knowledge Graph Completion
- **発表**: ESWC 2020 / PMC
- **著者**: Eric Peukert, Erhard Rahm ら
- **内容**: **複数ソースからの知識グラフ構築時**に、増分的にEntity Resolutionを行う手法。異なるデータソースから来た同じエンティティを統合し、KGを補完するための古典的・実践的なアプローチです。<source-chip title="PMC" url="https://pmc.ncbi.nlm.nih.gov/articles/PMC7250616/" />

---

## 9. 科学的知識グラフ構築のためのRAGフレームワーク

### Graphusion: A RAG Framework for Scientific Knowledge Graph Construction with a Global Perspective
- **発表**: arXiv 2024
- **著者**: Rui Yang ら
- **内容**: 科学的知識グラフをグローバルな視点で構築するRAGフレームワーク。エンティティの重複検出や統合を含む、大規模なKG構築パイプラインを提案しています。<source-chip title="arXiv" url="https://arxiv.org/html/2410.17600" />

---

## まとめ：研究の傾向

Entity Resolutionに関するRAG/GraphRAG研究は、主に次の方向性に分かれます。

| 方向性 | 代表研究 |
|---|---|
| **動的・新規言及の解決** | DynamicER (EMNLP 2024) |
| **KGノイズ除去・重複除去** | Less is More (arXiv 2025) |
| **マルチエージェントによるER** | Multi-Agent RAG Framework (MDPI 2025) |
| **LLM + グラフ精緻化** | Adaptive Graph Refinement (arXiv 2026) |
| **End-to-End ER + QA** | Amazon Alexa AI (EMNLP 2021) |
| **時間的・因果的一貫性** | Entity-Event KG (EACL 2026) |
| **情報損失軽減** | Triple Context Restoration (IJCAI 2025) |
| **マルチソースKG統合** | Incremental ER (ESWC 2020) |

---

## 実務での参考ポイント

特に参考になるのは、

- **DynamicER**: 新しい略称・社内用語が増えていく環境での名寄せ
- **Less is More**: 構築済みKGの重複エンティティ除去
- **Multi-Agent RAG Framework**: 複数の判定基準（文字列・文脈・グラフ構造）を組み合わせた名寄せ
- **Adaptive Graph Refinement**: LLMコストを抑えつつグラフ構造で名寄せ精度を上げる方法

これらは、社内RAGで「表記ゆれした固有名詞を自動統合したい」という要件に直接活かせる知見が多いです。