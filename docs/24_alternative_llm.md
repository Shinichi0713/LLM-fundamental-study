
当ブログではTransformerに代わるアーキテクチャとして[Mamba](https://yoshishinnze.hatenablog.com/entry/2026/01/25/182406)、[Mamba-3](https://yoshishinnze.hatenablog.com/entry/2026/04/27/023000)について説明したことがあります。
実際にTransformerが最適解ではないのでは？という動きは続いています。

本日は現在確認出来る限り、その他どんなアーキテクチャが考えられているかについて調査・説明します。

## Transformerに代わるアーキテクチャ

2026年現在、Transformerに代わるアーキテクチャとして最も注目を浴びているのは、主に以下の3つの潮流です。


### 1. State Space Models（SSM）: Mamba シリーズ

特に **Mamba**（およびその後継の **Mamba 2**、**Mamba 3**）が、Transformer代替の中心的存在として挙げられています。<source-chip title="Algorithmine" url="https://algorithmine.com/learn/mamba-rwkv-vs-transformers-2026" />

Mambaの最大の特徴は、Transformerの「自己注意機構」が持つ**二次的（二乗オーダー）の計算コスト**を、**線形オーダー**に抑えることです。Transformerでは系列長が2倍になると計算量が約4倍になりますが、Mambaでは系列長に比例するだけで済みます。また、推論時のメモリ使用量が文脈長に依存せず一定であるため、100万トークンを超える極めて長い文脈でも効率的に動作します。

2026年3月に公開された **Mamba 3** では、複素数値による状態追跡やマルチ入力マルチ出力（MIMO）の定式化などが導入され、ゲノム解析や音声処理、長時間系列予測などで高い性能を示しています。<source-chip title="arXiv" url="https://arxiv.org/html/2603.15569" />

一方で、Mambaは「文書内の電話番号を探す」ような**精密なトークン単位の検索タスク**や、MMLUなどの学習ベンチマークでは、同規模のTransformerにやや劣る傾向があると指摘されています。<source-chip title="Algorithmine" url="https://algorithmine.com/learn/mamba-rwkv-vs-transformers-2026" />

### 2. Linear Attention: RWKV

**RWKV**（Receptance Weighted Key Value）は、注意機構そのものを「線形化」することで、二次的な計算コストを回避するアーキテクチャです。<source-chip title="Algorithmine" url="https://algorithmine.com/learn/mamba-rwkv-vs-transformers-2026" />

RWKVは、学習時にはTransformerのように並列計算できつつ、推論時にはRNNのように「定数メモリ・線形スケーリング」で動作するのが特徴です。2026年初頭には **RWKV 7 G1** がリリースされ、特にエッジデバイスやCPU推論といった、メモリや計算資源が限られた環境での運用に適しています。<source-chip title="Presenc AI" url="https://presenc.ai/research/hybrid-attention-models-mamba-jamba-rwkv-2026" />


### 3. ハイブリッドアーキテクチャ（Transformer + Mamba）

2026年の実運用環境では、「Transformerを完全に置き換える」というよりも、**両者の長所を組み合わせたハイブリッドモデル**が最も現実的な選択肢として広がっています。<source-chip title="Presenc AI" url="https://presenc.ai/research/hybrid-attention-models-mamba-jamba-rwkv-2026" />

代表的な例として、AI21 Labsの **Jamba 1.5 Large**（Transformer + Mamba + MoEのハイブリッド、総パラメータ約398B/アクティブ94B）があります。このモデルは25.6万トークンの文脈窓を持ち、長文脈処理においてコストとレイテンシの面で純粋なTransformerを大きく上回る一方、短い文脈での品質も維持しています。

その他にも、IBMの **Bamba 9B**、**Zamba 2 7B**、コード生成向けの **Codestral Mamba** など、多くのハイブリッドモデルがリリースされています。


### 補足: その他の新興アーキテクチャ

- **Striped Hyena 2**: 畳み込みベース＋注意機構のハイブリッド。100万トークン以上の文脈に対応。
- **Liquid LFM 2**: MIT CSAILが開発した「Liquid Neural Network」ベース。サブ線形のメモリ消費が特徴。
- **Falcon Mamba 7B**: 純粋なSSMベースの汎用モデル。

## 性能比較

2025年から2026年にかけて、Transformerとその代替アーキテクチャ（Mamba、RWKV、ハイブリッドモデルなど）を体系的に比較した複数の重要な研究が発表されています。以下に主要なものをまとめます。

### 1. NVIDIAによる大規模実証研究（2024年）

**「An Empirical Study of Mamba-based Language Models」** <source-chip title="arXiv" url="https://arxiv.org/html/2406.07887v1" />

NVIDIA、プリンストン大学、カーネギーメロン大学などの研究者による、おそらく現時点で最も包括的な実証研究の一つです。同じデータセット、同じハイパーパラメータで、**8Bパラメータ規模**のモデルを訓練し、直接比較を行っています。

**比較したアーキテクチャ:**
- Mamba（純粋SSM）
- Mamba-2（純粋SSM）
- Mamba-2-Hybrid（43% Mamba-2 + 7% Self-Attention + 50% MLP）
- Transformer（LLaMA系）

**主要な結果:**
- **純粋なMamba/Mamba-2**は多くの言語モデリングタスクでTransformerに匹敵するか、それを上回る性能を示しました。
- しかし、**5-shot MMLU**や**Phonebook Lookup**（電話帳検索）など、文脈から情報を正確にコピー・検索するタスクでは、Transformerに約15ポイント遅れを取るなど、顕著な差が見られました。
- **Mamba-2-Hybrid**（ハイブリッド）は、評価した12の標準タスク**すべて**でTransformerを上回り、平均して **+2.65ポイント** の改善を達成。MMLUでも3.5ポイント高い精度を記録しました。
- 長文脈タスク（16K・32K・128K）でも、ハイブリッドモデルはTransformerと同等かそれ以上の性能を示しました。
- 推論速度では、ハイブリッドモデルがTransformerの**最大8倍高速**になると予測されています。

### 2. NeurIPS 2025採択論文：基盤能力の比較

**「How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?」**（arXiv: 2505.18522）<source-chip title="Paper Notes" url="https://en.papernotes.org/NeurIPS2025/llm_pretraining/how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train/" />

**110M〜1.3Bパラメータ規模**で、Transformer、Mamba-1/2、RWKV-6/7を比較した研究です。従来の「混合ドメイン事前学習＋同ドメイン評価」ではアーキテクチャ間の差が見えにくいことを指摘し、**「ドメイン制限事前学習＋分布外（OOD）テスト」** という新しい評価枠組みを提案しています。

**主要な結果:**

| アーキテクチャ | 混合ドメイン評価 | OOD評価 | 備考 |
|---|---|---|---|
| Transformer++ | 最良 | 最良 | 基準 |
| Mamba-1/2 | Transformerと同等 | **顕著な劣化** | 混合ドメインでは差が見えない |
| RWKV-6/7 | Transformerと同等 | **中程度の劣化** | Mambaよりはマシ |
| Top-1 Selection（新提案） | やや劣る | Transformerと同等 | 検証用のミニマリスト設計 |

**重要な知見:**
- 混合ドメインでの事前学習では、MambaもTransformerも同じように見えるが、**分布外のデータではMamba/RWKVの基盤能力が劣化**することが明らかになりました。
- この劣化の根本原因は、これらのアーキテクチャが「系列全体にわたる任意の選択能力（arbitrary selection over the full sequence）」を失っていることだと特定されています。
- Mambaの特徴的なコンポーネント（データ依存減衰、畳み込み）は、**収束速度には寄与するが、基盤能力には寄与しない**という驚くべき結果も出ています。


### 3. 実用的な長文脈ベンチマーク（2026年）

**「Mamba vs RWKV: 32K Context Benchmark on A100」**<source-chip title="TildAlice" url="https://tildalice.io/mamba-vs-rwkv-long-context-benchmark-32k-tokens/" />

実際の32Kトークン文脈で、Mamba 2.8B・RWKV 7B・Llama-2 7Bを比較した実験です。

**要約タスク:**
- Mamba: **85%** が一貫した要約を生成
- RWKV: **45%** に低下（12Kトークンを超えると劣化）
- Llama-2: 4Kの文脈窓制限で機能せず

**Needle-in-Haystack（大量の文脈から特定事実を検索）:**

| 位置 | Mamba正解 | RWKV正解 |
|---|---|---|
| 10Kトークン | 19/20 | 18/20 |
| 20Kトークン | 14/20 | 16/20 |
| 30Kトークン | 8/20 | 11/20 |

RWKVが20K・30K位置でMambaを上回る結果となりました。RWKVの指数減衰が、古い情報を「完全に忘れる」のではなく「薄く保持する」効果を持っているためと分析されています。

**メモリ使用量:**
- Mamba: **18GB**（32K文脈）
- Llama-2（32Kに外挿）: **約45GB**（KVキャッシュのため）

### 4. 計算効率と表現力の系統的ベンチマーク（2026年）

**「Benchmarking the Computational and Representational Efficiency of State Space Models against Transformers on Long-Context Dyadic Sessions」**（arXiv: 2601.01237）<source-chip title="arXiv" url="https://www.arxiv.org/pdf/2601.01237" />

心理療法の会話記録（6,000〜10,000トークン）を対象に、MambaとLLaMA Transformerを**50Mパラメータ規模**で同条件比較した研究です。

**計算効率の観点:**
- 512〜8,192トークンでのメモリ使用量と推論速度を測定し、SSMがTransformerを追い越す「臨界点」を特定。
- Mambaは理論通り線形スケーリングを示し、長文脈でのメモリ優位性を実証。

**表現力の観点:**
- Transformerは「早期オーバースムージング→後期回復」のパターンを示すのに対し、Mambaは「早期トークンの独自性を保持→後期均質化」という異なる動態を示すことが判明。
- これにより、両アーキテクチャが異なる「失敗モード」を持つことが説明されています。

### 5. その他の比較研究

| 研究名 | 主な比較対象 | 特徴 |
|---|---|---|
| **RankMamba**（arXiv: 2403.18276） | Mamba vs Transformer（文書ランキング） | 情報検索タスクでのMambaの有効性を評価 |
| **TransMamba**（arXiv: 2503.24067） | Transformer-Mambaハイブリッド | 系列レベルでのハイブリッド化の提案 |
| **SAMBA**（ICLR 2025） | Mamba + Attentionハイブリッド | 無限文脈向けのシンプルなハイブリッド設計 |
| **RWKV-X**（arXiv: 2504.21463） | RWKVのハイブリッド拡張 | 線形複雑性を維持しつつTransformer品質に近づける |

### 比較研究から見えてくる全体像

これらの研究を総合すると、以下のような傾向が明確になっています。

1. **純粋なMamba/RWKV**は計算効率・メモリ効率で圧倒的な優位性を持つ一方、**精密な情報検索・文脈コピー・分布外一般化**ではTransformerに劣る場面が見られる。

2. **ハイブリッドモデル**（Mamba + 少量のSelf-Attention、例：Jamba、Mamba-2-Hybrid）は、両者の長所を補い合い、多くのベンチマークで**純粋Transformerを上回る**結果を示している。

3. **評価方法が結果を大きく左右する**。従来の混合ドメインベンチマークでは差が見えにくく、OOD評価や長文脈評価で初めてアーキテクチャ間の本質的な差が浮き彫りになる。

## 実用されている場面

現時点（2026年8月）で、Transformer以外のアーキテクチャが実際に製品・サービスとして提供されている事例を、用途別に整理してご説明します。

### 1. 企業向け長文脈処理：Jamba シリーズ（AI21 Labs）

**アーキテクチャ**: Mamba + Transformer + MoE のハイブリッド

**実用化の形態**:
- **Google Cloud Vertex AI** と **Amazon Bedrock** で正式に提供中<source-chip title="Google Cloud Blog" url="https://cloud.google.com/blog/products/ai-machine-learning/jamba-1-5-model-family-from-ai21-labs-is-now-available-on-vertex-ai" /><source-chip title="AWS News Blog" url="https://aws.amazon.com/blogs/aws/jamba-1-5-family-of-models-by-ai21-labs-is-now-available-in-amazon-bedrock/" />
- AI21 Studio（同社のAPIプラットフォーム）でも利用可能

**主な用途**:
- **長文書の要約・分析**: 25.6万トークンの文脈窓を活かし、契約書、法律文書、研究論文、財務報告書などの長文書を一度に処理
- **企業内RAG（検索拡張生成）**: 大量の社内文書を文脈に直接投入して質問応答
- **マルチドキュメントQA**: 複数の資料を横断して回答を生成

**採用理由**: 同等サイズのTransformerモデルより低レイテンシー・低コストで、長文脈タスクの品質を維持

### 2. コード生成・補完：Codestral Mamba（Mistral AI）

**アーキテクチャ**: Mamba-2（純粋SSM）

**実用化の形態**:
- Mistral AIが2024年7月にリリース。Hugging Faceで公開
- NVIDIAとも連携し、最適化された実装を提供<source-chip title="NVIDIA Technical Blog" url="https://developer.nvidia.com/blog/revolutionizing-code-completion-with-codestral-mamba-the-next-gen-coding-llm/" />
- Capgeminiなどの企業が自社のソフトウェア開発支援AIに組み込み<source-chip title="Mistral AI" url="https://mistral.ai/customers/capgemini/" />

**主な用途**:
- **IDE統合コード補完**: Fill-in-the-Middle（FIM）技術で、既存コードの途中に適切なコードを挿入
- **長いコードベースの理解**: 数万トークンに及ぶコードファイル全体の文脈を保持
- **ローカル開発環境**: KVキャッシュ不要なため、開発者のPC上で軽量に動作

**採用理由**: コードは通常非常に長いシーケンスであり、Mambaの線形スケーリングが有利

### 3. 汎用ローカルLLM・エッジデバイス：Falcon Mamba 7B（TII）

**アーキテクチャ**: 純粋Mamba（SSM）

**実用化の形態**:
- アブダビのTechnology Innovation Institute（TII）が開発・公開
- Hugging Faceでモデル、量子化版（4bit、GGUF）、Instruct版が提供<source-chip title="Hugging Face" url="https://huggingface.co/tiiuae/falcon-mamba-7b" />
- Edgework.aiなどのエッジ推論プラットフォームで「Production-ready」として配布<source-chip title="Hugging Face" url="https://huggingface.co/forkjoin-ai/falcon-mamba-7b-safetensors" />

**主な用途**:
- **消費者向けGPUでの長文脈推論**: 8GB VRAMのGPU（RTX 4060等）で32Kトークン以上の文脈を処理
- **オフライン文書分析**: 機密文書をクラウドに送信せず、ローカルで長文書を要約・検索
- **主権AI（Sovereign AI）**: 特定国・組織内で完結するAIシステム

**採用理由**: 消費者級ハードウェアで長文脈が扱える点が、クラウド依存を避けたい組織に appealing

### 4. モバイル・オンデバイスAI：RWKV エコシステム

**アーキテクチャ**: RWKV（線形注意・RNN的）

**実用化の形態**:
- **RWKV-Runner**: OpenAI API互換のローカルサーバー（6,400+ Stars）<source-chip title="GitHub" url="https://github.com/josStorer/RWKV-Runner" />
- **RWKV App**: Android/iOS/Windows/macOS/Linux対応のプライバシー重視チャットアプリ<source-chip title="GitHub" url="https://github.com/RWKV-APP/RWKV_APP" />
- **AI00 RWKV Server**: Rust製の軽量推論サーバー
- **RWKV-7_localGPT**: ローカル文書チャットボット

**主な用途**:
- **スマートフォン上のプライベートAI**: データを端末内に留めるチャットボット
- **エッジデバイスでのリアルタイム対話**: CPUのみの環境でも動作
- **ストリーミング処理**: 音声認識・翻訳のリアルタイム処理（逐次処理に強い）
- **IoT機器への組み込み**: メモリ制限の厳しい環境

**採用理由**: 定数メモリ・線形スケーリングにより、スマートフォンや組み込み機器での実用が現実的

### 5. 超低消費電力エッジAI：BrainChip（Akidaプロセッサ）

**アーキテクチャ**: 状態空間モデル（SSM）＋ ニューロモーフィックハードウェア

**実用化の形態**:
- BrainChip社のAkidaプロセッサにSSMを実装<source-chip title="Edge AI Vision" url="https://www.edge-ai-vision.com/wp-content/uploads/2025/06/E2W07_Lewis_BrainChip_2025.pdf" />
- IPライセンス形態で半導体メーカーに提供

**主な用途**:
- **ウェアラブルデバイス**: 常時稼働の音声認識・異常検知
- **産業用センサー**: 長期間の時系列データ監視
- **自動車**: 車載システムでのリアルタイム処理

**採用理由**: Transformerの二次的計算コストをニューロモーフィックハードウェアでさらに削減

### 6. 研究・実験的実用：Zamba / Striped Hyena / Liquid

| モデル名 | 開発元 | 実用化の状況 | 主な用途 |
|---|---|---|---|
| **Zamba 2** | Zyphra | Hugging Faceで公開。ハイブリッド（Mamba+Attention） | 汎用対話、長文脈実験 |
| **Striped Hyena 2** | Together AI | 長文脈（100万トークン+）対応モデルとして提供 | ゲノム解析、長時間音声 |
| **Liquid LFM 2** | MIT CSAIL + Liquid AI | サブ線形メモリ消費を実現 | 大規模文脈の効率処理 |
| **SAMBA** | Microsoft Research | ICLR 2025採択。無限文脈を目指すハイブリッド | 研究・実験的利用 |

## 用途別の採用傾向

```
【長文書の企業分析（契約書・論文・報告書）】
→ Jamba 1.5 Large（AI21 Labs）※クラウドAPI

【開発者向けコード補完】
→ Codestral Mamba（Mistral AI）※IDEプラグイン

【機密文書のローカル分析】
→ Falcon Mamba 7B（TII）※消費者GPUで32K+文脈

【スマートフォン・プライバシー重視チャット】
→ RWKV-7 + RWKV App ※完全オンデバイス

【常時稼働のIoT・ウェアラブル】
→ BrainChip Akida + SSM ※超低消費電力

【研究・実験的長文脈（100万トークン+）】
→ Striped Hyena 2 / SAMBA
```

## 総括

Transformerの代替アーキテクチャに関する議論の本質は、以下の3点に集約されます。

**1. 解決しようとしている問題は「長文脈のコスト」**
Transformerは系列長に対して計算量が二乗オーダーで増え、KVキャッシュでメモリも線形に膨張します。Mamba（SSM）やRWKVは、これを「線形オーダー・定数メモリ」に抑えることで、100万トークン級の長文脈を現実的なコストで処理しようとしています。

**2. 純粋な代替アーキテクチャには「能力のトレードオフ」がある**
NVIDIAの実証研究やNeurIPS 2025採択論文が示す通り、MambaやRWKVは言語モデリングや長文脈処理で高い効率を示す一方、「文脈からの精密な情報検索・コピー・分布外（OOD）一般化」では同規模のTransformerに劣る傾向があります。つまり、速さと省メモリを得た代償として、全系列にわたる任意の情報への細かいアクセス能力が一部失われています。

**3. 現時点の現実的な答えは「ハイブリッド」**
「Transformerを完全に捨てる」のではなく、Mambaの線形効率とTransformerの自己注意機構の精度を組み合わせたハイブリッド（Jamba、Mamba-2-Hybridなど）が、現時点で最もバランスの取れた解です。実用化もこの形が主流で、企業の長文書処理にはJamba、コード補完にはCodestral Mamba、エッジ・モバイルにはRWKVやFalcon Mambaといった具合に、用途に応じた「部分的最適化」が進んでいます。

**総括すれば、Transformer代替の議論は「Transformerを倒す」物語ではなく、「Transformerの得意な部分は残しつつ、長文脈・エッジ・低コスト領域では異なるメカニズムを併用する」という、実用的なアーキテクチャの多様化の物語です。**



