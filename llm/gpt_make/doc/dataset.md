GPTのような大規模言語モデル（LLM）の学習には、**「多様性」「巨大なボリューム」「クオリティ」**の3つが揃ったデータセットが必要です。

現在、多くの研究やLlama 3などの開発で実際に使われている代表的な公開データセットを、用途別に分類して紹介します。

---

## 1. 汎用的な巨大テキストコーパス（事前学習用）

モデルに「言葉の基礎知識」や「世界の常識」を教えるためのデータセットです。

* **Common Crawl**: インターネット上の膨大なウェブページをクロールしたアーカイブ。非常に巨大ですが、ノイズも多いためフィルタリングして使われます。
* **RefinedWeb (Falcon)**: Common Crawlを厳格にクリーニングしたデータセット。
* **The Pile (EleutherAI)**: Wikipedia、arXiv、GitHub、StackExchangeなど、22個の高品質なサブセットを組み合わせた800GBのデータセット。
* **RedPajama**: Llamaの学習データを再現することを目的に作られた、1兆トークンを超えるオープンな巨大コーパス。

---

## 2. 日本語特化のデータセット

日本語のLLMを開発・強化する際に必須となるリソースです。

* **Japanese Wikipedia Dump**: 日本語LLM学習の「基本」です。事実に基づいた正確な文章が学べます。
* **mC4 (Multilingual C4)**: Googleが作成した多言語ウェブコーパス。日本語部分も非常に大きく、Common Crawl由来のデータが含まれます。
* **CC-100 (Japanese)**: 100以上の言語に対応したウェブコーパスの日本語版。
* **Common Voice (Japanese)**: 音声認識用ですが、テキストデータも会話形式の学習に利用されることがあります。

---

## 3. 指示遂行・対話用（ファインチューニング/SFT用）

「質問に答える」「要約する」といった特定のタスクを教えるためのデータセットです。

* **Alpaca Dataset**: GPT-3.5を使って生成された5.2万件の指示データ。現在の「指示学習（Instruction Tuning）」のブームの火付け役です。
* **ShareGPT**: ユーザーがChatGPTと行った会話を共有するサイトから収集されたデータ。自然な対話の流れを学ぶのに適しています。
* **OpenOrca**: 膨大な指示データを、より高度な推論プロセス（Chain-of-Thought）を含めて整理したデータセット。
* **Ichikara-instruction (理研)**: 日本の理化学研究所が公開している、高品質な日本語の指示データセット。

---

## 4. プログラミング・推論用

論理的思考やコード生成能力を高めるために使われます。

* **The Stack**: 許可されたライセンスを持つGitHub上の膨大なソースコード。Llama 3などのコード学習によく使われます。
* **StackExchange**: Q&Aサイトのデータ。問題解決のプロセスを学ぶのに最適です。

---

## 💡 学習データを扱う際の注意点：データパイプライン

公開データセットをそのままモデルに投入することは稀です。実際には以下のプロセス（データパイプライン）が最も重要になります。

1. **クリーニング**: HTMLタグの除去、文字化けの修正。
2. **デデュープリケーション（重複削除）**: 全く同じ文章が何度も出てくると、モデルの性能が低下するため、これを削除します。
3. **有害情報のフィルタリング**: 差別的表現や個人情報（PII）を除去します。
4. **トークナイズ**: 先ほど実装したコードのように、テキストを数値化します。

---
自己回帰モデル（Causal LLM）をゼロから、あるいは追加で学習させる場合、**「多様性」「量」「質」**の3つが揃ったデータセットが必要です。

日本語に特化し、現在の研究や開発で主流となっているデータセットをカテゴリー別に紹介します。

---

### 1. 巨大なWebコーパス（ベースモデルの学習用）

モデルに「日本語の基礎体力」をつけるためのデータです。

| データセット名 | 特徴 | 用途 |
| --- | --- | --- |
| **Common Crawl (CC)** | インターネット上の膨大なアーカイブ。 | 語彙や文法の学習、広範な知識の獲得。 |
| **Japanese mC4** | Googleが公開している巨大な多言語Webコーパスの日本語版。 | 大規模な事前学習（Pre-training）の標準。 |
| **CommonCrawl-Japanese** | 日本語に特化してクリーニングされたWebデータ。 | 日本の文化やトレンドに即した学習。 |

---

### 2. 高品質なテキスト（知能の向上用）

Webのノイズが少ない、論理的で正確な文章です。これらを混ぜることで、モデルの知能が劇的に向上します。

* **Wikipedia (日本語版)**: 事実関係の学習に最適。最もクリーンなデータの一つ。
* **青空文庫**: 小説や文学作品。情緒的な表現や、長文の文脈（コンテキスト）を維持する能力を養います。
* **ニュース記事 (CC-Newsなど)**: 正しい文法と時事知識の学習。

---

### 3. 日本語に特化した独自データ

日本の開発コミュニティや企業が公開している、信頼性の高いデータセットです。

* **Swallow Corpus**: 東京工業大学と産総研が公開した、精緻にクリーニングされた日本語Webコーパス。Llama系の日本語化によく使われます。
* **JGLUE (Japanese General Language Understanding Evaluation)**: 本来は評価用ですが、この形式に沿ったデータで学習させることで、読解力や論理推論力を高められます。

---

### 4. 対話・指示データ（チューニング用）

今回のように「質問に答える」「指示に従う」能力を持たせたい（SFT: Supervised Fine-Tuning）場合に必要です。

* **Ichikura (JIC-7b-Instruct)**: 日本語の指示・応答ペア。
* **Databricks Dolly 15k (日本語訳)**: 人間が作成した高品質な指示データセット。
* **OpenAssistant (日本語部分)**: 多様なユーザーとの対話が含まれます。

---

### 5. 学習用データの「黄金比」

高性能な日本語モデルを作る際の一般的な構成比率（イメージ）は以下の通りです。

1. **Webデータ (80-90%)**: 圧倒的な量で言語の構造を覚える。
2. **書籍・論文・Wiki (5-10%)**: 論理思考と事実の正確性を補強する。
3. **ソースコード (2-5%)**: Pythonなどのコードを混ぜると、不思議と日本語の「論理的推論能力」も上がることが知られています。

---

### 💡 実践的なアドバイス

もし個人や小規模なチームで学習を始めるなら、まずは **Hugging Face** で公開されている以下のリポジトリをチェックすることをお勧めします。

* `CommonCrawl` の日本語部分を抽出したサブセット
* `AozoraBunko` (青空文庫)
* `wikipedia` (language='ja')



```python
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

# 1. 設定
MODEL_NAME = "rinna/japanese-gpt-1b"
DATASET_NAME = "allenai/c4"
SEQ_LEN = 128  # 今回は短めに設定

# 2. トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. Japanese mC4 をストリーミングモードで読み込み
# split="train" で訓練データ、languages=["ja"] で日本語を指定
dataset = load_dataset(DATASET_NAME, "ja", split="train", streaming=True)

def tokenize_function(examples):
    # テキストをトークナイズ
    return tokenizer(examples["text"], truncation=False)

def group_texts(examples):
    # 全てのトークンを連結し、SEQ_LEN ごとに分割する（効率的な学習のため）
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # SEQ_LEN の倍数になるように切り捨て
    total_length = (total_length // SEQ_LEN) * SEQ_LEN
    
    result = {
        k: [t[i : i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)]
        for k, t in concatenated_examples.items()
    }
    # 自己回帰モデルの正解（labels）は input_ids と同じ（内部でずらされる）
    result["labels"] = result["input_ids"].copy()
    return result

# 4. 前処理パイプライン
# map関数で逐次処理を適用
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text", "timestamp", "url"]
)
lm_dataset = tokenized_dataset.map(group_texts, batched=True)

# 5. データの取り出しテスト
print("Fetching processed data...")
for i, batch in enumerate(lm_dataset.take(3)):
    print(f"\n--- Batch {i} ---")
    print(f"Decoded text: {tokenizer.decode(batch['input_ids'])[:100]}...")
    print(f"Shape: {len(batch['input_ids'])} tokens")
```
