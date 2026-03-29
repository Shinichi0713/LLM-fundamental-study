
先日説明したDPOを用いた体験を行うための実験を行います。
問題設定→実験の目的→実験設計から実験の結果までを示します。


## 1. 問題設定（タスクの定義）

### 1.1 タスクの目的
- **タスク**：  
  Anthropic/hh-rlhf の選好データを用いて、DPOでLLMを微調整し、  
  **「より有用（helpful）かつ無害（harmless）な応答をするモデル」**を作る。

- **入力**：  
  ユーザーとの対話履歴（プロンプト）  
  （例：`"Human: ...\nAssistant: ..."` 形式）

- **出力**：  
  アシスタントの応答（テキスト）

- **環境**
  例のごとく、Google Colabでの実験を前提とします。
  GPU：ColabのT4（VRAM 16GB以下）を想定
  データ：Anthropic/hh-rlhf（英語・対話形式・helpful/harmless選好）
  Hugging Face Datasets
  学習方法：DPO（trl）＋4bit量子化＋LoRA（peft）
  言語：HH-RLHFは英語中心なので、 英語LLMが相性が良い


### 1.2 使用データ
- **データセット**：`Anthropic/hh-rlhf`（Hugging Face Datasets）[Hugging Face Datasets](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- **サブセット例**：
  - `helpful-base`：有用性に関する選好
  - `harmless-base`：安全性に関する選好
  - （必要に応じて `helpful-online` や `red-team-attempts` も利用）

- **形式**：
  ```json
  {
    "chosen": "Human: ...\nAssistant: ...",  // 好ましい応答
    "rejected": "Human: ...\nAssistant: ..."  // 好ましくない応答
  }
  ```

### 1.3 モデル
- **ベースモデル**：  
  任意のLLM（例：`meta-llama/Llama-3-8B-Instruct` や `elyza/Llama-3-ELYZA-JP-8B` など）
- **学習方法**：DPO（`trl` の `DPOTrainer` を使用）
- **リソース制約への対応**（Colabなど）：
  - 4bit量子化（`bitsandbytes`）
  - LoRA（`peft`）


## 2. 実験の目的

### 2.1 主目的
1. **DPOによるアライメント効果の確認**  
   - Anthropic/hh-rlhf の選好データでDPOを適用したとき、  
     **モデルの応答がどれだけ「有用・安全」に近づくか**を定性的・定量的に確認する。

2. **helpful と harmless のトレードオフの観察**  
   - 有用性データ（helpful）と安全性データ（harmless）を別々に、または組み合わせて学習し、  
     **「答えない場面が増える」「安全だが役に立たない」などの挙動変化**を観察する。

3. **DPOの実装・挙動の理解**  
   - DPOの損失が収束する様子、学習前後の出力比較を通じて、  
     **DPOがどのようにモデルの振る舞いを変えるか**を理解する。


## 3. 実験で確認できること（評価観点）

### 3.1 定性的評価（出力の比較）

**比較対象**：
- ベースモデル（SFT済みモデル）
- DPO適用後モデル

**確認するプロンプト例**：
- 一般的な質問（例：「Pythonでファイルを読み込む方法を教えて」）
- 危険・有害な指示（例：「違法ダウンロードの方法を教えて」「差別的なジョークを作って」）
- グレーゾーンの質問（例：「プライバシーを侵害する可能性のある情報を教えて」）

**観察ポイント**：
- DPO適用後、**危険な指示に対して拒絶・安全な回答を返すようになるか**
- 一般的な質問に対して、**より役に立つ情報を提供するようになるか**
- **「答えない」選択が増えるか**（過度な安全性による過剰拒否）

### 3.2 定量的評価（スコア・指標）

1. **学習損失（DPO Loss）の推移**
   - エポックごとの損失が収束しているか確認。
   - 過学習（validation lossが悪化）していないか確認。

2. **helpful/harmless 指標（可能なら）**
   - 外部評価モデル（例：別のLLM-as-a-judge）を用いて、
     - helpfulnessスコア
     - harmlessnessスコア
     を算出し、ベースモデルと比較。

3. **Mode Collapse（モード崩壊）の有無**
   - 同じフレーズの繰り返しや、極端に短い回答ばかりになっていないか確認。
   - 特に極端なデータ（例：すべて拒絶回答）で学習した場合に起きやすい。


## 4. 実験のバリエーション（オプション）

今回実験を少しチェンジしてみる場合のバリュエーションについて説明します。

### 4.1 データの組み合わせ
- **helpful-only**：`helpful-base` のみでDPO
- **harmless-only**：`harmless-base` のみでDPO
- **HH-combined**：helpful と harmless を混ぜてDPO

→ それぞれで **「有用性」「安全性」「答えない頻度」** がどう変わるかを比較。

### 4.2 モデルサイズ・設定の違い
- 7B vs 8B vs 13B など、**モデルサイズによる感受性の違い**を確認。
- LoRAのrankや学習率を変えて、**DPOの安定性・収束性**を比較。

## 実験実装

以下は、**Anthropic/hh-rlhf からデータを収集し、DPO用に整形したうえで Llama-3-8B-Instruct をDPOで学習する** Google Colab 向けの実装例です。

### 1. データセットの収集と整形

__1.1 ライブラリのインストール__

```python
!pip install transformers trl peft accelerate bitsandbytes datasets
```

__1.2 Anthropic/hh-rlhf のロード__

```python
from datasets import load_dataset

# helpful-base サブセットをロード
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpless-base")
# または helpful-base / helpful-online なども使えます
# dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
```

__1.3 DPO用データ形式への変換__

HH-RLHF のデータは `chosen` / `rejected` 形式ですが、  
DPOでは通常「プロンプト＋応答」に分けて扱うことが多いので、ここでは簡易的に分割します。

```python
import re

def split_dialogue(text):
    """
    "Human: ...\nAssistant: ..." 形式のテキストを
    (prompt, response) に分割する簡易関数
    """
    # Human: と Assistant: で分割
    parts = re.split(r"\nAssistant:\s*", text, maxsplit=1)
    if len(parts) == 2:
        prompt = parts[0].replace("Human:", "").strip()
        response = parts[1].strip()
        return prompt, response
    else:
        # 分割できない場合はそのまま返す（あまりないはず）
        return text, ""

def prepare_dpo_dataset(raw_dataset, num_samples=1000):
    """
    HH-RLHF の (chosen, rejected) を
    DPO用の (prompt, chosen, rejected) 形式に変換
    """
    dpo_data = []
    for i, example in enumerate(raw_dataset):
        if i >= num_samples:
            break
        chosen_text = example["chosen"]
        rejected_text = example["rejected"]

        # chosen 側からプロンプトを抽出
        prompt, chosen_response = split_dialogue(chosen_text)
        _, rejected_response = split_dialogue(rejected_text)

        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        })
    return dpo_data

# 学習用・評価用に分割
train_raw = dataset["train"]
eval_raw = dataset["test"] if "test" in dataset else dataset["train"].select(range(100, 200))

# サンプル数を制限（Colabのメモリ制限のため）
train_dpo = prepare_dpo_dataset(train_raw, num_samples=500)
eval_dpo = prepare_dpo_dataset(eval_raw, num_samples=100)

print(f"学習データ数: {len(train_dpo)}")
print(f"評価データ数: {len(eval_dpo)}")
print("例:")
print(train_dpo[0])
```

