以下では、**Google Colab の無料枠で動かせる小さめのオープンソース LLM**（例：Qwen2-1.5B-Instruct）を Hugging Face からダウンロードし、**ハルシネーション評価データセット（HalluVerse25 など）** を使って評価する手順を説明します。

### 1. 前提と想定

- **モデル**：Qwen2-1.5B-Instruct（約 1.5B パラメータ、Colab 無料枠でも動かしやすいサイズ）
- **データセット**：HalluVerse25（多言語・細粒度ハルシネーション評価ベンチマーク）[arXiv: HalluVerse25](https://arxiv.org/abs/2503.07833)
- **評価タスク**：サンプルごとに「ハルシネーションあり／なし」を判定する二値分類タスク
- **環境**：Google Colab（ランタイムタイプ：GPU / T4 など）

### 2. Colab での環境構築

新しいノートブックで、以下のセルを順に実行します。

__2-1. ライブラリのインストール__

```python
!pip install transformers accelerate datasets torch
```

- `transformers`：Hugging Face のモデル・トークナイザ
- `datasets`：HalluVerse25 などのデータセットロード用
- `torch`：PyTorch（Colab に標準で入っている場合もありますが、念のため）

### 3. モデルのダウンロードと準備

__3-1. モデルとトークナイザのロード__

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # メモリ節約のため
    device_map="auto"
)
```

- `torch_dtype=torch.float16` と `device_map="auto"` により、GPU メモリ使用量を抑えつつ自動で GPU に載せます。
- Colab の無料 GPU（T4）でも 1.5B モデルなら十分動きます。

### 4. 評価データセットのロード

__4-1. HalluVerse25 のロード（例）__

HalluVerse25 は Hugging Face Datasets で公開されています[arXiv: HalluVerse25](https://arxiv.org/abs/2503.07833)。  
ここでは英語サブセットを想定します（実際のデータセット名は論文・Hugging Face ページで確認してください）。

```python
from datasets import load_dataset

# 例：HalluVerse25 の英語 QA サブセット（実際の名前は論文・HF で確認）
dataset = load_dataset("sabdalja/HalluVerse-M3", "en_qa")  # 仮の例
train_data = dataset["train"]
eval_data = dataset["validation"]  # または "test"
```

- データ構造はおおむね `{"question": ..., "answer": ..., "label": ...}` のような形式です。
- `label` が「ハルシネーションあり（1）／なし（0）」を表すとします。

### 5. モデルによるハルシネーション判定の実行

__5-1. プロンプトテンプレートの定義__

モデルに「この回答はハルシネーションを含むか？」を判定させるプロンプトを用意します。

```python
def build_prompt(question, answer):
    prompt = f"""You are a helpful assistant that detects hallucinations in LLM outputs.

Question: {question}
Answer: {answer}

Does the answer contain any hallucination (incorrect or fabricated information) compared to the question?
Please answer only "Yes" or "No"."""
    return prompt
```

__5-2. 推論関数の定義__

```python
def predict_hallucination(question, answer, max_new_tokens=10):
    prompt = build_prompt(question, answer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # プロンプト部分を除去
    response = response[len(prompt):].strip()
    return response
```

- `do_sample=False`（グリーディ生成）で安定させています。
- 出力からプロンプト部分を除去し、判定結果（"Yes"/"No"）だけを取り出します。

__5-3. バッチ評価（サンプル数を絞って実行）__

Colab の無料枠では時間制限があるため、評価サンプル数を絞って実行します。

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 評価用サンプル数を制限（例：100件）
eval_samples = eval_data.select(range(min(100, len(eval_data))))

pred_labels = []
true_labels = []

for example in eval_samples:
    question = example["question"]
    answer = example["answer"]
    true_label = example["label"]  # 0: no hallucination, 1: hallucination

    pred_text = predict_hallucination(question, answer)
    pred_label = 1 if "yes" in pred_text.lower() else 0

    pred_labels.append(pred_label)
    true_labels.append(true_label)

pred_labels = np.array(pred_labels)
true_labels = np.array(true_labels)

acc = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
```

- `true_label` はデータセット側の正解ラベル（0/1）です。
- モデルの出力が "Yes" を含むかどうかで `pred_label` を 1/0 に変換しています（簡易的な実装例です）。

### 6. 評価結果の解釈と注意点

- **Accuracy / F1** が低い場合：
  - モデルサイズが小さい（1.5B）ため、ハルシネーション検出タスクには性能不足の可能性があります。
  - プロンプト設計や判定ロジック（"Yes"/"No" の解釈）を改善する余地があります。
- **時間制限**：
  - Colab 無料枠は 12時間程度でセッションが切れることがあります。  
    長時間かかる場合は、サンプル数をさらに減らすか、有料版（Colab Pro 等）を検討してください。
- **モデル選択**：
  - より大きなモデル（Qwen2-7B など）を使うと精度は上がりますが、Colab 無料 GPU ではメモリ不足になる可能性があります。  
    その場合は `torch_dtype=torch.float16` や `device_map="auto"` を確認し、必要に応じてモデルを小さくするか、LoRA などで軽量化します。

### 7. まとめ

1. Colab で `transformers` と `datasets` をインストール
2. Hugging Face から Qwen2-1.5B-Instruct をダウンロード
3. HalluVerse25 などのハルシネーション評価データセットをロード
4. プロンプトを設計し、モデルに「ハルシネーションあり／なし」を判定させる
5. 正解ラベルと比較して Accuracy / F1 を計算

この流れで、**Google Colab の無料枠でも、オープンソース LLM をダウンロードしてハルシネーション評価を行うことが可能**です。  
より本格的な評価を行う場合は、モデルサイズ・プロンプト設計・評価指標（Precision/Recall など）を調整してください。