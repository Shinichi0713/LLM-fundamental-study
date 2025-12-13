from unsloth import FastLanguageModel
from datasets import load_dataset
import torch

# LLaMA 3.1 8B (約80億) よりも小さい Gemma 2B (約20億) に変更
model_name = "unsloth/gemma-2b-bnb-4bit"

# torch.bfloat16 (BFLOAT16) を使用できるか確認
# BFLOAT16はFP16よりも安定しており、メモリ効率が良い (T4 GPUで利用可能)
# 確保できない場合は torch.float16 にフォールバック
dtype = torch.bfloat16
if torch.cuda.is_available():
    if not torch.cuda.is_bf16_supported():
        dtype = torch.float16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=512,
    dtype=dtype,             # 最適なデータ型を選択
    load_in_4bit=True,       # QLoRAを強制
    # SFTTrainerで使用する場合、use_gradient_checkpointing='unsloth' を後で設定することで、
    # さらにメモリを節約できますが、ロード時点では不要です。
)

# データセットの読み込み
data = load_dataset("yahma/alpaca-cleaned")
print("Dataset loaded successfully.")

# ① まず、全データのうち 10% だけを使用（90% を破棄）
small_data = data["train"].train_test_split(test_size=0.9, seed=42)["train"]

# ② small_data を Train+Val と Test に分割（例：Test に 10% を割り当てる）
split1 = small_data.train_test_split(test_size=0.1, seed=42)
train_val_data = split1["train"]  # 全体の 90%（small_data のうち）
test_data = split1["test"]        # 全体の 10%（small_data のうち）

# ③ train_val_data をさらに Train と Validation に分割（例：Validation に 10% を割り当てる）
split2 = train_val_data.train_test_split(test_size=0.1, seed=42)
train_data = split2["train"]  # train_val_data の 90%（全体で約 81%）
eval_data = split2["test"]     # train_val_data の 10%（全体で約 9%）

# ④ テンプレートを適用する関数の定義
def format_examples(example):
    instr = example["instruction"]
    inp = example["input"] if example["input"] else ""
    # テンプレート形式の文字列を作成
    example["text"] = (
        f"### Instruction:\n{instr}\n"
        f"### Input:\n{inp}\n"
        f"### Response:\n{example['output']}"
    )
    return example

# ⑤ 各データセットにフォーマットを適用
train_data = train_data.map(format_examples)
eval_data = eval_data.map(format_examples)
test_data = test_data.map(format_examples)

# ⑥ 変換後の例を確認（最初の 200 文字を表示）
print(train_data[0]["text"][:200])

# トークナイズを行う関数を定義
def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512)

# バッチ処理で高速化しつつトークナイズを適用
train_tokens = train_data.map(tokenize_batch, batched=True, remove_columns=["instruction","input","output","text"])
eval_tokens = eval_data.map(tokenize_batch, batched=True, remove_columns=["instruction","input","output","text"])

# トークナイズ後のデータ例を確認
print(train_tokens[0].keys())
# 出力例: dict_keys(['input_ids', 'attention_mask'])

from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 1. モデルにLoRAアダプタを挿入して学習モードにする
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none"
)
# LoRA適用後、学習可能パラメータ数を確認（任意）
model.print_trainable_parameters()

# 2. データコラレータ（バッチ処理時のデータ整形）を用意
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 3. 学習時の各種ハイパーパラメータを設定
training_args = TrainingArguments(
    output_dir="./llama-unsloth-model",
    per_device_train_batch_size=2,
    num_train_epochs=1, # 必要に応じて増やしてください
    learning_rate=2e-4,
    fp16=True,                         # 16ビット精度で計算（A100等のGPUではbf16推奨）
    gradient_checkpointing=True,       # 勾配チェックポイントでメモリ節約
    logging_steps=50,
    save_steps=200,
    # evaluation_strategy="epoch",       # 1エポックごとに評価
    save_total_limit=1,
)
# 4. Trainerオブジェクトの初期化（モデル・データ・設定の紐付け）
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_tokens,
    eval_dataset=eval_tokens,
    data_collator=data_collator,
    args=training_args
)

# 5. 学習の実行
trainer.train()

# 推論用のプロンプトを準備
instruction = "Explain the importance of sleep in simple terms."
input_text = ""  # 追加の入力がない場合は空文字
prompt = f"### Instruction:\\n{instruction}\\n### Input:\\n{input_text}\\n### Response:\\n"

# トークナイズしてモデルに入力
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# モデルによるテキスト生成
output_ids = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
# トークン列を文字列にデコード
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)




