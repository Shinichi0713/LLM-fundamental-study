import torch
from model import MambaLikeLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル初期化（学習前のランダム重み）
vocab_size = 50257  # GPT-2 トークナイザの vocab_size
d_model = 512
n_layers = 6

model = MambaLikeLM(
    vocab_size=vocab_size,
    tokenizer=None,  # 内部で GPT-2 トークナイザを自動ロード
    d_model=d_model,
    n_layers=n_layers,
).to(device)

# 3件のプロンプトを用意
prompts = [
    "The history of",
    "In mathematics,",
    "Artificial intelligence is",
]

print("=== 学習前の生成テスト（ランダム初期化） ===")
for i, prompt in enumerate(prompts, 1):
    print(f"\n--- Test {i}: '{prompt}' ---")
    generated = model.generate(
        prompt=prompt,
        max_len=30,      # 追加トークン数
        temperature=0.7,
    )
    print(generated)