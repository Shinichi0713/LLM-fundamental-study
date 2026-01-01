import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

# 1. 設定（Config）
config = PPOConfig(
    model_name="gpt2", # 今回は軽量なGPT-2を例にします
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
)

# 2. モデルとトークナイザーの準備
# RLHFでは「価値関数（Value Head）」を持つ特別なモデル構造を使います
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = create_reference_model(model) # 更新前のモデル（KL制約用）
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. PPOトレーナーの初期化
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

# 4. ダミーのプロンプトデータ
query_txt = "Once upon a time,"
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# 5. 強化学習ループ（1ステップの例）
# --- A: 生成 (Rollout) ---
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
response_tensor = ppo_trainer.generate(query_tensor.squeeze(), **generation_kwargs)
response_txt = tokenizer.decode(response_tensor.squeeze())

# --- B: 報酬の付与 (Evaluation) ---
# 本来はここで学習済みの「報酬モデル」に文章を通します。
# 今回は例として「文章が長ければ報酬が高い」というダミー報酬を設定します。
reward = [torch.tensor(len(response_txt) / 10.0)] 

# --- C: 学習 (Optimization) ---
# クエリ、回答、報酬のセットを渡してPPOで更新
stats = ppo_trainer.step([query_tensor.squeeze()], [response_tensor.squeeze()], reward)

print(f"Query: {query_txt}")
print(f"Response: {response_txt}")
print(f"Reward: {reward[0].item()}")
print(f"PPO Step Complete. Stats: {stats['ppo/loss/total']}")

