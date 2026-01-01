import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from datasets import load_dataset
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. モデル・トークナイザ
# =========================
MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # PPO必須

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# =========================
# 2. PPO 設定
# =========================
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=4,
    mini_batch_size=2,
    ppo_epochs=2,
    target_kl=0.1,
)

ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=ppo_config,
)

# =========================
# 3. データ（超簡易）
# =========================
prompts = [
    "Explain reinforcement learning simply.",
    "What is a cat?",
    "Tell me a short joke.",
    "Why is the sky blue?",
]

# =========================
# 4. Reward Model（擬似）
# =========================
def reward_fn(texts):
    """
    擬似報酬:
    ・短すぎると低評価
    ・適度な長さを好む
    """
    rewards = []
    for t in texts:
        length = len(t.split())
        reward = min(length / 20.0, 1.0)  # 最大1.0
        rewards.append(reward)
    return torch.tensor(rewards).to(device)

# =========================
# 5. PPO 学習ループ
# =========================
for step in range(5):
    batch_prompts = random.sample(prompts, k=ppo_config.batch_size)

    query_tensors = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True
    ).input_ids.to(device)

    # --- 生成 ---
    response_tensors = ppo_trainer.generate(
        query_tensors,
        max_new_tokens=40,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    responses = tokenizer.batch_decode(
        response_tensors[:, query_tensors.shape[1]:],
        skip_special_tokens=True
    )

    # --- 報酬計算 ---
    rewards = reward_fn(responses)

    # --- PPO step ---
    stats = ppo_trainer.step(
        query_tensors,
        response_tensors,
        rewards
    )

    print(f"\n=== Step {step} ===")
    for p, r, rew in zip(batch_prompts, responses, rewards):
        print(f"Prompt: {p}")
        print(f"Response: {r.strip()}")
        print(f"Reward: {rew.item():.2f}")
