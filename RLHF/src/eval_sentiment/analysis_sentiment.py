import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

# 1. 基本設定
device = 0 if torch.cuda.is_available() else "cpu"
model_name = "distilgpt2" # Colabで高速に動くサイズ

# 2. PPOの設定
config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    batch_size=32,
    mini_batch_size=8, # VRAM節約のため小さめに設定
    optimize_cuda_cache=True
)

# 3. モデルとトークナイザーの準備
# PPO用に「価値頭(Value Head)」を追加したモデルをロード
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(device)
ref_model = create_reference_model(model) # 更新前のモデルと比較用
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# 4. 報酬モデル (Sentiment Analysis) の準備
# 生成された文章がどれくらいポジティブかを判定する「審判」
reward_model = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

# 5. ダミーデータの準備 (プロンプト)
def tokenize(sample):
    sample["input_ids"] = tokenizer.encode("The movie was", add_special_tokens=False)
    sample["query"] = "The movie was"
    return sample

# 実験用に32個の同じクエリを用意
from datasets import Dataset

# 5. データセットの準備 (リストではなく Dataset オブジェクトにする)
raw_data = {
    "query": ["The movie was"] * 256,
    "input_ids": [tokenizer.encode("The movie was")] * 256
}
dataset = Dataset.from_dict(raw_data)
dataset.set_format("torch") # PyTorchテンソルとして扱えるように設定

# 6. PPO Trainerの初期化 (これでエラーが消えるはずです)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset)

# 6. PPO Trainerの初期化
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset)

# 7. 学習ループ (体験用のため10ステップ)
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 10,
}

print("Starting PPO Training...")
print("Starting PPO Training...")

for epoch in tqdm(range(50)): # 50エポックに増やして変化を見やすくします
    all_rewards = []
    
    for batch in ppo_trainer.dataloader:
        # クエリのテンソル化
        query_tensors = [ids.to(device) for ids in batch["input_ids"]]

        # A. Responseの生成
        # ppo_trainer.generate はリスト[torch.Tensor]を返します
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        
        # デコードしてテキスト化
        queries = [tokenizer.decode(q) for q in query_tensors]
        responses = [tokenizer.decode(r) for r in response_tensors]

        # B. 報酬(Reward)の計算
        # query と response を結合して評価
        texts = [q + r for q, r in zip(queries, responses)]
        pipe_outputs = reward_model(texts)
        
        # スコアの計算（5.0倍のスケーリングで学習を加速）
        step_rewards = [
            torch.tensor((out["score"] if out["label"] == "POSITIVE" else -out["score"]) * 5.0).to(device) 
            for out in pipe_outputs
        ]
        
        all_rewards.extend([r.item() for r in step_rewards])

        # C. PPOによる更新
        # ここで query, response, reward を渡してモデルをアップデート
        stats = ppo_trainer.step(query_tensors, response_tensors, step_rewards)
        
    # エポックごとの進捗表示
    avg_reward = sum(all_rewards) / len(all_rewards)
    print(f" Epoch {epoch} | Average Reward: {avg_reward:.4f}")

print("Training Complete.")