from trl import PPOConfig

ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=1,
    ppo_epochs=2,
    target_kl=0.1
)

ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=ppo_config
)


def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


for step in range(30):
    batch_prompts = prompts
    responses = []
    rewards = []

    for prompt in batch_prompts:
        response = generate_response(prompt)
        responses.append(response)
        rewards.append(reward_fn(response))

    # PPO step
    ppo_trainer.step(batch_prompts, responses, rewards)

    print(f"\n=== Step {step} ===")
    print(f"Reward mean: {sum(rewards)/len(rewards):.2f}")
    print("Sample output:")
    print(responses[0])