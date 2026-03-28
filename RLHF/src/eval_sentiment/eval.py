def evaluate_model(ppo_model, ref_model, tokenizer, reward_model, prompts):
    ppo_model.eval()
    results = []

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 15,
    }

    print(f"{'Prompt':<20} | {'Model Type':<15} | {'Response':<30} | {'Score':<5}")
    print("-" * 80)

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # 1. 学習後モデル (PPO) の生成
        with torch.no_grad():
            ppo_output = ppo_model.generate(input_ids, **generation_kwargs)
            ppo_text = tokenizer.decode(ppo_output[0], skip_special_tokens=True)
            ppo_score = reward_model(ppo_text)[0]
            ppo_final_score = ppo_score["score"] if ppo_score["label"] == "POSITIVE" else -ppo_score["score"]

        # 2. 学習前モデル (Reference) の生成
        with torch.no_grad():
            ref_output = ref_model.generate(input_ids, **generation_kwargs)
            ref_text = tokenizer.decode(ref_output[0], skip_special_tokens=True)
            ref_score = reward_model(ref_text)[0]
            ref_final_score = ref_score["score"] if ref_score["label"] == "POSITIVE" else -ref_score["score"]

        results.append({
            "prompt": prompt,
            "ppo_text": ppo_text,
            "ref_text": ref_text,
            "ppo_score": ppo_final_score,
            "ref_score": ref_final_score
        })

        print(f"{prompt:<20} | {'PPO (After)':<15} | {ppo_text[len(prompt):].strip():<30} | {ppo_final_score:.4f}")
        print(f"{'':<20} | {'Ref (Before)':<15} | {ref_text[len(prompt):].strip():<30} | {ref_final_score:.4f}")
        print("-" * 80)

    return results

# 評価の実行
test_prompts = ["The movie was", "I thought the film", "This story is"]
eval_results = evaluate_model(model, ref_model, tokenizer, reward_model, test_prompts)