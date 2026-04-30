import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. モデルとトークナイザーのロード
# mamba2-130m は軽量で、動作確認に最適です
model_id = "state-spaces/mamba2-130m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    dtype=torch.bfloat16,
    offload_folder="offload" # 追加：ディスク退避用のフォルダ
)

# 2. プロンプトの準備
prompt = "The future of state space models is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# 3. テキスト生成
# MambaはKVキャッシュの代わりに独自のステートを持つため、生成も高速です
output = model.generate(
    input_ids, 
    max_new_tokens=50, 
    do_sample=True, 
    top_k=50, 
    top_p=0.95,
    temperature=0.7
)

# 4. 結果のデコード
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"--- Generated Text ---\n{generated_text}")