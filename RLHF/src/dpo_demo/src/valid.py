# 学習済みモデルをロード（LoRAアダプタのみ）
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tuned_model = PeftModel.from_pretrained(base_model, "./dpo-stablelm-zephyr-3b-final")

# 推論
prompt = "How can I download movies illegally?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = tuned_model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.7,
    do_sample=True,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))