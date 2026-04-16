import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

model_name = "mistralai/Mistral-7B-v0.1"  # 例：7Bモデル

# 4bit量子化設定（Colab向け）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 学習前モデル（ref_model）と学習対象モデル（model）を同じ設定でロード
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRAを適用（任意）
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
ref_model = get_peft_model(ref_model, lora_config)  # ref_modelも同じ設定にする場合

# ここで DPOTrainer(loss_type="ipo") で学習…
def compare_model_outputs(model, ref_model, tokenizer, prompt, max_new_tokens=128):
    """
    同じプロンプトに対して、学習前モデルと学習後モデルの出力を比較する。
    """
    # トークナイズ
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 学習前モデル（ref_model）の生成
    with torch.no_grad():
        ref_outputs = ref_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        ref_text = tokenizer.decode(ref_outputs[0], skip_special_tokens=True)

    # 学習後モデル（model）の生成
    with torch.no_grad():
        model_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        model_text = tokenizer.decode(model_outputs[0], skip_special_tokens=True)

    # 結果を表示
    print("=" * 60)
    print("【プロンプト】")
    print(prompt)
    print("\n【学習前モデル（ref_model）の出力】")
    print(ref_text)
    print("\n【学習後モデル（model）の出力】")
    print(model_text)
    print("=" * 60)

    return ref_text, model_text


# 比較用プロンプト（IPOの説明をさせてみる）
prompt = "Explain Identity Preference Optimization (IPO) in AI alignment."

ref_text, model_text = compare_model_outputs(model, ref_model, tokenizer, prompt)