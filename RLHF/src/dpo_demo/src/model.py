from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "stabilityai/stablelm-zephyr-3b"

# トークナイザ
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # pad_token を eos_token に設定

# config をロードして pad_token_id を設定
config = AutoConfig.from_pretrained(model_name)
config.pad_token_id = tokenizer.pad_token_id

# CPU用4bit量子化設定
cpu_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 参照モデル（ref_model）も4bit量子化してCPUに載せる
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=cpu_bnb_config,
    device_map={"": "cpu"},
    dtype=torch.float16,
    config=config,  # 修正したconfigを渡す
)

# GPU用4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 学習対象モデル（model）はGPUに載せる
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    config=config,  # 修正したconfigを渡す
)


from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()