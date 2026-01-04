import torch
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. BLIP-2 (Q-Former) のロード ---
print("Loading BLIP-2...")
blip_processor = Blip2Processor.from_pretrained("salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto"
)

# --- 2. LLaVA (Linear/MLP Projector) のロード ---
print("Loading LLaVA-NeXT...")
llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
llava_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", load_in_8bit=True, device_map="auto"
)

# --- 3. 画像とプロンプトの準備 ---
url = "https://web.smartnews.com/img/original/2023/11/17/0Fp7lG.jpg" # 賑やかな街並みや複雑な画像がおすすめ
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
prompt = "Describe this image in detail."

# --- 4. BLIP-2 で推論 ---
print("\n--- BLIP-2 Output ---")
inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
generated_ids = blip_model.generate(**inputs, max_new_tokens=100)
print(blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip())

# --- 5. LLaVA で推論 ---
print("\n--- LLaVA Output ---")
# LLaVA用のプロンプトフォーマット
llava_prompt = f"[INST] <image>\n{prompt} [/INST]"
inputs = llava_processor(llava_prompt, image, return_tensors="pt").to(device)
output = llava_model.generate(**inputs, max_new_tokens=100)
print(llava_processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[-1].strip())



# 1. ビジョンエンコーダとLLMをロード
# (ここでは例としてBLIP-2の内部構造を想定)
vision_encoder = blip_model.vision_model
llm = blip_model.language_model
connector = blip_model.qformer  # または MLP層

# 2. 全てのパラメータを一旦「更新しない」設定にする
for param in vision_encoder.parameters():
    param.requires_grad = False

for param in llm.parameters():
    param.requires_grad = False

# 3. Connector（Q-Formerなど）だけ「更新する」設定にする
for param in connector.parameters():
    param.requires_grad = True

# これで、学習時の optimizer に渡すのは connector.parameters() だけでよくなります
optimizer = torch.optim.AdamW(connector.parameters(), lr=1e-4)
