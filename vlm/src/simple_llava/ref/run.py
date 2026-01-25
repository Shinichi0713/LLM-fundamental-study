import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests

# 1. モデルとプロセッサのロード
model_id = "llava-hf/llava-1.5-7b-hf"
# 4-bit量子化でロードすると、無料版Colab(T4 GPU)でも動きます
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto",
    load_in_4bit=True
)
processor = AutoProcessor.from_pretrained(model_id)

# 2. 画像とプロンプトの準備
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "USER: <image>\nWhat is shown in this image? ASSISTANT:"

# 3. 推論の実行
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)

# 4. 結果のデコード
print(processor.decode(output[0], skip_special_tokens=True))