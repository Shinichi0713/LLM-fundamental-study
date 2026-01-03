import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 1. デバイスの設定とモデルのロード
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "salesforce/blip2-opt-2.7b"

# メモリ節約のため float16 を使用
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

# 2. 画像の取得（ここでは公式のサンプル画像を使用）
url = "http://images.cocodataset.org/val2017/000000039769.jpg" # 2匹の猫が寝ている画像
raw_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# 3. 前処理
inputs = processor(images=raw_image, return_tensors="pt").to(device, torch.float16)

# 4. Q-Formerの出力を取得 (output_attentions=True がポイント)
with torch.no_grad():
    outputs = model.get_text_features(
        **inputs, 
        return_dict=True, 
        output_attentions=True
    )

# Q-Formerの最終層のクロスアテンションを取得
# 形状: [batch, num_heads, num_queries, num_patches]
# BLIP-2のViTパッチ数は通常 257 (16x16 + CLSトークン)
cross_attentions = outputs.qformer_outputs.cross_attentions[-1]

# 5. 可視化 (最初のいくつかのクエリがどこを見ているか)
num_queries_to_show = 4
fig, axes = plt.subplots(1, num_queries_to_show + 1, figsize=(20, 5))

# 元画像の表示
axes[0].imshow(raw_image)
axes[0].set_title("Original Image")
axes[0].axis("off")

# 各クエリのアテンションマップを表示
# ヘッド平均を取って簡易化
avg_attn = cross_attentions[0].mean(dim=0).cpu().float().numpy() 

for i in range(num_queries_to_show):
    # CLSトークンを除いた 16x16 のパッチにリシェイプ
    # BLIP-2のパッチ配置に合わせてリサイズ
    mask = avg_attn[i, 1:].reshape(16, 16)
    
    # 元画像と同じサイズにリサイズ（補完あり）
    axes[i+1].imshow(raw_image)
    axes[i+1].imshow(mask, cmap='jet', alpha=0.6, extent=(0, raw_image.size[0], raw_image.size[1], 0))
    axes[i+1].set_title(f"Query {i+1} Attention")
    axes[i+1].axis("off")

plt.tight_layout()
plt.show()