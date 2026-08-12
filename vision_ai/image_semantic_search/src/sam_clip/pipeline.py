import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 1. デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. SAMモデルのロード
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

# 3. CLIPモデルのロード
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# 4. 対象画像の準備（サンプル画像のダウンロードまたはお手持ちの画像を指定）
!wget -q -O input_image.jpg "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg"

image_path = "input_image.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h, w, _ = image_rgb.shape

# 5. セグメンテーション対象の候補クラス（テキストプロンプト）の定義
class_names = ["background", "dog", "car", "tree", "grass", "person", "building"]
text_prompts = [f"a photo of a {c}" for c in class_names]

# CLIPテキスト特徴量の算出
text_tokens = clip.tokenize(text_prompts).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 6. SAMによる領域マスク生成
print("Generating SAM masks...")
masks = mask_generator.generate(image_rgb)
print(f"Total masks generated: {len(masks)}")

# 7. 各マスク領域のクロップ & CLIPによるセマンティック分類
semantic_map = np.zeros((h, w), dtype=int)
confidence_map = np.zeros((h, w), dtype=float)

crops = []
valid_masks = []

for mask_data in masks:
    m = mask_data['segmentation']
    bbox = mask_data['bbox']  # [x, y, w, h]
    x, y, bw, bh = [int(v) for v in bbox]

    # クロップ用領域の切り出し（余白を持たせる）
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    
    crop_img = image_rgb[y1:y2, x1:x2]
    if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
        continue
    
    pil_crop = Image.fromarray(crop_img)
    crop_tensor = clip_preprocess(pil_crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_feature = clip_model.encode_image(crop_tensor)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        
        # コサイン類似度計算
        similarity = (100.0 * image_feature @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        
        predicted_class_idx = indices[0].item()
        confidence = values[0].item()

    # スコアに基づきセマンティックマップへ書き込み
    # より高い確信度の領域で更新
    mask_indices = m.astype(bool)
    update_pixels = mask_indices & (confidence > confidence_map)
    semantic_map[update_pixels] = predicted_class_idx
    confidence_map[update_pixels] = confidence

# 8. 結果の可視化
cmap = plt.get_cmap("tab10")
color_mask = np.zeros((h, w, 3), dtype=np.uint8)

for idx in range(len(class_names)):
    color = (np.array(cmap(idx)[:3]) * 255).astype(np.uint8)
    color_mask[semantic_map == idx] = color

# 原画像とのアルファブレンド
overlay = cv2.addWeighted(image_rgb, 0.5, color_mask, 0.5, 0)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("SAM-CLIP Semantic Segmentation Result")
plt.imshow(overlay)
plt.axis("off")
plt.show()

print("Class Legend:")
for i, name in enumerate(class_names):
    print(f"- Class {i}: {name}")