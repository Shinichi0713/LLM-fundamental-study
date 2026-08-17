"""
HQ-SAM 2 vs SAM 2.1 輪郭精度比較 PoC（プロンプトベース / Largeモデル）
Google Colab 用スクリプト

推奨実行手順:
1. Colab メニュー → ランタイム → セッションを再起動
2. 以下のセルを上から順に実行

特徴:
- SAM 2.1 Large vs HQ-SAM 2 Large を比較
- プロンプト（点またはbbox）で同じ物体を固定して比較
- ベースラインSAM 2.1のパッチ（hq_token_only / interm_embeddings を無視）
- 逐次ロードでVRAM節約
- 輪郭点数・長さ・差分を定量的に比較
"""

# ============================================================
# セル0: GPUメモリ完全クリア & 環境設定
# ============================================================
import os
import gc
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 既存変数を削除
for var_name in ["model_sam2", "model_hq", "predictor_sam2", "predictor_hq"]:
    if var_name in globals():
        del globals()[var_name]

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    print(f"VRAM reserved:  {torch.cuda.memory_reserved() / 1e9:.3f} GB")
    print("\n✅ GPU memory cleared!")
    print("   VRAM allocated が 0 に近くない場合は「ランタイム」→「セッションを再起動」を実行してください。")

# ============================================================
# セル1: セットアップ & インストール
# ============================================================
!git clone https://github.com/SysCV/sam-hq.git
import os
os.chdir("sam-hq/sam-hq2")
!pip install -e ".[notebooks]" -q
!mkdir -p checkpoints
print("Setup complete!")

# ============================================================
# セル2: モデルチェックポイントのダウンロード（Large版）
# ============================================================
!wget -q -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
!wget -q -P checkpoints https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt
print("Large checkpoints downloaded!")

# ============================================================
# セル3: ライブラリのインポート
# ============================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# ============================================================
# セル4: ベースラインSAM 2.1用パッチ関数
# ============================================================

def patch_baseline_model(model):
    """
    ベースラインSAM 2.1にHQ-SAM2のkwargsを無視するパッチを適用。
    HQ-SAM2のpredictorが渡す hq_token_only / interm_embeddings を無視する。
    """
    _orig_forward = model.forward
    def _patched_forward(*args, **kwargs):
        kwargs.pop("hq_token_only", None)
        kwargs.pop("interm_embeddings", None)
        return _orig_forward(*args, **kwargs)
    model.forward = _patched_forward

    decoder = None
    for attr_name in ["sam_mask_decoder", "mask_decoder", "decoder"]:
        if hasattr(model, attr_name):
            decoder = getattr(model, attr_name)
            break
    if decoder is None:
        for name, module in model.named_modules():
            if "mask_decoder" in name.lower() or "decoder" in type(module).__name__.lower():
                if hasattr(module, "forward"):
                    decoder = module
                    break
    if decoder is not None and hasattr(decoder, "forward"):
        _orig_decoder = decoder.forward
        def _patched_decoder(*args, **kwargs):
            kwargs.pop("hq_token_only", None)
            kwargs.pop("interm_embeddings", None)
            return _orig_decoder(*args, **kwargs)
        decoder.forward = _patched_decoder
        print(f"  -> Patched decoder: {type(decoder).__name__}")
    else:
        print("  -> Warning: decoder not found, forward patch only")
    return model


def free_gpu_memory():
    """GPUメモリを解放"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if torch.cuda.is_available():
        print(f"  VRAM freed. Allocated: {torch.cuda.memory_allocated()/1e9:.3f} GB")


device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# セル5: サンプル画像の取得
# ============================================================
url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/kite.jpeg"
image = np.array(Image.open(BytesIO(requests.get(url).content)).convert("RGB"))
print(f"Image shape: {image.shape}")

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")
plt.show()

# ============================================================
# セル6: プロンプトの設定（点またはbboxで対象を固定）
# ============================================================
print("=" * 60)
print("プロンプト設定")
print("=" * 60)
print(f"画像サイズ: {image.shape[1]} x {image.shape[0]}")
print("対象を指定する方法を選んでください:")
print("  1: 点プロンプト（対象上の1点）")
print("  2: バウンディングボックス（矩形）")

try:
    choice = input("選択 (1 または 2): ").strip()
except EOFError:
    choice = "1"

if choice == "2":
    # bboxモード
    print("\nbboxの座標を入力してください [x1, y1, x2, y2]")
    try:
        x1 = int(input("x1 (左上x): "))
        y1 = int(input("y1 (左上y): "))
        x2 = int(input("x2 (右下x): "))
        y2 = int(input("y2 (右下y): "))
    except (ValueError, EOFError):
        print("無な入力。デフォルトbbox [130, 40, 340, 260] を使用します。")
        x1, y1, x2, y2 = 130, 40, 340, 260
    input_box = np.array([x1, y1, x2, y2])
    input_point = None
    input_label = None
    prompt_type = "box"
else:
    # 点プロンプトモード（デフォルト）
    print("\n対象上の点の座標を入力してください [x, y]")
    try:
        x = int(input("x 座標: "))
        y = int(input("y 座標: "))
    except (ValueError, EOFError):
        print("無効な入力。デフォルト (230, 150) を使用します。")
        x, y = 230, 150
    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    input_box = None
    prompt_type = "point"

# プロンプト可視化
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(image)
if prompt_type == "point":
    ax.scatter(input_point[:, 0], input_point[:, 1], c="red", s=200,
               marker="*", edgecolors="white", linewidths=2, zorder=5)
    ax.set_title(f"Point Prompt: ({input_point[0,0]}, {input_point[0,1]})")
else:
    rect = plt.Rectangle((input_box[0], input_box[1]),
                         input_box[2] - input_box[0],
                         input_box[3] - input_box[1],
                         fill=False, edgecolor="red", linewidth=2)
    ax.add_patch(rect)
    ax.set_title(f"Box Prompt: [{input_box[0]}, {input_box[1]}, {input_box[2]}, {input_box[3]}]")
ax.axis("off")
plt.show()

# ============================================================
# セル7: SAM 2.1 Large で推論（モデル1のみGPU使用）
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: SAM 2.1 (Large) 推論")
print("=" * 60)

sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_ckpt = "checkpoints/sam2.1_hiera_large.pt"

print("Loading SAM 2.1 Large...")
print(f"  VRAM before load: {torch.cuda.memory_allocated()/1e9:.3f} GB")
model_sam2 = build_sam2(sam2_cfg, sam2_ckpt, device=device)
model_sam2 = patch_baseline_model(model_sam2)
predictor_sam2 = SAM2ImagePredictor(model_sam2)
print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.3f} GB")

print("Predicting mask...")
with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
    predictor_sam2.set_image(image)
    if prompt_type == "point":
        masks_sam2, scores_sam2, _ = predictor_sam2.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=True,
        )
    else:
        masks_sam2, scores_sam2, _ = predictor_sam2.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
    best_idx_sam2 = int(np.argmax(scores_sam2))
    mask_sam2_best = masks_sam2[best_idx_sam2]

print(f"  Candidates: {len(scores_sam2)}, Best score: {scores_sam2[best_idx_sam2]:.3f}")

# マスクをCPUにコピーしてモデルを解放
mask_sam2_arr = mask_sam2_best.copy()
print("Unloading SAM 2.1 from GPU...")
del model_sam2, predictor_sam2
free_gpu_memory()

# ============================================================
# セル8: HQ-SAM 2 Large で推論（モデル2のみGPU使用）
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: HQ-SAM 2 (Large) 推論")
print("=" * 60)

hq_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
hq_ckpt = "checkpoints/sam2.1_hq_hiera_large.pt"

print("Loading HQ-SAM 2 Large...")
print(f"  VRAM before load: {torch.cuda.memory_allocated()/1e9:.3f} GB")
model_hq = build_sam2(hq_cfg, hq_ckpt, device=device)
predictor_hq = SAM2ImagePredictor(model_hq)
print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.3f} GB")

print("Predicting mask...")
with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
    predictor_hq.set_image(image)
    if prompt_type == "point":
        masks_hq, scores_hq, _ = predictor_hq.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=True,
        )
    else:
        masks_hq, scores_hq, _ = predictor_hq.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
    best_idx_hq = int(np.argmax(scores_hq))
    mask_hq_best = masks_hq[best_idx_hq]

print(f"  Candidates: {len(scores_hq)}, Best score: {scores_hq[best_idx_hq]:.3f}")

mask_hq_arr = mask_hq_best.copy()
print("Unloading HQ-SAM 2 from GPU...")
del model_hq, predictor_hq
free_gpu_memory()

# ============================================================
# セル9: IoU確（同じ物体を比較できているか）
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: マスク一致度確認")
print("=" * 60)

intersection = np.logical_and(mask_sam2_arr, mask_hq_arr).sum()
union = np.logical_or(mask_sam2_arr, mask_hq_arr).sum()
iou = intersection / union if union > 0 else 0.0

print(f"IoU between SAM 2.1 and HQ-SAM 2 masks: {iou:.3f}")
if iou >= 0.7:
    print("  ✅ 同じ物体を比較できています（IoU >= 0.7）")
elif iou >= 0.5:
    print("  ⚠️  やや一致（IoU 0.5〜0.7）。結果を確認してください。")
else:
    print("  ❌ 一致度が低いです（IoU < 0.5）。プロンプトを調整してください。")

# ============================================================
# セル10: 輪郭抽出 & 可視化
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: 輪郭抽出 & 可視化")
print("=" * 60)

def get_contours(mask_arr):
    m = mask_arr.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnts

cnt_sam2 = get_contours(mask_sam2_arr)
cnt_hq = get_contours(mask_hq_arr)

img_sam2 = image.copy()
cv2.drawContours(img_sam2, cnt_sam2, -1, (255, 0, 0), 2)
img_hq = image.copy()
cv2.drawContours(img_hq, cnt_hq, -1, (0, 255, 0), 2)
img_both = image.copy()
cv2.drawContours(img_both, cnt_sam2, -1, (255, 0, 0), 2)
cv2.drawContours(img_both, cnt_hq, -1, (0, 255, 0), 2)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0,0].imshow(image); axes[0,0].set_title("Original"); axes[0,0].axis("off")
axes[0,1].imshow(mask_sam2_arr, cmap="gray"); axes[0,1].set_title(f"SAM 2.1 Mask (score={scores_sam2[best_idx_sam2]:.3f})"); axes[0,1].axis("off")
axes[0,2].imshow(mask_hq_arr, cmap="gray"); axes[0,2].set_title(f"HQ-SAM 2 Mask (score={scores_hq[best_idx_hq]:.3f})"); axes[0,2].axis("off")
axes[1,0].imshow(img_sam2); axes[1,0].set_title("SAM 2.1 (Red)"); axes[1,0].axis("off")
axes[1,1].imshow(img_hq); axes[1,1].set_title("HQ-SAM 2 (Green)"); axes[1,1].axis("off")
axes[1,2].imshow(img_both); axes[1,2].set_title("Overlay: SAM(Red) vs HQ(Green)"); axes[1,2].axis("off")
plt.tight_layout()
plt.savefig("/home/user/outputs/hqsam2_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# セル11: 差分マスク
# ============================================================
mask_diff = np.abs(mask_hq_arr.astype(float) - mask_sam2_arr.astype(float))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(mask_sam2_arr, cmap="gray", vmin=0, vmax=1); axes[0].set_title("SAM 2.1"); axes[0].axis("off")
axes[1].imshow(mask_hq_arr, cmap="gray", vmin=0, vmax=1); axes[1].set_title("HQ-SAM 2"); axes[1].axis("off")
im = axes[2].imshow(mask_diff, cmap="hot", vmin=0, vmax=1); axes[2].set_title("Difference"); axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046)
plt.tight_layout()
plt.savefig("/home/user/outputs/hqsam2_diff.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Difference pixels (>0.5): {np.sum(mask_diff > 0.5):,} / {mask_diff.size:,} ({100*np.sum(mask_diff>0.5)/mask_diff.size:.2f}%)")

# ============================================================
# セル12: 境界詳細度の定量的比較
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: 境界詳細度の定量的比較")
print("=" * 60)

pts_sam2 = sum(len(c) for c in cnt_sam2)
pts_hq = sum(len(c) for c in cnt_hq)
len_sam2 = sum(cv2.arcLength(c, False) for c in cnt_sam2) if cnt_sam2 else 0
len_hq = sum(cv2.arcLength(c, False) for c in cnt_hq) if cnt_hq else 0

print(f"SAM 2.1:   points={pts_sam2:,}, length={len_sam2:.1f}")
print(f"HQ-SAM 2:  points={pts_hq:,}, length={len_hq:.1f}")
if pts_sam2 > 0:
    print(f"Point ratio (HQ/SAM): {pts_hq/pts_sam2:.2f}x")
if len_sam2 > 0:
    print(f"Length ratio (HQ/SAM): {len_hq/len_sam2:.2f}x")

# ============================================================
# セル13: 境界の拡大表示
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: 境界の拡大表示")
print("=" * 60)

def crop_to_contour(image, contours, padding=40):
    if not contours:
        return image, (0, 0, image.shape[1], image.shape[0])
    all_pts = np.vstack(contours)
    x, y, bw, bh = cv2.boundingRect(all_pts)
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(image.shape[1], x + bw + padding), min(image.shape[0], y + bh + padding)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)

crop, (x1, y1, x2, y2) = crop_to_contour(image, cnt_hq if cnt_hq else cnt_sam2, padding=40)

crop_sam = image[y1:y2, x1:x2].copy()
if cnt_sam2:
    cv2.drawContours(crop_sam, [c - np.array([[x1, y1]]) for c in cnt_sam2], -1, (255, 0, 0), 2)
crop_hq_img = image[y1:y2, x1:x2].copy()
if cnt_hq:
    cv2.drawContours(crop_hq_img, [c - np.array([[x1, y1]]) for c in cnt_hq], -1, (0, 255, 0), 2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(crop); axes[0].set_title("Cropped Region"); axes[0].axis("off")
axes[1].imshow(crop_sam); axes[1].set_title("SAM 2.1 (Red)"); axes[1].axis("off")
axes[2].imshow(crop_hq_img); axes[2].set_title("HQ-SAM 2 (Green)"); axes[2].axis("off")
plt.tight_layout()
plt.savefig("/home/user/outputs/hqsam2_zoom.png", dpi=200, bbox_inches="tight")
plt.show()

print("\n✅ HQ-SAM 2 vs SAM 2.1 PoC complete!")
