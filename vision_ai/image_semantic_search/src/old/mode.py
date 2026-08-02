import os
import sys

# 1. 発見されたルートフォルダを Python パスに最優先で追加
sys.path.insert(0, "/content/detectron2")
sys.path.insert(0, "/content/detectron2/detectron2")

import detectron2.checkpoint as checkpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T


# ---------------------------------------------------------
# 2. Config のセットアップ
# ---------------------------------------------------------
def setup_regionclip_cfg(config_file_path: str, weight_path: str):
    cfg = get_cfg()

    # RegionCLIP 固有の Config キーを直接追加
    cfg.MODEL.CLIP = get_cfg()
    cfg.MODEL.CLIP.TEXT_EMB_PATH = ""
    cfg.MODEL.CLIP.TEXT_EMB_DIM = 512
    cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG = ""
    cfg.MODEL.CLIP.BB_RPN_WEIGHTS = ""
    cfg.MODEL.CLIP.CROP_REGION_TYPE = "grid"
    cfg.MODEL.CLIP.MULTIPLY_ZERO_WEIGHT = False
    cfg.MODEL.CLIP.NO_TEXT_EMB = False
    cfg.MODEL.CLIP.CLSS_NAME_PATH = ""

    if os.path.exists(config_file_path):
        cfg.merge_from_file(config_file_path)

    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


config_path = "/content/detectron2/configs/CLIP_FastRCNN_R50x4_C4.yaml"
weight_path = "/content/detectron2/downloaded_file.pth"

cfg = setup_regionclip_cfg(config_path, weight_path)


# ---------------------------------------------------------
# 3. モデルの構築と重みのロード
# ---------------------------------------------------------
print("RegionCLIP モデルを構築中...")
model = build_model(cfg)
model.eval()

# 重みファイルの読み込み
checkpointer.DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
print("重みファイルのロードが正常に完了しました！")


# ---------------------------------------------------------
# 4. 領域切り出しと特徴量抽出・類似度計算
# ---------------------------------------------------------
def process_matching(
    image_path: str, boxes: list[list[float]], texts: list[str]
):
    image_pil = Image.open(image_path).convert("RGB")

    # 1. 画像領域（クロップ）の前処理
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    region_tensors = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image_pil.crop((x1, y1, x2, y2))
        region_tensors.append(transform(crop))

    region_batch = torch.stack(region_tensors).to(cfg.MODEL.DEVICE)

    with torch.no_grad():
        # 2. GeneralizedRCNN からの視覚特徴量（Region Embeddings）抽出
        # 構造に応じて適切なバックボーン / アテンションプールを呼び出し
        if hasattr(model, "backbone"):
            feat = model.backbone(region_batch)
            if isinstance(feat, dict):
                feat = list(feat.values())[-1]  # 最終層の Feature Map

            # Attentional Pooling / Visual Projection 適用
            if hasattr(model.backbone, "attnpool"):
                region_features = model.backbone.attnpool(feat)
            elif hasattr(model, "roi_heads") and hasattr(
                model.roi_heads, "box_predictor"
            ):
                region_features = feat
            else:
                region_features = (
                    feat.mean(dim=[-2, -1]) if feat.ndim == 4 else feat
                )
        else:
            region_features = model(region_batch)

        region_embeds = F.normalize(region_features, p=2, dim=-1)

        # 3. テキスト特徴量（Text Embeddings）の抽出
        # GeneralizedRCNN の下層にテキスト用プロジェクションが存在するか判定
        tokens = clip.tokenize(texts).to(cfg.MODEL.DEVICE)

        if hasattr(model, "test_text_features") and getattr(
            model, "test_text_features", None
        ) is not None:
            text_embeds = model.test_text_features
        elif hasattr(model, "text_encoder"):
            text_features = model.text_encoder(tokens)
            text_embeds = F.normalize(text_features, p=2, dim=-1)
        else:
            # 標準 CLIP のテキストエンコーダを利用（RN50x4 対応）
            clip_model, _ = clip.load("RN50x4", device=cfg.MODEL.DEVICE)
            text_features = clip_model.encode_text(tokens)
            text_embeds = F.normalize(text_features, p=2, dim=-1)

        # 4. コサイン類似度の計算
        similarity_matrix = torch.matmul(region_embeds, text_embeds.T)

    return similarity_matrix

# ---------------------------------------------------------
# 5. 実行
# ---------------------------------------------------------
img_path = "/content/detectron2/450-20141030191952193143.jpg"

# バウンディングボックス例: [x1, y1, x2, y2]
bounding_boxes = [
    [50, 100, 200, 300],
    [220, 120, 380, 250],
    [400, 80, 580, 220],
    [250, 300, 350, 450],
]

# 照合用テキスト
text_queries = [
    "a red ceramic mug",
    "a white round plate",
    "a blue mug on the table",
    "a fresh green apple",
]

if os.path.exists(img_path):
    sim_matrix = process_matching(img_path, bounding_boxes, text_queries)

    print("\n=== 類似度行列 (Similarity Matrix) ===")
    print(sim_matrix.cpu().numpy().round(3))

    top1_indices = torch.argmax(sim_matrix, dim=1)
    print("\n=== 各領域の Top-1 マッチング結果 ===")
    for idx, text_idx in enumerate(top1_indices):
        score = sim_matrix[idx, text_idx].item()
        print(
            f"領域 {idx+1} (Box: {bounding_boxes[idx]}) -> 予測: '{text_queries[text_idx]}' (Score: {score:.3f})"
        )
else:
    print(
        f"画像ファイルが存在しません: {img_path}\nパスを確認して再実行してください。"
    )

import os
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import torch
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------
# 1. デバイスの設定とモデル/プロセッサのロード
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "/content/detectron2/downloaded_file.pth"
base_model_name = "openai/clip-vit-base-patch32"

print("ベースモデルとプロセッサをロード中...")
processor = CLIPProcessor.from_pretrained(base_model_name)
model = CLIPModel.from_pretrained(base_model_name)

# 重みファイルの読み込み
checkpoint = torch.load(checkpoint_path, map_location="cpu")
if isinstance(checkpoint, dict) and "model" in checkpoint:
    raw_state_dict = checkpoint["model"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    raw_state_dict = checkpoint["state_dict"]
else:
    raw_state_dict = checkpoint


# --- キー名のマッピング関数 ---
def convert_detectron2_to_hf_clip(d2_state_dict, model_state_dict):
    """Detectron2 / RegionCLIP のキー名を Hugging Face CLIP のキー名に変換する"""
    mapped_state_dict = {}

    # 代表的なプレフィックスの置換ルール
    for d2_key, v in d2_state_dict.items():
        hf_key = d2_key

        # Detectron2 特有のプレフィックスを除去・変換
        if hf_key.startswith("backbone.visual."):
            hf_key = hf_key.replace("backbone.visual.", "vision_model.")
        elif hf_key.startswith("visual."):
            hf_key = hf_key.replace("visual.", "vision_model.")
        elif hf_key.startswith("lang_encoder."):
            hf_key = hf_key.replace("lang_encoder.", "text_model.")

        # 変換後のキーがモデル側に存在するか確認
        if hf_key in model_state_dict:
            # 形状 (shape) が一致する場合のみ追加
            if v.shape == model_state_dict[hf_key].shape:
                mapped_state_dict[hf_key] = v
            else:
                print(
                    f"[Shape Mismatch Skiped] {d2_key} -> {hf_key}: {v.shape} vs {model_state_dict[hf_key].shape}"
                )

    return mapped_state_dict


# キー名のマッピング処理
converted_state_dict = convert_detectron2_to_hf_clip(
    raw_state_dict, model.state_dict()
)

# マッピングした重みを適用
missing_keys, unexpected_keys = model.load_state_dict(
    converted_state_dict, strict=False
)

print("\n=== 重みの適用結果 ===")
print(
    f"- 成功して読み込まれたキー数 : {len(converted_state_dict)} / {len(model.state_dict())}"
)
print(f"- 未一致のキー数 (Missing)  : {len(missing_keys)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()


# ---------------------------------------------------------
# Helper: 出力オブジェクトから Tensor を安全に取り出す関数
# ---------------------------------------------------------
def _extract_tensor(output) -> torch.Tensor:
    """モデルの出力（Tensor または dataclass オブジェクト）から 確実に対象の Tensor を取り出す"""
    if isinstance(output, torch.Tensor):
        return output
    elif hasattr(output, "image_embeds"):
        return output.image_embeds
    elif hasattr(output, "text_embeds"):
        return output.text_embeds
    elif hasattr(output, "pooler_output"):
        return output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]  # CLS トークン
    else:
        raise ValueError(
            f"未対応の出力型です: {type(output)}. Tensor の取り出しに失敗しました。"
        )


# ---------------------------------------------------------
# 2. 領域埋め込み（Region Embeddings）の抽出関数 (堅牢化版)
# ---------------------------------------------------------
def extract_region_embeddings(
    image: Image.Image, boxes: list[list[float]]
) -> torch.Tensor:
    region_crops = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image.crop((x1, y1, x2, y2))
        region_crops.append(crop)

    inputs = processor(images=region_crops, return_tensors="pt").to(device)

    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            raw_output = model.get_image_features(**inputs)
            region_features = _extract_tensor(raw_output)
        else:
            vision_outputs = model.vision_model(**inputs)
            feat_tensor = _extract_tensor(vision_outputs)

            if hasattr(model, "visual_projection"):
                region_features = model.visual_projection(feat_tensor)
            else:
                region_features = feat_tensor

    # 確実に Tensor であることを確認して L2 正規化
    region_embeddings = F.normalize(region_features, p=2, dim=-1)
    return region_embeddings


# ---------------------------------------------------------
# 3. テキスト埋め込み（Text Embeddings）の抽出関数 (堅牢化版)
# ---------------------------------------------------------
def extract_text_embeddings(texts: list[str]) -> torch.Tensor:
    inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            raw_output = model.get_text_features(**inputs)
            text_features = _extract_tensor(raw_output)
        else:
            text_outputs = model.text_model(**inputs)
            feat_tensor = _extract_tensor(text_outputs)

            if hasattr(model, "text_projection"):
                text_features = model.text_projection(feat_tensor)
            else:
                text_features = feat_tensor

    # 確実に Tensor であることを確認して L2 正規化
    text_embeddings = F.normalize(text_features, p=2, dim=-1)
    return text_embeddings


# ---------------------------------------------------------
# 4. 実行用サンプル (パイプライン全体の検証)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 画像の読み込み
    try:
        sample_image = Image.open(
            "/content/detectron2/450-20141030191952193143.jpg"
        )
    except FileNotFoundError:
        print("指定パスの画像が見つからないため、ダミー画像で実行します。")
        sample_image = Image.new("RGB", (640, 480), color=(240, 240, 240))

    # クロップ用領域座標 (Bounding Boxes: [x1, y1, x2, y2])
    bounding_boxes = [
        [50, 100, 200, 300],
        [220, 120, 380, 250],
        [400, 80, 580, 220],
        [250, 300, 350, 450],
    ]

    # 検索用テキストフレーズ
    text_queries = [
        "a red tomate",
        "a plate with rice",
        "piled plates",
        "a green vegetable",
    ]

    # --- 埋め込みベクトルの抽出 ---
    print("1. 領域埋め込み抽出中...")
    region_embeds = extract_region_embeddings(sample_image, bounding_boxes)
    print(
        f"   -> 抽出完了: 領域数 {region_embeds.shape[0]}, 次元数 {region_embeds.shape[1]}"
    )

    print("2. テキスト埋め込み抽出中...")
    text_embeds = extract_text_embeddings(text_queries)
    print(
        f"   -> 抽出完了: テキスト数 {text_embeds.shape[0]}, 次元数 {text_embeds.shape[1]}"
    )

    # --- 類似度行列 (コサイン類似度) の計算 ---
    similarity_matrix = torch.matmul(region_embeds, text_embeds.T)

    print("\n=== 類似度行列 (Similarity Matrix) ===")
    print(similarity_matrix.cpu().numpy().round(3))

    # 各領域に対して最も一致度の高いテキストの判定 (Top-1 予測)
    top1_indices = torch.argmax(similarity_matrix, dim=1)
    print("\n=== 各領域のTop-1マッチング結果 ===")
    for idx, text_idx in enumerate(top1_indices):
        score = similarity_matrix[idx, text_idx].item()
        print(
            f"領域 {idx+1} (Box: {bounding_boxes[idx]}) -> 予測: '{text_queries[text_idx]}' (Score: {score:.3f})"
        )

"""
RegionCLIP: 学習済み重みを直接ロードして、画像から領域(box)と埋め込み表現(feats)を抽出するスクリプト
"""

import os
import sys
import PIL.Image

# ----------------------------------------------------------------------
# 0. Pillow (PIL) v10+ 互換性パッチ
# ----------------------------------------------------------------------
if not hasattr(PIL.Image, "LINEAR"):
    if hasattr(PIL.Image, "Resampling"):
        PIL.Image.LINEAR = PIL.Image.Resampling.BILINEAR
    elif hasattr(PIL.Image, "BILINEAR"):
        PIL.Image.LINEAR = PIL.Image.BILINEAR

for attr in ["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS", "BOX", "HAMMING"]:
    if not hasattr(PIL.Image, attr) and hasattr(PIL.Image, "Resampling"):
        setattr(PIL.Image, attr, getattr(PIL.Image.Resampling, attr))

if not hasattr(PIL.Image, "ANTIALIAS") and hasattr(PIL.Image, "Resampling"):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

# ----------------------------------------------------------------------
# 1. RegionCLIP 版 Detectron2 を優先読み込み
# ----------------------------------------------------------------------
REGIONCLIP_REPO = "/content/RegionCLIP"
if REGIONCLIP_REPO not in sys.path:
    sys.path.insert(0, REGIONCLIP_REPO)

# 公式 detectron2 のインポートキャッシュをクリアして RegionCLIP 版に切り替え
for m in [k for k in sys.modules if k.startswith("detectron2")]:
    del sys.modules[m]

import numpy as np
import torch
from torch.nn import functional as F

# RegionCLIP 側の detectron2 からインポート（add_regionclip_config は不要です）
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer

# ----------------------------------------------------------------------
# 2. 設定ファイルのパスと重みファイル
# ----------------------------------------------------------------------
CONFIG_FILE = os.path.join(
    REGIONCLIP_REPO,
    "configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml",
)
OFFLINE_RPN_CONFIG = os.path.join(
    REGIONCLIP_REPO,
    "configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
)

MODEL_WEIGHTS = "/content/regionclip_finetuned-lvis_rn50.pth"
RPN_WEIGHTS = "/content/rpn_lvis_866_lsj.pth"
TEXT_EMB_PATH = "/content/lvis_1203_cls_emb.pth"
OPENSET_TEST_TEXT_EMB_PATH = "/content/lvis_1203_cls_emb.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------
# 3. Config とモデル構築
# ----------------------------------------------------------------------


def build_cfg():
    cfg = (
        get_cfg()
    )  # RegionCLIP 版 get_cfg により MODEL.CLIP などの拡張設定が自動で読み込まれます
    cfg.merge_from_file(CONFIG_FILE)

    opts = [
        "MODEL.WEIGHTS",
        MODEL_WEIGHTS,
        "MODEL.CLIP.CROP_REGION_TYPE",
        "RPN",
        "MODEL.CLIP.MULTIPLY_RPN_SCORE",
        "True",
        "MODEL.CLIP.OFFLINE_RPN_CONFIG",
        OFFLINE_RPN_CONFIG,
        "MODEL.CLIP.BB_RPN_WEIGHTS",
        RPN_WEIGHTS,
        "MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED",
        "True",
        "MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST",
        "100",
        "MODEL.DEVICE",
        DEVICE,
    ]
    if TEXT_EMB_PATH is not None:
        opts += ["MODEL.CLIP.TEXT_EMB_PATH", TEXT_EMB_PATH]
    if OPENSET_TEST_TEXT_EMB_PATH is not None:
        opts += [
            "MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH",
            OPENSET_TEST_TEXT_EMB_PATH,
        ]

    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def build_model(cfg):
    model = DefaultTrainer.build_model(cfg)

    # 重みのロード
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )

    if (
        cfg.MODEL.META_ARCHITECTURE
        in ["CLIPRCNN", "CLIPFastRCNN", "PretrainFastRCNN"]
        and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None
        and cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN"
    ):
        DetectionCheckpointer(
            model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True
        ).resume_or_load(cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False)

    model.roi_heads.box_predictor.vis = True
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


# ----------------------------------------------------------------------
# 4. 前処理と特徴量抽出
# ----------------------------------------------------------------------


def load_image(cfg, file_name):
    image = utils.read_image(file_name, format=cfg.INPUT.FORMAT)
    height, width = image.shape[0], image.shape[1]

    augs = utils.build_augmentation(cfg, False)
    aug_input = T.AugInput(image)
    T.AugmentationList(augs)(aug_input)
    image = aug_input.image

    image_tensor = torch.as_tensor(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    )
    return [{"image": image_tensor, "height": height, "width": width}]


@torch.no_grad()
def extract_region_features(model, cfg, batched_inputs):
    # RPN 領域提案
    images = model.offline_preprocess_image(batched_inputs)
    offline_feats = model.offline_backbone(images.tensor)
    proposals, _ = model.offline_proposal_generator(
        images, offline_feats, None
    )

    # 特徴抽出
    images2 = model.preprocess_image(batched_inputs)
    feats = model.backbone(images2.tensor)

    proposal_boxes = [p.proposal_boxes for p in proposals]
    box_features = model.roi_heads._shared_roi_transform(
        [feats[f] for f in model.roi_heads.in_features],
        proposal_boxes,
        model.backbone.layer4,
    )
    region_feats = model.backbone.attnpool(box_features)

    # クラス推論とフィルタリング
    predictions = model.roi_heads.box_predictor(region_feats)
    pred_instances, keep_indices = model.roi_heads.box_predictor.inference(
        predictions, proposals
    )
    results = model._postprocess(pred_instances, batched_inputs)

    boxes = results[0]["instances"].get("pred_boxes").tensor.cpu()
    classes = results[0]["instances"].get("pred_classes").cpu()
    probs = F.softmax(predictions[0], dim=-1)[keep_indices[0]].cpu()
    kept_feats = region_feats[keep_indices[0]].cpu()

    return {
        "boxes": boxes,
        "classes": classes,
        "probs": probs,
        "feats": kept_feats,
    }


# ----------------------------------------------------------------------
# 5. 実行
# ----------------------------------------------------------------------
if __name__ == "__main__":
    IMAGE_PATH = "/content/detectron2/450-20141030191952193143.jpg"

    cfg = build_cfg()
    model = build_model(cfg)

    batched_inputs = load_image(cfg, IMAGE_PATH)
    output = extract_region_features(model, cfg, batched_inputs)

    print("=== 特徴量の抽出に成功しました ===")
    print("boxes shape:", output["boxes"].shape)
    print("feats shape:", output["feats"].shape)
    print("classes (top 10):", output["classes"][:10])