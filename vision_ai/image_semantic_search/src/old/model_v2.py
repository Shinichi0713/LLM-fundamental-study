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
# 1. パスの設定とモジュールキャッシュのクリア
# ----------------------------------------------------------------------
REGIONCLIP_REPO = "/content/RegionCLIP"
if REGIONCLIP_REPO not in sys.path:
    sys.path.insert(0, REGIONCLIP_REPO)

# メモリ上の標準 detectron2 キャッシュを消去し、RegionCLIP 版を優先読み込み
for m in [k for k in sys.modules if k.startswith("detectron2")]:
    del sys.modules[m]

import numpy as np
import torch
from torch.nn import functional as F

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
    cfg = get_cfg()

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
    results = model._postprocess(pred_instances, batched_instances=pred_instances, inputs=batched_inputs) if hasattr(model, "_postprocess") else model._postprocess(pred_instances, batched_inputs)

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
# 1. パスの通し込み (RegionCLIP 固有の config やモデル定義を参照)
# ----------------------------------------------------------------------
REGIONCLIP_REPO = "/content/RegionCLIP"
REGIONCLIP_SRC = os.path.join(REGIONCLIP_REPO, "src")

for p in [REGIONCLIP_REPO, REGIONCLIP_SRC]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer

# RegionCLIP 固有の Config 拡張をインポート
from regionclip.config import add_regionclip_config

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
    cfg = get_cfg()
    add_regionclip_config(cfg)
    cfg.merge_from_file(CONFIG_FILE)

    opts = [
        "MODEL.WEIGHTS", MODEL_WEIGHTS,
        "MODEL.CLIP.CROP_REGION_TYPE", "RPN",
        "MODEL.CLIP.MULTIPLY_RPN_SCORE", "True",
        "MODEL.CLIP.OFFLINE_RPN_CONFIG", OFFLINE_RPN_CONFIG,
        "MODEL.CLIP.BB_RPN_WEIGHTS", RPN_WEIGHTS,
        "MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED", "True",
        "MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST", "100",
        "MODEL.DEVICE", DEVICE,
    ]
    if TEXT_EMB_PATH is not None:
        opts += ["MODEL.CLIP.TEXT_EMB_PATH", TEXT_EMB_PATH]
    if OPENSET_TEST_TEXT_EMB_PATH is not None:
        opts += ["MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH", OPENSET_TEST_TEXT_EMB_PATH]

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
        cfg.MODEL.META_ARCHITECTURE in ["CLIPRCNN", "CLIPFastRCNN", "PretrainFastRCNN"]
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

    image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return [{"image": image_tensor, "height": height, "width": width}]

@torch.no_grad()
def extract_region_features(model, cfg, batched_inputs):
    # RPN 領域提案
    images = model.offline_preprocess_image(batched_inputs)
    offline_feats = model.offline_backbone(images.tensor)
    proposals, _ = model.offline_proposal_generator(images, offline_feats, None)

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


import os
import shutil
import detectron2

# 公式 detectron2 のインストール先から _C バイナリを特定
site_packages_d2 = os.path.dirname(detectron2.__file__)
c_ext_file = None
for f in os.listdir(site_packages_d2):
    if f.startswith("_C") and (f.endswith(".so") or f.endswith(".pyd")):
        c_ext_file = f
        break

print(f"検出された C++ 拡張ファイル: {c_ext_file}")

# RegionCLIP 側の detectron2 にシンボリックリンクを作成
src_path = os.path.join(site_packages_d2, c_ext_file)
dst_dir = "/content/RegionCLIP/detectron2"
dst_path = os.path.join(dst_dir, c_ext_file)

if os.path.exists(src_path):
    if os.path.exists(dst_path) or os.path.islink(dst_path):
        os.remove(dst_path)
    os.symlink(src_path, dst_path)
    print(f"RegionCLIP へのリンク作成が完了しました: {dst_path}")

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

# 公式 detectron2 のインポートキャッシュをクリア
for m in [k for k in sys.modules if k.startswith("detectron2")]:
    del sys.modules[m]

import numpy as np
import torch
from torch.nn import functional as F

# RegionCLIP 側の detectron2 からインポート
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, add_regionclip_config
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
    cfg = get_cfg()
    add_regionclip_config(cfg)
    cfg.merge_from_file(CONFIG_FILE)

    opts = [
        "MODEL.WEIGHTS", MODEL_WEIGHTS,
        "MODEL.CLIP.CROP_REGION_TYPE", "RPN",
        "MODEL.CLIP.MULTIPLY_RPN_SCORE", "True",
        "MODEL.CLIP.OFFLINE_RPN_CONFIG", OFFLINE_RPN_CONFIG,
        "MODEL.CLIP.BB_RPN_WEIGHTS", RPN_WEIGHTS,
        "MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED", "True",
        "MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST", "100",
        "MODEL.DEVICE", DEVICE,
    ]
    if TEXT_EMB_PATH is not None:
        opts += ["MODEL.CLIP.TEXT_EMB_PATH", TEXT_EMB_PATH]
    if OPENSET_TEST_TEXT_EMB_PATH is not None:
        opts += ["MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH", OPENSET_TEST_TEXT_EMB_PATH]

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
        cfg.MODEL.META_ARCHITECTURE in ["CLIPRCNN", "CLIPFastRCNN", "PretrainFastRCNN"]
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

    image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return [{"image": image_tensor, "height": height, "width": width}]

@torch.no_grad()
def extract_region_features(model, cfg, batched_inputs):
    # RPN 領域提案
    images = model.offline_preprocess_image(batched_inputs)
    offline_feats = model.offline_backbone(images.tensor)
    proposals, _ = model.offline_proposal_generator(images, offline_feats, None)

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