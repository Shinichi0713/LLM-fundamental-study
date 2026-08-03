import os
import sys

# 1. RegionCLIP リポジトリおよび src ディレクトリを Python パスに追加
REGIONCLIP_DIR = "/content/RegionCLIP"
REGIONCLIP_SRC = os.path.join(REGIONCLIP_DIR, "src")

for path in [REGIONCLIP_DIR, REGIONCLIP_SRC]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

# 2. インポートテスト
try:
    from regionclip.config import add_regionclip_config
    print("RegionCLIP モジュールの読み込みに成功しました！")
except ModuleNotFoundError as e:
    print(f"インポートエラー: {e}")
    print("現在の sys.path 内に regionclip フォルダが存在するか確認してください。")

import os

print("=== /content/RegionCLIP の内部構造 ===")
target_dir = "/content/RegionCLIP"

for root, dirs, files in os.walk(target_dir):
    depth = root.replace(target_dir, "").count(os.sep)
    if depth <= 2:  # 2階層目まで表示
        indent = "  " * depth
        print(f"{indent}[D] {os.path.basename(root)}/")
        sub_indent = "  " * (depth + 1)
        for f in files[:5]:  # 各フォルダ最大5ファイルまで表示
            print(f"{sub_indent}- {f}")

import os
import sys

# ----------------------------------------------------------------------
# 1. パスの自動検索と設定
# ----------------------------------------------------------------------


def find_and_add_regionclip():
    # 1. 定番のパスを優先チェック
    candidate_paths = [
        "/content/RegionCLIP",
        "/content/detectron2",
        os.getcwd(),
    ]
    for path in candidate_paths:
        if os.path.exists(os.path.join(path, "regionclip")):
            if path not in sys.path:
                sys.path.insert(0, path)
            print(f"regionclip モジュールを検出しました: {path}")
            return path

    # 2. 見つからない場合は /content 以下を自動探索
    for root, dirs, _ in os.walk("/content"):
        if "regionclip" in dirs:
            if root not in sys.path:
                sys.path.insert(0, root)
            print(f"regionclip モジュールを自動発見しました: {root}")
            return root

    raise FileNotFoundError(
        "regionclip フォルダが見つかりませんでした。`git clone https://github.com/microsoft/RegionCLIP.git` が実行されているか確認してください。"
    )


# 自動検索とパスの通し込みを実行
REGIONCLIP_REPO = find_and_add_regionclip()

# モジュールのインポート（パスが通った後に呼び出し）
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
import numpy as np
from regionclip.config import add_regionclip_config
import torch
from torch.nn import functional as F

print("すべてのインポートが正常に完了しました！")

"""
RegionCLIP: 学習済み重みを直接ロードして、画像から領域(box)と埋め込み表現(feats)を抽出するスクリプト
"""

import os
import sys
import numpy as np
import torch
from torch.nn import functional as F

# ----------------------------------------------------------------------
# 1. パスとモジュールの優先設定
# ----------------------------------------------------------------------
REGIONCLIP_REPO = "/content/RegionCLIP"
if REGIONCLIP_REPO not in sys.path:
    sys.path.insert(0, REGIONCLIP_REPO)

# モジュールキャッシュのクリア
for m in [k for k in sys.modules if k.startswith("detectron2")]:
    del sys.modules[m]

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import add_regionclip_config, get_cfg
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer

# ----------------------------------------------------------------------
# 2. 設定ファイルのパス指定
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
    add_regionclip_config(cfg)  # RegionCLIP 独自スキーマを追加
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


import os

print("=== RegionCLIP 内の Config 設定関数を検索中 ===")
found = False

for root, dirs, files in os.walk("/content/RegionCLIP"):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if "def add_" in line and "config" in line:
                            print(f"発見 ({filepath}:{line_num}): {line.strip()}")
                            found = True
            except Exception:
                pass

if not found:
    print("条件に一致する `def add_*_config` 関数が見つかりませんでした。")

print("\n=== tools/plain_train_net.py または train_net.py の import 文を確認 ===")
for tool_file in ["tools/plain_train_net.py", "tools/train_net.py"]:
    full_path = os.path.join("/content/RegionCLIP", tool_file)
    if os.path.exists(full_path):
        print(f"\n--- {tool_file} ---")
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "import" in line and ("config" in line or "clip" in line):
                    print(line.strip())