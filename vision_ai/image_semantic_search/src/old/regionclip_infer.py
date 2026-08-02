"""
RegionCLIP: 学習済み重みを直接ロードして、画像から領域(box)と埋め込み表現(feats)を抽出する。

前提:
- RegionCLIPパッケージがインストール済みであること
  (git clone https://github.com/microsoft/RegionCLIP.git 済み、
   その中で `python -m pip install -e .` 済みの環境で実行してください)
- 以下の重みが手元にあること(finetuned版検出器 + RPN + 概念埋め込みの正しい組み合わせ。
  公式 visualize_transfer_learning.sh のRN50finetuned版の設定に準拠)
    - regionclip_finetuned-lvis_rn50.pth  (LVIS 866ベースカテゴリでfinetuning済みの検出器, ResNet50)
    - rpn_lvis_866_lsj.pth                (LSJ(large-scale jittering)で学習されたRPN。
                                            finetuned検出器と組みで使うRPNは、事前学習モデル用の
                                            rpn_lvis_866.pth とは別物なので注意)
    - lvis_1203_cls_emb.pth               (LVIS1203概念のテキスト埋め込み。
                                            TEXT_EMB_PATH / OPENSET_TEST_TEXT_EMB_PATH の
                                            両方にこの同じファイルを指定する)

tools/extract_region_features.py をCLIとして呼ぶのではなく、
モデル構築〜推論までを直接Pythonから呼び出せる形にしたもの。
中身は同スクリプトの create_model() / extract_region_feats() のロジックを
関数として抜き出したもの。
"""

import os
import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T


# ----------------------------------------------------------------------
# 1. 設定: 環境に合わせてパスを書き換えてください
# ----------------------------------------------------------------------

REGIONCLIP_REPO = "/content/RegionCLIP"  # git clone した場所 (configファイル参照用)

# finetuned検出器をカスタム画像に使う場合は custom_img 設定を使う
# (NUM_CLASSES=1203, MASK_ON=False, NO_BOX_DELTA=True などが定義済み)
CONFIG_FILE = os.path.join(
    REGIONCLIP_REPO,
    "configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml",
)
OFFLINE_RPN_CONFIG = os.path.join(
    REGIONCLIP_REPO,
    "configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
)

# 実際に保存した重みのパス
MODEL_WEIGHTS = "/content/regionclip_finetuned-lvis_rn50.pth"
RPN_WEIGHTS = "/content/rpn_lvis_866_lsj.pth"   # ← finetuned版用のRPN (lsj付き。従来のrpn_lvis_866.pthとは別物)
TEXT_EMB_PATH = "/content/lvis_1203_cls_emb.pth"          # 訓練時カテゴリ分類用
OPENSET_TEST_TEXT_EMB_PATH = "/content/lvis_1203_cls_emb.pth"  # テスト時カテゴリ分類用(同じファイルでOK)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# 2. cfg構築とモデルのロード
# ----------------------------------------------------------------------

def build_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE)
    opts = [
        "MODEL.WEIGHTS", MODEL_WEIGHTS,
        "MODEL.CLIP.CROP_REGION_TYPE", "RPN",
        "MODEL.CLIP.MULTIPLY_RPN_SCORE", "True",
        "MODEL.CLIP.OFFLINE_RPN_CONFIG", OFFLINE_RPN_CONFIG,
        "MODEL.CLIP.BB_RPN_WEIGHTS", RPN_WEIGHTS,
        "MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED", "True",  # finetuned版RPN(_lsj)を使う場合に必要
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
    """重みを直接ロードしてモデルを構築する(train_net.py / extract_region_features.py と同じ手順)"""
    model = DefaultTrainer.build_model(cfg)

    # 1) RegionCLIP本体(視覚エンコーダ + 分類ヘッド)の重みをロード
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )

    # 2) RPN(領域提案器)の重みを別途ロード
    if (
        cfg.MODEL.META_ARCHITECTURE in ["CLIPRCNN", "CLIPFastRCNN", "PretrainFastRCNN"]
        and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None
        and cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN"
    ):
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True).resume_or_load(
            cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
        )

    assert model.clip_crop_region_type == "RPN"
    assert model.use_clip_c4
    assert model.use_clip_attpool
    model.roi_heads.box_predictor.vis = True  # RPNスコアを掛ける前の確信度を取得
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


# ----------------------------------------------------------------------
# 3. 画像の読み込み・前処理
# ----------------------------------------------------------------------

def load_image(cfg, file_name):
    """1枚の画像を読み込み、モデルの入力形式に変換する"""
    image = utils.read_image(file_name, format=cfg.INPUT.FORMAT)
    height, width = image.shape[0], image.shape[1]  # 前処理前のサイズ(=最終出力の座標系)

    augs = utils.build_augmentation(cfg, False)
    aug_input = T.AugInput(image)
    T.AugmentationList(augs)(aug_input)
    image = aug_input.image

    image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return [{"image": image_tensor, "height": height, "width": width}]


# ----------------------------------------------------------------------
# 4. 推論: 領域(box)と埋め込み表現(feats)を抽出
# ----------------------------------------------------------------------

@torch.no_grad()
def extract_region_features(model, cfg, batched_inputs):
    """
    戻り値 (dict):
      boxes: [#領域数, 4]  元画像座標系でのバウンディングボックス (x1, y1, x2, y2)
      feats: [#領域数, d]  各領域の埋め込み表現 (RN50は d=1024)
      (TEXT_EMB_PATHを指定した場合のみ)
      classes: [#領域数]   予測クラスID
      probs:   [#領域数, C+1] 各クラスの確信度 (最後がbackground)
    """
    # (a) オフラインのRPNで領域候補を得る
    images = model.offline_preprocess_image(batched_inputs)
    offline_feats = model.offline_backbone(images.tensor)
    proposals, _ = model.offline_proposal_generator(images, offline_feats, None)

    # (b) 認識用バックボーンで特徴マップを得る
    images2 = model.preprocess_image(batched_inputs)
    feats = model.backbone(images2.tensor)

    # (c) 各領域の特徴を切り出してCLIPのAttentionPoolで埋め込みに変換
    proposal_boxes = [p.proposal_boxes for p in proposals]
    box_features = model.roi_heads._shared_roi_transform(
        [feats[f] for f in model.roi_heads.in_features],
        proposal_boxes,
        model.backbone.layer4,
    )
    region_feats = model.backbone.attnpool(box_features)  # ← これが領域埋め込み

    if cfg.MODEL.CLIP.TEXT_EMB_PATH is None:
        # クラス非依存: RPNが出した領域そのままの埋め込みを返す
        results = model._postprocess(proposals, batched_inputs)
        boxes = results[0]["instances"].get("proposal_boxes").tensor.cpu()
        return {"boxes": boxes, "feats": region_feats.cpu()}
    else:
        # 概念埋め込みを使って分類し、クラスごとのNMS後の領域を返す
        predictions = model.roi_heads.box_predictor(region_feats)
        pred_instances, keep_indices = model.roi_heads.box_predictor.inference(
            predictions, proposals
        )
        results = model._postprocess(pred_instances, batched_inputs)

        boxes = results[0]["instances"].get("pred_boxes").tensor.cpu()
        classes = results[0]["instances"].get("pred_classes").cpu()
        probs = F.softmax(predictions[0], dim=-1)[keep_indices[0]].cpu()
        kept_feats = region_feats[keep_indices[0]].cpu()

        return {"boxes": boxes, "classes": classes, "probs": probs, "feats": kept_feats}


# ----------------------------------------------------------------------
# 5. 実行例
# ----------------------------------------------------------------------

if __name__ == "__main__":
    IMAGE_PATH = "/content/RegionCLIP/datasets/custom_images/sample.jpg"  # 対象画像に変更

    cfg = build_cfg()
    model = build_model(cfg)

    batched_inputs = load_image(cfg, IMAGE_PATH)
    output = extract_region_features(model, cfg, batched_inputs)

    print("boxes shape:", output["boxes"].shape)
    print("feats shape:", output["feats"].shape)
    if "classes" in output:
        print("classes:", output["classes"][:10])
        print("probs shape:", output["probs"].shape)

    # 必要ならファイルに保存
    # torch.save(output, "/content/output_region_feats.pth")
