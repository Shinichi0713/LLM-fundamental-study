# セル 2: AMBER データのダウンロードと解凍
import os
import requests
import zipfile

# AMBER リポジトリから data.zip を取得（例）
# 実際にはリポジトリの data/ ディレクトリを直接クローンする方が確実です
!git clone https://github.com/junyangwang0410/AMBER.git
amber_dir = "/content/AMBER"

# 画像ディレクトリとアノテーションのパス
image_dir = os.path.join(amber_dir, "data", "image")
annotation_path = os.path.join(amber_dir, "data", "query_discriminative.json")

print("Image dir:", image_dir)
print("Annotation file:", annotation_path)

# セル 1: パス設定
import os
import json


amber_dir = "/content/AMBER"
image_dir = os.path.join(amber_dir, "data", "image")

# 判別タスク用クエリファイル（例）
annotation_path = os.path.join(amber_dir, "data", "query", "query_discriminative.json")

print("Image dir:", image_dir)
print("Annotation file:", annotation_path)


with open(annotation_path, "r", encoding="utf-8") as f:
    annotations = json.load(f)

print(f"Total samples: {len(annotations)}")
print("Example annotation:")
print(json.dumps(annotations[0], indent=2, ensure_ascii=False))


# セル 4: BLIP-2 モデルのロード
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16
).to(device)

