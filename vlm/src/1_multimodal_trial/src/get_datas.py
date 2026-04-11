import os
from PIL import Image

# 画像ディレクトリ
image_dir = "Flickr8k_Dataset/"
# キャプションファイル（例：Flickr8k.token.txt）
caption_file = "Flickr8k_text/Flickr8k.token.txt"

# キャプションファイルを読み込み
captions = {}
with open(caption_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        img_id = parts[0].split("#")[0]  # 例: 1000268201_693b08cb0e.jpg#0
        caption = parts[1]
        if img_id not in captions:
            captions[img_id] = []
        captions[img_id].append(caption)

# 画像＋キャプションのペアを取得
for img_id, cap_list in list(captions.items())[:5]:  # 最初の5件を表示
    img_path = os.path.join(image_dir, img_id)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        print(f"Image: {img_id}")
        print("Captions:", cap_list)
        print("-" * 40)