import numpy as np
from PIL import Image
import random
import os

# 画像サイズとグリッド設定
GRID_SIZE = 5     # 5x5マス
CELL_SIZE = 64    # 1マスのピクセル数
IMG_SIZE = GRID_SIZE * CELL_SIZE  # 画像全体のサイズ

def generate_single_black_cell_image_and_label():
    """
    5x5グリッドからランダムに1マスだけ黒く塗った画像を生成し、
    そのマスの座標（行, 列）を返す
    """
    # 白いキャンバス（グレースケール画像）
    img_array = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)

    # ランダムに1マスを選ぶ（0〜4の範囲）
    row = random.randint(0, GRID_SIZE - 1)
    col = random.randint(0, GRID_SIZE - 1)

    # 選んだマスを黒く塗る
    y_start = row * CELL_SIZE
    y_end = (row + 1) * CELL_SIZE
    x_start = col * CELL_SIZE
    x_end = (col + 1) * CELL_SIZE

    img_array[y_start:y_end, x_start:x_end] = 0  # 黒=0

    # PIL Imageに変換
    img = Image.fromarray(img_array, mode='L')  # 'L' = 8bitグレースケール

    return img, (row, col)


def generate_dataset(num_images=100, output_dir="dataset"):
    """
    指定枚数分の画像とannotationファイルを生成する
    """
    os.makedirs(output_dir, exist_ok=True)

    annotations = []

    for i in range(num_images):
        img, (row, col) = generate_single_black_cell_image_and_label()

        # 画像保存
        img_path = os.path.join(output_dir, f"img_{i:04d}.png")
        img.save(img_path)

        # annotation情報を記録
        annotations.append({
            "image_id": i,
            "file_name": f"img_{i:04d}.png",
            "black_cell_row": row,
            "black_cell_col": col
        })

    # annotationをテキストファイルに保存（CSV形式の例）
    with open(os.path.join(output_dir, "annotations.txt"), "w") as f:
        f.write("image_id,file_name,black_cell_row,black_cell_col\n")
        for ann in annotations:
            f.write(f"{ann['image_id']},{ann['file_name']},{ann['black_cell_row']},{ann['black_cell_col']}\n")

    print(f"Generated {num_images} images and annotations in '{output_dir}/'")


import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
import matplotlib.pyplot as plt

# 画像サイズとグリッド設定
GRID_SIZE = 5
CELL_SIZE = 64
IMG_SIZE = GRID_SIZE * CELL_SIZE

def generate_single_black_cell_image_and_label():
    """
    5x5グリッドからランダムに1マスだけ黒く塗った画像を生成し、
    そのマスの座標（行, 列）を返す
    """
    img_array = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)

    row = random.randint(0, GRID_SIZE - 1)
    col = random.randint(0, GRID_SIZE - 1)

    y_start = row * CELL_SIZE
    y_end = (row + 1) * CELL_SIZE
    x_start = col * CELL_SIZE
    x_end = (col + 1) * CELL_SIZE

    img_array[y_start:y_end, x_start:x_end] = 0
    img = Image.fromarray(img_array, mode='L')
    return img, (row, col)


def generate_dataset(num_images=10, output_dir="dataset"):
    """
    指定枚数分の画像とannotationファイルを生成する
    """
    os.makedirs(output_dir, exist_ok=True)

    annotations = []

    for i in range(num_images):
        img, (row, col) = generate_single_black_cell_image_and_label()

        img_path = os.path.join(output_dir, f"img_{i:04d}.png")
        img.save(img_path)

        annotations.append({
            "image_id": i,
            "file_name": f"img_{i:04d}.png",
            "black_cell_row": row,
            "black_cell_col": col
        })

    # annotationをテキストファイルに保存
    with open(os.path.join(output_dir, "annotations.txt"), "w") as f:
        f.write("image_id,file_name,black_cell_row,black_cell_col\n")
        for ann in annotations:
            f.write(f"{ann['image_id']},{ann['file_name']},{ann['black_cell_row']},{ann['black_cell_col']}\n")

    print(f"Generated {num_images} images and annotations in '{output_dir}/'")
    return annotations


def display_samples(annotations, output_dir="dataset", num_samples=5):
    """
    生成したデータセットからランダムにサンプルを選び、
    画像とラベルを表示する
    """
    # ランダムにサンプルを選ぶ
    sampled_indices = random.sample(range(len(annotations)), min(num_samples, len(annotations)))

    # 画像を読み込んで表示
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    if num_samples == 1:
        axes = [axes]  # 1枚だけのときはリスト化

    for idx, ax in zip(sampled_indices, axes):
        ann = annotations[idx]
        img_path = os.path.join(output_dir, ann["file_name"])
        img = Image.open(img_path)

        ax.imshow(img, cmap="gray")
        ax.set_title(f"ID:{ann['image_id']}\nBlack cell: ({ann['black_cell_row']}, {ann['black_cell_col']})")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 100枚の画像とannotationを生成
    generate_dataset(num_images=100)

