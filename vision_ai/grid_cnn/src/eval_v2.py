import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
import os

# 画像サイズとグリッド設定（既存コードと合わせる）
GRID_SIZE = 5
CELL_SIZE = 64
IMG_SIZE = GRID_SIZE * CELL_SIZE
LINE_WIDTH = 2

# 画像変換（学習時と同じ変換を適用）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデルの読み込み（例：SimpleGridCNN）
# model = SimpleGridCNN().to(device)
# model.load_state_dict(torch.load("model.pth"))  # 学習済みモデルを読み込む場合
model.eval()


def generate_single_black_cell_image_and_label():
    """
    5x5グリッドからランダムに1マスだけ黒く塗った画像を生成し、
    そのマスの座標（行, 列）を返す（グリッド線なし）
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

    # グリッド線は描画しない（draw_grid_linesを呼び出さない）
    return img, (row, col)


def predict_black_cell(model, image, transform, device):
    """
    1枚の画像から黒マスの位置を予測する
    """
    model.eval()
    image_tensor = transform(image).unsqueeze(0).to(device)  # バッチ次元を追加

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_id = predicted.item()

    # class_id -> (row, col)
    row_pred = class_id // GRID_SIZE
    col_pred = class_id % GRID_SIZE

    return row_pred, col_pred


def run_inference_on_random_images(num_trials=5):
    """
    ランダムに生成した画像（グリッド線なし）に対して推論を行い、
    正解とモデルの予測を表示する
    """
    print(f"=== ランダム画像に対する推論結果（{num_trials}回）===")

    for i in range(num_trials):
        # 1. ランダムに1マスだけ黒く塗った画像を生成（グリッド線なし）
        img, (row_true, col_true) = generate_single_black_cell_image_and_label()

        # 2. モデルで予測
        row_pred, col_pred = predict_black_cell(model, img, transform, device)

        # 3. 結果を表示
        print(f"試行 {i+1}:")
        print(f"  正解: 黒マス = ({row_true}, {col_true})")
        print(f"  予測: 黒マス = ({row_pred}, {col_pred})")
        print(f"  一致: {'○' if (row_true, col_true) == (row_pred, col_pred) else '×'}")
        print()


if __name__ == "__main__":
    # 5回の推論を実行
    run_inference_on_random_images(num_trials=5)