import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import random
import matplotlib.pyplot as plt

# ======================
# 1. モデル定義（再掲）
# ======================
class SimpleGridCNN(nn.Module):
    def __init__(self, grid_size=5, img_size=320):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = grid_size * grid_size

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.fc_input_size = self._get_fc_input_size(img_size)
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def _get_fc_input_size(self, img_size):
        x = torch.zeros(1, 1, img_size, img_size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ======================
# 2. 画像生成関数（再掲）
# ======================
GRID_SIZE = 5
CELL_SIZE = 64
IMG_SIZE = GRID_SIZE * CELL_SIZE
LINE_WIDTH = 2

def draw_grid_lines(img, grid_size=GRID_SIZE, cell_size=CELL_SIZE, line_width=LINE_WIDTH):
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    for col in range(1, grid_size):
        x = col * cell_size
        draw.line([(x, 0), (x, img_height)], fill=0, width=line_width)

    for row in range(1, grid_size):
        y = row * cell_size
        draw.line([(0, y), (img_width, y)], fill=0, width=line_width)

    draw.rectangle([0, 0, img_width - 1, img_height - 1], outline=0, width=line_width)
    return img

def generate_single_black_cell_image_and_label():
    img_array = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)

    row = random.randint(0, GRID_SIZE - 1)
    col = random.randint(0, GRID_SIZE - 1)

    y_start = row * CELL_SIZE
    y_end = (row + 1) * CELL_SIZE
    x_start = col * CELL_SIZE
    x_end = (col + 1) * CELL_SIZE

    img_array[y_start:y_end, x_start:x_end] = 0
    img = Image.fromarray(img_array, mode='L')
    img = draw_grid_lines(img)

    return img, (row, col)


# ======================
# 3. 変換とデバイス設定
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ======================
# 4. モデル初期化（未学習状態）
# ======================
model = SimpleGridCNN(grid_size=GRID_SIZE, img_size=IMG_SIZE).to(device)
model.eval()


# ======================
# 5. 入出力の確認＋可視化
# ======================
def visualize_input_output():
    """
    1枚のランダム画像を生成し、モデルへの入力と出力を表示・可視化する
    """
    # 1. ランダムに1マスだけ黒く塗った画像を生成
    img_pil, (row_true, col_true) = generate_single_black_cell_image_and_label()

    # 2. モデルへの入力（テンソル）を確認
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # (1, 1, H, W)
    print("=== モデルへの入力（テンソル）===")
    print(f"形状: {img_tensor.shape}")  # torch.Size([1, 1, 320, 320])
    print(f"最小値: {img_tensor.min().item():.3f}")
    print(f"最大値: {img_tensor.max().item():.3f}")
    print(f"平均値: {img_tensor.mean().item():.3f}")
    print()

    # 3. モデルの出力（ロジット）を確認
    with torch.no_grad():
        outputs = model(img_tensor)  # (1, 25)

    print("=== モデルの出力（ロジット）===")
    print(f"形状: {outputs.shape}")  # torch.Size([1, 25])
    print("ロジットの先頭5要素（例）:")
    for i in range(5):
        print(f"  class_{i}: {outputs[0, i].item():.3f}")
    print("...（中略）")
    print()

    # 4. 予測クラスIDと座標の計算
    _, predicted_class = torch.max(outputs, 1)
    class_id = predicted_class.item()
    row_pred = class_id // GRID_SIZE
    col_pred = class_id % GRID_SIZE

    print("=== 予測結果 ===")
    print(f"予測クラスID: {class_id}")
    print(f"予測座標: ({row_pred}, {col_pred})")
    print(f"正解座標: ({row_true}, {col_true})")
    print(f"一致: {'○' if (row_true, col_true) == (row_pred, col_pred) else '×'}")
    print()

    # 5. 可視化（画像＋正解＋予測）
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 左：元の画像
    axes[0].imshow(img_pil, cmap="gray")
    axes[0].set_title("Input Image (PIL)")
    axes[0].axis("off")

    # 右：テンソルを画像に戻したもの（参考）
    img_tensor_np = img_tensor.squeeze().cpu().numpy()  # (H, W)
    axes[1].imshow(img_tensor_np, cmap="gray")
    axes[1].set_title("Model Input (Tensor → numpy)")
    axes[1].axis("off")

    plt.suptitle(
        f"True: ({row_true}, {col_true}) | Pred: ({row_pred}, {col_pred}) | Match: {'✓' if (row_true, col_true) == (row_pred, col_pred) else '✗'}"
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_input_output()