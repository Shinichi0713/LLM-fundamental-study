# 画像サイズとグリッド設定
GRID_SIZE = 5
CELL_SIZE = 64
IMG_SIZE = GRID_SIZE * CELL_SIZE
LINE_WIDTH = 2

def draw_grid_lines(img, grid_size=GRID_SIZE, cell_size=CELL_SIZE, line_width=LINE_WIDTH):
    """
    画像にグリッド線（黒枠）を描画する
    """
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    # 縦線（列の境界）
    for col in range(1, grid_size):
        x = col * cell_size
        draw.line([(x, 0), (x, img_height)], fill=0, width=line_width)

    # 横線（行の境界）
    for row in range(1, grid_size):
        y = row * cell_size
        draw.line([(0, y), (img_width, y)], fill=0, width=line_width)

    # 外枠
    draw.rectangle([0, 0, img_width - 1, img_height - 1], outline=0, width=line_width)
    return img

def generate_single_black_cell_image_and_label():
    """
    5x5グリッドからランダムに1マスだけ黒く塗った画像を生成し、
    そのマスの座標（行, 列）を返す（回帰用：ラベルは (row, col) の浮動小数点数）
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

    # グリッド線を描画（任意：必要に応じてコメントアウト）
    img = draw_grid_lines(img)

    # ラベルは (row, col) の浮動小数点数（回帰用）
    label = (float(row), float(col))

    return img, label

def generate_regression_dataset(num_images=100, output_dir="regression_dataset"):
    """
    指定枚数分の画像とannotationファイルを生成する（回帰用）
    - 画像：5x5グリッドに1マスだけ黒く塗った画像
    - ラベル：黒マスの座標 (row, col) を浮動小数点数で保存
    """
    os.makedirs(output_dir, exist_ok=True)

    annotations = []

    for i in range(num_images):
        # 1枚の画像とラベルを生成
        img, (row, col) = generate_single_black_cell_image_and_label()

        # 画像保存
        img_path = os.path.join(output_dir, f"img_{i:04d}.png")
        img.save(img_path)

        # annotation情報を記録（回帰用：row, col を浮動小数点数で保存）
        annotations.append({
            "image_id": i,
            "file_name": f"img_{i:04d}.png",
            "black_cell_row": row,
            "black_cell_col": col
        })

    # annotationをテキストファイルに保存（CSV形式）
    with open(os.path.join(output_dir, "annotations.txt"), "w") as f:
        f.write("image_id,file_name,black_cell_row,black_cell_col\n")
        for ann in annotations:
            f.write(f"{ann['image_id']},{ann['file_name']},{ann['black_cell_row']},{ann['black_cell_col']}\n")

    print(f"Generated {num_images} regression images and annotations in '{output_dir}/'")

import torch
from torch.utils.data import Dataset

class GridRegressionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = []

        # annotationファイルを読み込み
        with open(annotations_file, "r") as f:
            lines = f.readlines()[1:]  # ヘッダーをスキップ
            for line in lines:
                parts = line.strip().split(",")
                image_id = int(parts[0])
                file_name = parts[1]
                black_cell_row = float(parts[2])
                black_cell_col = float(parts[3])
                self.annotations.append({
                    "image_id": image_id,
                    "file_name": file_name,
                    "black_cell_row": black_cell_row,
                    "black_cell_col": black_cell_col
                })

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 画像読み込み
        img_path = os.path.join(self.img_dir, self.annotations[idx]["file_name"])
        image = Image.open(img_path).convert("L")

        # ラベル取得（回帰用：2次元ベクトル）
        row = self.annotations[idx]["black_cell_row"]
        col = self.annotations[idx]["black_cell_col"]
        label = torch.tensor([row, col], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

def check_dataset_samples(num_samples=5):
  """
  ランダムに生成した画像とその座標を表示し、データセットの入出力をチェックする
  """
  print(f"=== データセットの入出力チェック（{num_samples}サンプル）===")

  fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
  if num_samples == 1:
      axes = [axes]  # 1サンプルの場合の処理

  for i in range(num_samples):
      # 1枚の画像とラベルを生成
      img, (row, col) = generate_single_black_cell_image_and_label()

      # 画像を表示
      axes[i].imshow(img, cmap='gray')
      axes[i].set_title(f"Sample {i+1}\nBlack cell: ({int(row)}, {int(col)})")
      axes[i].axis('off')

      # コンソールにも情報を出力
      print(f"サンプル {i+1}:")
      print(f"  画像サイズ: {img.size} (幅 x 高さ)")
      print(f"  黒マスの座標: ({row}, {col})")
      print(f"  対応するクラスID（参考）: {int(row) * GRID_SIZE + int(col)}")
      print()

  plt.tight_layout()
  plt.show()