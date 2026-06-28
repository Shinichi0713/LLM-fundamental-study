import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class GridDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        annotations_file: CSV形式のannotationファイル
        img_dir: 画像が保存されているディレクトリ
        transform: 画像変換（正規化など）
        """
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_name = row["file_name"]
        img_path = os.path.join(self.img_dir, img_name)

        # 画像読み込み（グレースケール）
        image = Image.open(img_path).convert("L")

        # ラベル（黒マスの行・列）からクラスIDに変換
        # 例: (row, col) = (2, 3) -> class_id = 2*5 + 3 = 13
        row_idx = row["black_cell_row"]
        col_idx = row["black_cell_col"]
        class_id = row_idx * 5 + col_idx  # 5x5固定

        if self.transform:
            image = self.transform(image)

        return image, class_id


# 画像変換の定義（正規化など）
transform = transforms.Compose([
    transforms.ToTensor(),  # PIL Image -> Tensor (0-1)
    transforms.Normalize(mean=[0.5], std=[0.5])  # グレースケール用
])

