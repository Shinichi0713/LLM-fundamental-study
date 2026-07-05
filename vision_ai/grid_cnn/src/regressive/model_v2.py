import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGridCNNWithAttention(nn.Module):
    """
    5x5グリッド画像から黒マスの位置を予測するCNNモデル
    空間アテンション機構を追加し、可視化も可能
    """
    def __init__(self, grid_size=5, img_size=320):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = grid_size * grid_size  # 25クラス

        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # 全結合層の入力サイズを計算
        self.fc_input_size = self._get_fc_input_size(img_size)

        # 全結合層
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

        # 空間アテンション用の層
        # 特徴マップのチャネル数を1に圧縮し、空間ごとの重要度を出す
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),  # 出力: (batch, 1, H, W)
            nn.Sigmoid()  # 0〜1の重み
        )

    def _get_fc_input_size(self, img_size):
        x = torch.zeros(1, 1, img_size, img_size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x, return_attention=False):
        """
        x: (batch_size, 1, img_size, img_size)
        return_attention: Trueならアテンション重みも返す
        """
        # 特徴抽出
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # (B, 32, H/2, W/2)

        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (B, 64, H/4, W/4)

        # 空間アテンションの計算
        attention_weights = self.spatial_attention(x)  # (B, 1, H/4, W/4)

        # アテンション適用（重み付き特徴マップ）
        attended_x = x * attention_weights  # 要素ごとの積

        # フラット化
        x_flat = attended_x.view(attended_x.size(0), -1)

        # 全結合層
        x_flat = F.relu(self.fc1(x_flat))
        logits = self.fc2(x_flat)

        if return_attention:
            # アテンション重みを(バッチ, H, W)に整形して返す
            att = attention_weights.squeeze(1)  # (B, H, W)
            return logits, att
        else:
            return logits
        
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(model, image_tensor):
    """
    image_tensor: (1, 1, H, W) のテンソル
    """
    model.eval()
    with torch.no_grad():
        logits, att_map = model(image_tensor, return_attention=True)

    # テンソル → numpy
    img_np = image_tensor.squeeze().cpu().numpy()  # (H, W)
    att_np = att_map.squeeze().cpu().numpy()       # (H_att, W_att)

    # 予測クラス（グリッド位置）
    pred_class = logits.argmax(dim=1).item()
    pred_row = pred_class // model.grid_size
    pred_col = pred_class % model.grid_size

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 元画像
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title(f'Input image')
    axes[0].axis('off')

    # アテンションマップ（ヒートマップ）
    im = axes[1].imshow(att_np, cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Attention map\nPred: ({pred_row}, {pred_col})')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(model, image_tensor):
    """
    image_tensor: (1, 1, H, W) のテンソル
    """
    model.eval()
    with torch.no_grad():
        logits, att_map = model(image_tensor, return_attention=True)

    # テンソル → numpy
    img_np = image_tensor.squeeze().cpu().numpy()  # (H, W)
    att_np = att_map.squeeze().cpu().numpy()       # (H_att, W_att)

    # 予測クラス（グリッド位置）
    pred_class = logits.argmax(dim=1).item()
    pred_row = pred_class // model.grid_size
    pred_col = pred_class % model.grid_size

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 元画像
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title(f'Input image')
    axes[0].axis('off')

    # アテンションマップ（ヒートマップ）
    im = axes[1].imshow(att_np, cmap='hot', interpolation='nearest')
    # axes[1].set_title(f'Attention map\nPred: ({pred_row}, {pred_col})')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.show()

def visualize_heatmap(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert("L")
    image_tensor = transform(image).unsqueeze(0).to(device)  # バッチ次元を追加
    visualize_attention(model, image_tensor)

# 使用例
paths_image = glob.glob(os.path.join("/content/dataset", "*.png"))
selected_images = random.sample(paths_image, 5)

for selected_image in selected_images:
  visualize_heatmap(model, selected_image, transform, device)