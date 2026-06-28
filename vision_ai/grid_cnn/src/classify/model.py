import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGridCNN(nn.Module):
    """
    5x5グリッド画像から黒マスの位置を予測するCNNモデル
    """
    def __init__(self, grid_size=5, img_size=320):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = grid_size * grid_size  # 25クラス

        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1ch -> 32ch
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32ch -> 64ch
        self.pool = nn.MaxPool2d(2)  # 1/2にダウンサンプリング

        # 全結合層の入力サイズを計算
        # 例: img_size=320 -> pool後 160 -> 80 -> 40
        # 実際には forward の途中で形状を確認してから決める
        self.fc_input_size = self._get_fc_input_size(img_size)

        # 全結合層
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def _get_fc_input_size(self, img_size):
        """
        全結合層の入力サイズを計算するためのダミーフォワード
        """
        x = torch.zeros(1, 1, img_size, img_size)  # バッチサイズ1のダミー入力
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.numel()  # 全要素数

    def forward(self, x):
        # x: (batch_size, 1, img_size, img_size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # フラット化
        x = x.view(x.size(0), -1)

        # 全結合層
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # ロジット出力（softmaxは損失関数側で）

        return x