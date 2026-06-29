import torch
import torch.nn as nn
import torch.nn.functional as F

class GridRegressionCNN(nn.Module):
    """
    5x5グリッド画像から黒マスの座標 (row, col) を回帰予測するCNN
    """
    def __init__(self, grid_size=5):
        super(GridRegressionCNN, self).__init__()
        self.grid_size = grid_size
        
        # CNN部分（特徴抽出）
        self.conv_layers = nn.Sequential(
            # 入力: (1, 320, 320) ※5x64=320
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),  # (16, 160, 160)
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),  # (32, 80, 80)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),  # (64, 40, 40)
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),  # (128, 20, 20)
        )
        
        # 全結合層（回帰用）
        # 入力: 128 * 20 * 20 = 51200
        self.fc1 = nn.Linear(128 * 20 * 20, 128)
        # 出力層: (row, col) の2次元
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # CNNで特徴抽出
        x = self.conv_layers(x)
        
        # フラット化
        x = x.view(x.size(0), -1)
        
        # 全結合層
        x = F.relu(self.fc1(x))
        
        # 出力層: sigmoidで[0,1]に収め、(grid_size-1)を掛けて[0,4]に変換
        coords = torch.sigmoid(self.fc2(x)) * (self.grid_size - 1)
        
        return coords