import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # スキップ接続（チャネル数や解像度が変わる場合の調整）
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        out = F.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class AdvancedGridRegressionCNN(nn.Module):
    """
    5x5グリッド画像から黒マスの座標 (row, col) を回帰予測する
    より複雑なCNNモデル（Residualブロック + BatchNorm + Dropout + Attention）
    """
    def __init__(self, grid_size=5, dropout_prob=0.2):
        super(AdvancedGridRegressionCNN, self).__init__()
        self.grid_size = grid_size
        
        # 初期畳み込み層
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residualブロック群
        self.layer1 = self._make_layer(32, 64, 2, stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout_prob=dropout_prob)
        
        # Channel Attention
        self.attention = ChannelAttention(256, reduction=16)
        
        # Global Average Pooling + 全結合層
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout_fc = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(128, 2)  # (row, col) の2次元

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout_prob):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_prob))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初期畳み込み
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Residualブロック群
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Channel Attention
        x = self.attention(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # 全結合層
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        # 出力層: sigmoidで[0,1]に収め、(grid_size-1)を掛けて[0,4]に変換
        coords = torch.sigmoid(self.fc2(x)) * (self.grid_size - 1)
        
        return coords

# モデルインスタンス化
model = AdvancedGridRegressionCNN(grid_size=5, dropout_prob=0.2)

# ダミー入力で形状確認
dummy_input = torch.randn(1, 1, 320, 320)  # (batch, channel, height, width)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # (1, 2)
print(f"Output values: {output}")       # [row_pred, col_pred] の近似値

# モデル・損失関数・オプティマイザ
model = AdvancedGridRegressionCNN(grid_size=5, dropout_prob=0.2).to(device)
criterion = nn.MSELoss()  # または nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
