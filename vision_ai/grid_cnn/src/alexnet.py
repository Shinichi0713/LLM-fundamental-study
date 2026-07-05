import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    """
    AlexNetのPyTorch実装
    入力: (batch_size, 3, 224, 224)
    出力: (batch_size, num_classes)
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # 第1層
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第2層
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第3層
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第4層
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第5層
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # 特徴抽出
        x = self.features(x)

        # フラット化
        x = x.view(x.size(0), -1)

        # 分類
        x = self.classifier(x)
        return x
    
import torchvision.models as models

# 学習済み重み付きAlexNet
model = models.alexnet(pretrained=True)

# 出力クラス数を変更する場合
model.classifier[6] = nn.Linear(4096, new_num_classes)

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    ResNetの基本ブロック（2層の畳み込み＋ショートカット接続）
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ショートカット接続（入力と出力のチャネル数・空間サイズが異なる場合）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.expansion * out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # ショートカット接続を加算
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNetの実装（ResNet-18相当）
    入力: (batch_size, 3, 224, 224)
    出力: (batch_size, num_classes)
    """
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # 最初の層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差ブロック群
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 全結合層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 入力: (B, 3, 224, 224)
        x = F.relu(self.bn1(self.conv1(x)))  # -> (B, 64, 112, 112)
        x = self.maxpool(x)                  # -> (B, 64, 56, 56)

        x = self.layer1(x)  # -> (B, 64, 56, 56)
        x = self.layer2(x)  # -> (B, 128, 28, 28)
        x = self.layer3(x)  # -> (B, 256, 14, 14)
        x = self.layer4(x)  # -> (B, 512, 7, 7)

        x = self.avgpool(x)  # -> (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # -> (B, 512)
        x = self.fc(x)             # -> (B, num_classes)
        return x


def ResNet18(num_classes=1000):
    """
    ResNet-18を返すヘルパー関数
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)