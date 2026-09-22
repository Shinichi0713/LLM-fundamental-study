import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. Stochastic Depth (DropPath) モジュールの定義
# ---------------------------------------------------------
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # 評価（推論）時、または drop_prob=0 の場合はそのまま通す
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        # バッチごとのドロップマスク作成 (Shape: [Batch_size, 1, 1, 1])
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor) # 0 または 1

        # スキップされた層のスケール補正 (1 / keep_prob)
        output = x.div(keep_prob) * binary_tensor
        return output

# ---------------------------------------------------------
# 2. ResNet Block（Stochastic Depth組み込み）
# ---------------------------------------------------------
class BasicBlockWithSD(nn.Module):
    def __init__(self, in_planes, planes, stride=1, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.sd = StochasticDepth(drop_prob)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 変換パス（Convブロック）に対してStochastic Depthを適用し、残差接続と足し合わせる
        out = self.shortcut(x) + self.sd(out)
        out = self.relu(out)
        return out

# ---------------------------------------------------------
# 3. リニアスケール（Linear Decay）付き Stochastic Depth ResNet
# ---------------------------------------------------------
class ResNetForSD(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, max_drop_prob=0.2):
        super().__init__()
        self.in_planes = 16
        total_blocks = sum(num_blocks)
        block_idx = 0

        # 線形にドロップ確率を増加（浅い層は低確率、深い層は高確率でドロップ）
        get_drop_prob = lambda idx: (idx / total_blocks) * max_drop_prob

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1, block_idx = self._make_layer(block, 16, num_blocks[0], stride=1, block_idx=block_idx, get_drop_prob=get_drop_prob)
        self.layer2, block_idx = self._make_layer(block, 32, num_blocks[1], stride=2, block_idx=block_idx, get_drop_prob=get_drop_prob)
        self.layer3, block_idx = self._make_layer(block, 64, num_blocks[2], stride=2, block_idx=block_idx, get_drop_prob=get_drop_prob)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, block_idx, get_drop_prob):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            drop_prob = get_drop_prob(block_idx)
            layers.append(block(self.in_planes, planes, s, drop_prob=drop_prob))
            self.in_planes = planes
            block_idx += 1
        return nn.Sequential(*layers), block_idx

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# ---------------------------------------------------------
# 4. データセット（CIFAR-100）準備
# ---------------------------------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# ---------------------------------------------------------
# 5. 学習用ループ関数の定義
# ---------------------------------------------------------
def train_model(model, epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        train_accs.append(train_acc)

        # 評価フェーズ
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    return train_accs, test_accs

# ---------------------------------------------------------
# 6. 比較実験の実行（Standard ResNet vs Stochastic Depth ResNet）
# ---------------------------------------------------------
# ResNet-32 (各グループ 5 ブロック = 計 30 残差ブロック)
num_blocks = [5, 5, 5]

print("=== 1. Standard ResNet (Drop Prob = 0.0) の学習 ===")
model_standard = ResNetForSD(BasicBlockWithSD, num_blocks, max_drop_prob=0.0).to(device)
std_train_acc, std_test_acc = train_model(model_standard, epochs=15)

print("\n=== 2. Stochastic Depth ResNet (Max Drop Prob = 0.3) の学習 ===")
model_sd = ResNetForSD(BasicBlockWithSD, num_blocks, max_drop_prob=0.3).to(device)
sd_train_acc, sd_test_acc = train_model(model_sd, epochs=15)

# ---------------------------------------------------------
# 7. グラフ可視化
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
epochs_range = range(1, 16)

plt.plot(epochs_range, std_train_acc, 'b--', label='Standard Train Acc')
plt.plot(epochs_range, std_test_acc, 'b-', label='Standard Test Acc')

plt.plot(epochs_range, sd_train_acc, 'r--', label='Stochastic Depth Train Acc')
plt.plot(epochs_range, sd_test_acc, 'r-', label='Stochastic Depth Test Acc')

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Effect of Stochastic Depth on Overfitting & Generalization')
plt.legend()
plt.grid(True)
plt.show()

