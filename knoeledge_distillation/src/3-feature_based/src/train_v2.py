import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

# ===============================
# 1. 再現性・環境設定
# ===============================
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=1000
)

# ===============================
# 2. モデル定義
# ===============================

# ---- Teacher (やや厚めCNN) ----
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc1(x))
        logits = self.fc2(feat)
        return logits, feat


# ---- Student (Convベースで表現を近づける) ----
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc1(x))
        logits = self.fc2(feat)
        return logits, feat


# ===============================
# 3. 評価関数
# ===============================
def test(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, _ = model(data)
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100. * correct / len(test_loader.dataset)


# ===============================
# 4. 学習ループ（蒸留切替）
# ===============================
def train_process(teacher, student, mode='none', epochs=5, alpha=0.7):
    student.to(device)
    teacher.eval()

    if mode == 'feature':
        regressor = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512)
        ).to(device)
        optimizer = optim.Adam(
            list(student.parameters()) + list(regressor.parameters()), lr=1e-3
        )
    else:
        regressor = None
        optimizer = optim.Adam(student.parameters(), lr=1e-3)

    history = []
    print(f"\n=== Training Mode: {mode} ===")

    for epoch in range(1, epochs + 1):
        student.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            s_logits, s_feat = student(data)
            with torch.no_grad():
                t_logits, t_feat = teacher(data)

            loss_cls = F.cross_entropy(s_logits, target)

            if mode == 'output':
                T = 3.0
                loss_distill = F.kl_div(
                    F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T ** 2)
                loss = alpha * loss_cls + (1 - alpha) * loss_distill

            elif mode == 'feature':
                # ---- Feature正規化付き蒸留 ----
                s_proj = F.normalize(regressor(s_feat), dim=1)
                t_norm = F.normalize(t_feat, dim=1)
                loss_distill = F.mse_loss(s_proj, t_norm)
                loss = alpha * loss_cls + (1 - alpha) * loss_distill

            else:
                loss = loss_cls

            loss.backward()
            optimizer.step()

        acc = test(student)
        history.append(acc)
        print(f"Epoch {epoch}: Test Accuracy = {acc:.2f}%")

    return history


# ===============================
# 5. 実行
# ===============================

# ---- 教師モデル学習 ----
print("\n--- Training Teacher Model ---")
teacher = TeacherModel().to(device)
t_optimizer = optim.Adam(teacher.parameters(), lr=1e-3)

for epoch in range(1, 11):
    teacher.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        t_optimizer.zero_grad()
        logits, _ = teacher(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        t_optimizer.step()

    print(f"Teacher Epoch {epoch}: Accuracy = {test(teacher):.2f}%")

# ---- Student比較 ----
epochs = 5
history_none = train_process(teacher, StudentModel(), mode='none', epochs=epochs)
history_output = train_process(teacher, StudentModel(), mode='output', epochs=epochs)
history_feature = train_process(teacher, StudentModel(), mode='feature', epochs=epochs)


# ===============================
# 6. 可視化
# ===============================
plt.figure(figsize=(10, 5))
plt.plot(history_none, marker='o', label='Label Only')
plt.plot(history_output, marker='s', label='Output Distillation')
plt.plot(history_feature, marker='^', label='Feature Distillation')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Knowledge Distillation Comparison")
plt.legend()
plt.grid(True)
plt.show()
