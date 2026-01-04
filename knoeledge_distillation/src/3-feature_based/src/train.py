import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 環境設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

# --- 2. モデル定義 ---
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc1(x))
        logits = self.fc2(feat)
        return logits, feat

class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        feat = F.relu(self.fc1(x))
        logits = self.fc2(feat)
        return logits, feat

# --- 3. 学習・評価用関数 ---
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

def train_process(teacher, student, mode='none', epochs=5):
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    regressor = None
    if mode == 'feature':
        regressor = nn.Linear(128, 512).to(device)
        optimizer = optim.Adam(list(student.parameters()) + list(regressor.parameters()), lr=0.001)

    history = []
    teacher.eval()
    
    print(f"\nStarting Training: Mode={mode}")
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
                # 温度T=3でSoftmaxをなだらかにしてKL散布図で蒸留
                T = 3.0
                soft_s = F.log_softmax(s_logits/T, dim=1)
                soft_t = F.softmax(t_logits/T, dim=1)
                loss_distill = F.kl_div(soft_s, soft_t, reduction='batchmean') * (T**2)
                loss = 0.5 * loss_cls + 0.5 * loss_distill
            elif mode == 'feature':
                # 中間層のMSE誤差
                loss_distill = F.mse_loss(regressor(s_feat), t_feat)
                loss = 0.5 * loss_cls + 0.5 * loss_distill
            else:
                loss = loss_cls

            loss.backward()
            optimizer.step()
        
        acc = test(student)
        history.append(acc)
        print(f"Epoch {epoch}: Test Accuracy = {acc:.2f}%")
    return history

# --- 4. 実行セクション ---

# A. 教師モデルの学習（3エポックで十分高精度になります）
print("--- Training Teacher Model ---")
teacher = TeacherModel().to(device)
t_optimizer = optim.Adam(teacher.parameters(), lr=0.001)
for epoch in range(1, 4):
    teacher.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        t_optimizer.zero_grad()
        logits, _ = teacher(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        t_optimizer.step()
    print(f"Teacher Epoch {epoch}: Accuracy = {test(teacher):.2f}%")

# B. 各モードで生徒モデルを学習
epochs = 5
history_none = train_process(teacher, StudentModel().to(device), mode='none', epochs=epochs)
history_output = train_process(teacher, StudentModel().to(device), mode='output', epochs=epochs)
history_feature = train_process(teacher, StudentModel().to(device), mode='feature', epochs=epochs)

# --- 5. 可視化 ---

# 学習曲線の表示
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), history_none, label='Normal (Label Only)', marker='o')
plt.plot(range(1, epochs+1), history_output, label='Output Distillation', marker='s')
plt.plot(range(1, epochs+1), history_feature, label='Feature Distillation', marker='^')
plt.title('Comparison of Student Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# 推論結果の可視化
def visualize_predictions(model, title):
    model.eval()
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        logits, _ = model(data)
        preds = logits.argmax(dim=1)
    
    plt.figure(figsize=(12, 3))
    for i in range(6):
        plt.subplot(1, 6, i+1)
        plt.imshow(data[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title(f"Pred: {preds[i].item()}\n(True: {target[i].item()})")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

print("\n--- Inference Visualizations ---")
visualize_predictions(teacher, "Teacher Model (CNN)")
visualize_predictions(StudentModel().to(device), "Student Model (Untrained)") # 比較用