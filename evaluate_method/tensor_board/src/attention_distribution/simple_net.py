import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 1. ログの保存先を指定
writer = SummaryWriter('runs/parameter_experiment')

# 2. シンプルなモデルを定義
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 1)
        
        # 【実験ポイント】あえて重みを大きな値で初期化してみる
        # これにより、学習が進むにつれて分布が「縮小」していく様子が見えます
        nn.init.constant_(self.fc1.weight, 1.0) 
        nn.init.constant_(self.fc2.weight, 1.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 3. ダミーデータでの学習ループ
for epoch in range(100):
    # ダミーの入力と正解（y = sum(x) のような単純なタスク）
    inputs = torch.randn(64, 10)
    targets = torch.sum(inputs, dim=1, keepdim=True)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # --- パラメータをTensorBoardに記録 ---
    for name, param in model.named_parameters():
        # ヒストグラムを記録（これがTensorBoardで見れる！）
        writer.add_histogram(name, param, epoch)
        # 勾配の分布も記録（消失/爆発のチェックに便利）
        if param.grad is not None:
            writer.add_histogram(f"{name}.grad", param.grad, epoch)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

writer.close()
print("学習完了。TensorBoardを起動して確認してください。")