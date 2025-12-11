import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# データ生成：複合サイン波（LSTMが得意なパターン）
# ---------------------------------------------------------
np.random.seed(0)
t = np.linspace(0, 50, 1500)

y = (
    0.6 * np.sin(0.5 * t) +   # 長周期
    0.3 * np.sin(2.0 * t) +   # 中周期
    0.1 * np.sin(5.0 * t)     # 短周期
)

# ノイズは少なめに
y += 0.02 * np.random.randn(len(y))

# スケール変換
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1,1))


# ---------------------------------------------------------
# 時系列データ作成用関数
# ---------------------------------------------------------
def create_dataset(data, time_steps=50):
    X, Y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:i+time_steps])
        Y.append(data[i+time_steps])
    return np.array(X), np.array(Y)


# 時系列を作成
time_steps = 50
X, Y = create_dataset(y_scaled, time_steps)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# 80% を学習
train_size = int(len(X_tensor) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
Y_train, Y_test = Y_tensor[:train_size], Y_tensor[train_size:]


# ---------------------------------------------------------
# LSTM モデル
# ---------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# モデル生成
model = LSTMModel(hidden_size=64)

# ---------------------------------------------------------
# 학습 설정
# ---------------------------------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

train_losses = []
test_losses = []

# ---------------------------------------------------------
# 学習ループ
# ---------------------------------------------------------
epochs = 50

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, Y_train)
    loss.backward()
    optimizer.step()

    # テスト
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, Y_test).item()

    train_losses.append(loss.item())
    test_losses.append(test_loss)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.6f}, Test Loss: {test_loss:.6f}")


# ---------------------------------------------------------
# 推論
# ---------------------------------------------------------
model.eval()
with torch.no_grad():
    pred = model(X_test).numpy()

pred_inv = scaler.inverse_transform(pred)
true_inv = scaler.inverse_transform(Y_test.numpy())


# ---------------------------------------------------------
# 可視化
# ---------------------------------------------------------
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.title("Loss Curve (Train/Test)")
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()

plt.subplot(1,2,2)
plt.title("Prediction (LSTM)")
plt.plot(true_inv, label="True Signal")
plt.plot(pred_inv, label="LSTM Predict")
plt.legend()

plt.show()
