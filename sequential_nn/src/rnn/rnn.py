import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================
# 1. データ作成（正弦波）
# ============================
# sin 波データ
x = np.linspace(0, 200, 2000)
y = np.sin(x * 0.05)    # 周期をゆっくりに

# 時系列データを作る関数
def make_dataset(y, seq_len=30):
    xs, ys = [], []
    for i in range(len(y)-seq_len):
        xs.append(y[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(xs), np.array(ys)

SEQ_LEN = 30
X, Y = make_dataset(y, SEQ_LEN)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (batch, seq, 1)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

# 訓練用・テスト用に分割
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]


# ============================
# 2. RNN モデル定義
# ============================
class RNNPredictor(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, h = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 最後の時刻の隠れ状態から予測
        return out

model = RNNPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# ============================
# 3. 学習
# ============================
epochs = 50
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    
    pred = model(X_train)
    loss = criterion(pred, Y_train)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# ============================
# 4. 学習曲線を可視化
# ============================
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("RNN 学習曲線")
plt.show()


# ============================
# 5. テストデータで未来予測
# ============================
with torch.no_grad():
    pred_test = model(X_test).squeeze().numpy()

y_true = Y_test.squeeze().numpy()

plt.figure(figsize=(12,5))
plt.plot(y_true, label="True")
plt.plot(pred_test, label="RNN Prediction")
plt.title("RNNによる未来予測（時系列を学習できた例）")
plt.legend()
plt.show()