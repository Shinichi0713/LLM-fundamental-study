import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# データ生成：正弦波
# ---------------------------------------------------------
np.random.seed(0)
t = np.linspace(0, 100, 500)
y = np.sin(t) + 0.1 * np.random.randn(len(t))   # ノイズ付き

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1,1))

# 時系列データ作成
def create_dataset(data, time_steps=20):
    X, Y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:i+time_steps])
        Y.append(data[i+time_steps])
    return np.array(X), np.array(Y)

time_steps = 20
X, Y = create_dataset(y_scaled, time_steps)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

train_size = int(len(X_tensor) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
Y_train, Y_test = Y_tensor[:train_size], Y_tensor[train_size:]

# ---------------------------------------------------------
# モデル定義（RNN / LSTM）
# ---------------------------------------------------------
class RNNModel(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ---------------------------------------------------------
# 学習関数
# ---------------------------------------------------------
def train_model(model, X_train, Y_train, X_test, Y_test, epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        test_out = model(X_test)
        test_loss = criterion(test_out, Y_test).item()

        train_losses.append(loss.item())
        test_losses.append(test_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

    return train_losses, test_losses


# ---------------------------------------------------------
# モデル作成 & 学習
# ---------------------------------------------------------
rnn = RNNModel()
lstm = LSTMModel()

print("==== RNN 学習 ====")
rnn_train_loss, rnn_test_loss = train_model(rnn, X_train, Y_train, X_test, Y_test)

print("\n==== LSTM 学習 ====")
lstm_train_loss, lstm_test_loss = train_model(lstm, X_train, Y_train, X_test, Y_test)


# ---------------------------------------------------------
# 予測
# ---------------------------------------------------------
rnn.eval()
lstm.eval()

rnn_pred = rnn(X_test).detach().numpy()
lstm_pred = lstm(X_test).detach().numpy()

# 逆正規化
rnn_pred_inv = scaler.inverse_transform(rnn_pred)
lstm_pred_inv = scaler.inverse_transform(lstm_pred)
Y_test_inv = scaler.inverse_transform(Y_test.numpy())

# ---------------------------------------------------------
# 結果可視化
# ---------------------------------------------------------

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.title("Loss Curve (RNN vs LSTM)")
plt.plot(rnn_test_loss, label="RNN Test Loss")
plt.plot(lstm_test_loss, label="LSTM Test Loss")
plt.legend()

plt.subplot(1,2,2)
plt.title("Prediction Comparison")
plt.plot(Y_test_inv, label="True")
plt.plot(rnn_pred_inv, label="RNN Predict")
plt.plot(lstm_pred_inv, label="LSTM Predict")
plt.legend()

plt.show()
