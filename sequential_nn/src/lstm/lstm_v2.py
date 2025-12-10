import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ================================================
# 1. 正弦波データを生成
# ================================================
def generate_sin_wave(seq_length=3000):
    x = np.arange(seq_length)
    y = np.sin(0.02 * x) + 0.1*np.random.randn(seq_length)  # ノイズあり sin 波
    return y


data = generate_sin_wave()


# ================================================
# 2. Dataset の作成（過去 50 ステップ → 次の 1 ステップ予測）
# ================================================
class SinDataset(Dataset):
    def __init__(self, data, input_len=50):
        self.data = data
        self.input_len = input_len

    def __len__(self):
        return len(self.data) - self.input_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]
        y = self.data[idx + self.input_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

input_len = 50
dataset = SinDataset(data, input_len)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# ================================================
# 3. LSTM モデル定義
# ================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 最後のステップだけ使用
        return out


model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ================================================
# 4. 学習ループ
# ================================================
loss_list = []

for epoch in range(20):
    for x, y in loader:
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        pred = model(x)
        loss = criterion(pred.squeeze(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_list.append(loss.item())
    print(f"Epoch {epoch+1}/20  Loss: {loss.item():.5f}")


# --------------------------------------------
# 学習曲線の表示
# --------------------------------------------
plt.plot(loss_list)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()


# ================================================
# 5. 未来の波形を LSTM に予測させる
# ================================================
model.eval()

input_seq = torch.tensor(data[-input_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
predicted = []

for _ in range(200):  # 200 ステップ未来を予測
    with torch.no_grad():
        next_val = model(input_seq).item()
    predicted.append(next_val)

    # 入力シーケンスを更新（古い値を削除し、予測値を追加）
    input_seq = torch.cat(
        [input_seq[:, 1:, :], torch.tensor([[[next_val]]], dtype=torch.float32)],
        dim=1
    )


# ================================================
# 6. 結果の可視化
# ================================================
plt.figure(figsize=(12, 5))
plt.plot(data[-300:], label="Past (real data)")
plt.plot(range(300, 300+200), predicted, label="Predicted future")
plt.legend()
plt.title("LSTM Sin Wave Prediction")
plt.show()
