import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# シード固定
torch.manual_seed(42)

class SimpleSSM(nn.Module):
    def __init__(self, state_dim=1):
        super().__init__()
        self.state_dim = state_dim
        # パラメータ（固定値）
        self.A = nn.Parameter(torch.tensor([0.9]))  # 状態の減衰率
        self.B = nn.Parameter(torch.tensor([1.0]))  # 入力の重み
        self.C = nn.Parameter(torch.tensor([1.0]))  # 状態から出力への写像

    def forward(self, x):
        # x: (batch, seq_len)
        batch, seq_len = x.shape
        h = torch.zeros(batch, self.state_dim)  # 初期状態
        outputs = []
        states = []

        for t in range(seq_len):
            # 状態更新: h_t = A * h_{t-1} + B * x_t
            h = self.A * h + self.B * x[:, t:t+1]
            # 出力: y_t = C * h_t
            y = self.C * h
            outputs.append(y)
            states.append(h)

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, 1)
        states = torch.cat(states, dim=1)    # (batch, seq_len, 1)
        return outputs, states

# データ生成: 正弦波 + ノイズ
def generate_data(seq_len=100):
    t = torch.linspace(0, 4 * torch.pi, seq_len)
    x = torch.sin(t) + 0.1 * torch.randn(seq_len)  # ノイズ付き正弦波
    return x.unsqueeze(0)  # (1, seq_len)

# モデルとデータ
model = SimpleSSM(state_dim=1)
x = generate_data(seq_len=50)

# 順伝播
with torch.no_grad():
    y_pred, states = model(x)

# 可視化
plt.figure(figsize=(12, 4))

# 入力と出力の比較
plt.subplot(1, 2, 1)
plt.plot(x.squeeze().numpy(), label='入力 x(t)')
plt.plot(y_pred.squeeze().numpy(), label='出力 y(t)')
plt.title('SSM の入力と出力')
plt.legend()

# 状態の推移
plt.subplot(1, 2, 2)
plt.plot(states.squeeze().numpy(), label='状態 h(t)', color='red')
plt.title('SSM の内部状態')
plt.legend()

plt.tight_layout()
plt.show()