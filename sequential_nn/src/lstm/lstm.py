# 必要ライブラリ
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# デバイス
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ======== データ生成：Adding Problem ========
# 入力: 長さ T の系列、各ステップは (value, marker)
# - value: U(0,1)
# - marker: 0, ただしランダムに2箇所だけ1になる
# 目的: marker==1 のときの value を足したスカラーを予測

def generate_adding_dataset(n_samples, seq_len=100):
    X = np.random.rand(n_samples, seq_len).astype(np.float32)  # values
    markers = np.zeros((n_samples, seq_len), dtype=np.float32)
    targets = np.zeros((n_samples,), dtype=np.float32)
    for i in range(n_samples):
        idx = np.random.choice(seq_len, size=2, replace=False)
        markers[i, idx] = 1.0
        targets[i] = X[i, idx].sum()
    # combine channels: shape (n, seq_len, 2)
    inputs = np.stack([X, markers], axis=-1)
    return torch.tensor(inputs), torch.tensor(targets).unsqueeze(1)  # target shape (n,1)

# ハイパーパラメータ
SEQ_LEN = 100
N_TRAIN = 4000
N_VAL = 1000
BATCH = 128
EPOCHS = 60
LR = 0.001
HIDDEN = 128

X_train, y_train = generate_adding_dataset(N_TRAIN, seq_len=SEQ_LEN)
X_val, y_val = generate_adding_dataset(N_VAL, seq_len=SEQ_LEN)

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
val_ds = torch.utils.data.TensorDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH)

# ======== モデル定義: SimpleRNN と LSTM を比較 ========
class SeqRegressorRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, nonlinearity='tanh')
        self.head = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # x: (batch, seq_len, 2)
        out, hn = self.rnn(x)  # out: (b, seq_len, hidden)
        last = out[:, -1, :]   # 末尾タイムステップの出力
        return self.head(last)

class SeqRegressorLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

# ======== トレーニング関数 ========
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)
        train_losses.append(train_loss)

        # validation
        model.eval()
        with torch.no_grad():
            running = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                running += loss.item() * xb.size(0)
            val_loss = running / len(val_loader.dataset)
            val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch==1:
            print(f"Epoch {epoch:03d}  Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}")
    return model, train_losses, val_losses

# ======== 実験: RNN vs LSTM ========
rnn_model = SeqRegressorRNN()
lstm_model = SeqRegressorLSTM()

print("Training simple RNN...")
rnn_trained, rnn_train_losses, rnn_val_losses = train_model(rnn_model, train_loader, val_loader)

print("\nTraining LSTM...")
lstm_trained, lstm_train_losses, lstm_val_losses = train_model(lstm_model, train_loader, val_loader)

# ======== 可視化: ロス推移 ========
plt.figure(figsize=(8,4))
plt.plot(rnn_train_losses, label="RNN Train")
plt.plot(rnn_val_losses, '--', label="RNN Val")
plt.plot(lstm_train_losses, label="LSTM Train")
plt.plot(lstm_val_losses, '--', label="LSTM Val")
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (log scale)")
plt.legend()
plt.title("Training and Validation Loss: RNN vs LSTM")
plt.grid(True)
plt.show()

# ======== 可視化: 予測精度（散布図） ========
def eval_scatter(model, X, y, n_points=200):
    model.eval()
    with torch.no_grad():
        inp = X[:n_points].to(device)
        pred = model(inp).cpu().numpy().ravel()
    target = y[:n_points].numpy().ravel()
    plt.figure(figsize=(5,5))
    plt.scatter(target, pred, s=8, alpha=0.6)
    plt.plot([0,2],[0,2], 'k--')  # 理想線 y=x
    plt.xlabel("True sum")
    plt.ylabel("Predicted sum")
    plt.title(type(model).__name__)
    plt.axis('equal')
    plt.grid(True)
    plt.show()

print("RNN predictions scatter:")
eval_scatter(rnn_trained, X_val, y_val)
print("LSTM predictions scatter:")
eval_scatter(lstm_trained, X_val, y_val)

# ======== 可視化: 例を取って系列内部を表示（どの位置を参照しているか） ========
def show_example_prediction(model, seq_len=SEQ_LEN):
    model.eval()
    Xs, ys = generate_adding_dataset(1, seq_len=seq_len)
    Xs_t = Xs.to(device)
    with torch.no_grad():
        pred = model(Xs_t).cpu().numpy().ravel()[0]
    values = Xs[0,:,0].numpy()
    markers = Xs[0,:,1].numpy().astype(int)
    true_sum = ys.item()
    plt.figure(figsize=(10,2))
    plt.plot(values, label='value sequence')
    plt.scatter(np.where(markers==1)[0], values[markers==1], color='red', s=80, label='masked positions')
    plt.title(f"True sum={true_sum:.4f}, Predicted={pred:.4f}  ({type(model).__name__})")
    plt.xlabel("Time step")
    plt.legend()
    plt.show()

print("One example RNN:")
show_example_prediction(rnn_trained)
print("One example LSTM:")
show_example_prediction(lstm_trained)