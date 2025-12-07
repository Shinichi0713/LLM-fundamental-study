
__例題:__ ドロップアウトの効果

ドロップアウトの効果が分かる例題について扱います。
扱う対象はニューラルネットワークでドロップアウトの有無に差をつけて、小規模データによる学習で学習と、ニューラルネットワークによる推論の精度の差を確認します。

* 小規模データはすぐ **過学習する（train精度だけ上がってtestは悪化する）**
* ドロップアウトは**ランダムにニューロンを無効化** → 特定のニューロンへの依存を防ぎ、より汎化性能が良くなる
* 結果として
- ドロップアウトなし → train精度100%、test精度低い（過学習）
- ドロップアウトあり → train精度やや低い、test精度が高い（汎化）

このコードでは、

* 2クラス分類（2次元の簡単な点群データ）
* ドロップアウトあり／なし の **2つのモデルを学習**
* **学習曲線（train/test loss）を比較してプロット**
* ドロップアウトの効果が一目で分かる

実装コードは以下です。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ===== データ作成（半月型、学習しやすいが過学習しやすい） =====
X, y = make_moons(n_samples=800, noise=0.25, random_state=0)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== モデル定義（Dropout なし） =====
class NetNoDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

# ===== モデル定義（Dropout あり） =====
class NetDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # ★ここが重要
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # ★ここも重要
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

# ===== モデル準備 =====
model_no = NetNoDropout()
model_do = NetDropout()

criterion = nn.CrossEntropyLoss()
optimizer_no = optim.Adam(model_no.parameters(), lr=0.001)
optimizer_do = optim.Adam(model_do.parameters(), lr=0.001)

# ===== 学習関数 =====
def train_model(model, optimizer, X_train, y_train, X_test, y_test, epochs=200):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # --- train ---
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # --- test ---
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

    return train_losses, test_losses

# ===== 学習 =====
train_no, test_no = train_model(model_no, optimizer_no, X_train, y_train, X_test, y_test)
train_do, test_do = train_model(model_do, optimizer_do, X_train, y_train, X_test, y_test)

# ===== 可視化 =====
plt.figure(figsize=(10,5))
plt.title("Dropout の有無による学習曲線の違い")
plt.plot(train_no, label="Train (No Dropout)")
plt.plot(test_no, label="Test (No Dropout)")
plt.plot(train_do, label="Train (Dropout)", linestyle="--")
plt.plot(test_do, label="Test (Dropout)", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
```

__出力の結果__

__ドロップアウトなし__

* train loss → どんどん下がる（完璧に暗記）
* test loss → 途中から上昇（過学習）

__ドロップアウトあり__

* train loss → やや下がりにくい（ニューロンを間引くため）
* test loss → 安定して低い（汎化性能が向上）

![1765066724389](image/explanation/1765066724389.png)
