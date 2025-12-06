import numpy as np

# 1. データ生成
# y = 3 * x + 2 という線形関係を持つダミーデータを生成
np.random.seed(42)
X = 2 * np.random.rand(100, 1) # 100個のランダムな入力特徴
y = 3 * X + 2 + np.random.randn(100, 1) # ノイズを加えたターゲット値

# バイアス項のためにXに1列を追加 (X0 = 1)
X_b = np.c_[np.ones((100, 1)), X]

## ----------------------------------------------------

# 2. SGD最適化アルゴリズムの実装
class SGD:
    """確率的勾配降下法 (Stochastic Gradient Descent)"""
    def __init__(self, learning_rate=0.01):
        """
        初期化
        :param learning_rate: 学習率 (eta)
        """
        self.lr = learning_rate

    def update(self, params, grads):
        """
        パラメータの更新を行うメソッド
        :param params: 更新対象のパラメータ (重みとバイアス)
        :param grads: 勾配 (各パラメータの偏微分値)
        :return: 更新されたパラメータ
        """
        # パラメータ = パラメータ - 学習率 * 勾配
        new_params = params - self.lr * grads
        return new_params

## ----------------------------------------------------

# 3. 線形モデルと損失関数の実装
class LinearRegressionModel:
    """線形回帰モデル (最小二乗法)"""
    def __init__(self, num_features):
        # パラメータ (重みとバイアス)をランダムに初期化
        # 1つのバイアス項と1つの特徴量に対する重み
        self.W = np.random.randn(num_features, 1)

    def predict(self, X):
        """予測値の計算 (線形結合)"""
        # X: (バッチサイズ, 特徴量数), W: (特徴量数, 1) -> ドット積: (バッチサイズ, 1)
        return X.dot(self.W)

    def loss_and_grad(self, X, y_true):
        """
        損失 (MSE) の計算と、パラメータに関する勾配の計算を行う
        :param X: バッチの入力データ
        :param y_true: バッチの正解データ
        :return: 損失 (loss), 勾配 (grads)
        """
        m = len(X) # バッチサイズ

        # 予測
        y_pred = self.predict(X)

        # 損失 (平均二乗誤差: MSE)
        loss = np.sum((y_pred - y_true)**2) / m

        # 勾配の計算 (最小二乗法の解析的勾配)
        # 勾配 = 2/m * X^T * (y_pred - y_true)
        grads = 2/m * X.T.dot(y_pred - y_true)
        
        return loss, grads

## ----------------------------------------------------

# 4. 訓練の実行

# ハイパーパラメータの設定
n_epochs = 50       # エポック数 (全データセットを何周するか)
t0, t1 = 5, 50      # 学習率スケジュール用のパラメータ

# SGDの特徴: 学習率を徐々に下げることで収束を助ける
def learning_schedule(t):
    """イテレーションtに応じた学習率を返す関数"""
    return t0 / (t + t1)

# モデルとオプティマイザの初期化
model = LinearRegressionModel(num_features=X_b.shape[1])
optimizer = SGD() # 初期学習率は0.01 (今回はスケジュールで上書き)

m = len(X_b)
history = []
n_batches = 10 # ミニバッチサイズ

for epoch in range(n_epochs):
    # データをシャッフル (SGDの重要なステップ)
    shuffled_indices = np.random.permutation(m)
    X_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    for i in range(0, m, n_batches):
        # ミニバッチの取得
        X_batch = X_shuffled[i:i + n_batches]
        y_batch = y_shuffled[i:i + n_batches]

        # 損失と勾配の計算
        loss, grads = model.loss_and_grad(X_batch, y_batch)
        
        # 学習率の調整 (学習スケジュールを適用)
        t = epoch * m + i
        optimizer.lr = learning_schedule(t)
        
        # パラメータの更新 (SGDステップ)
        model.W = optimizer.update(model.W, grads)
        
        history.append(loss)
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}, Learning Rate: {optimizer.lr:.6f}")

# 結果の表示
print("\n--- 学習結果 ---")
print(f"学習後のパラメータ (W0(バイアス), W1(重み)):\n{model.W.flatten()}")
print(f"真のパラメータ (W0=2, W1=3) に近づきました。")
