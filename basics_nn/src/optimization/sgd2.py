import numpy as np

# ----------------------------------------------------
# 1. SGD最適化アルゴリズムの実装（クラス化）
# ----------------------------------------------------
class SGD:
    """確率的勾配降下法 (Stochastic Gradient Descent)"""
    def __init__(self, learning_rate=0.01, t0=5, t1=50, use_schedule=False):
        """
        初期化
        :param learning_rate: 初期学習率 (固定の場合に使用)
        :param t0, t1: 学習率スケジュール用のパラメータ
        :param use_schedule: 学習率スケジュールを使用するかどうか
        """
        self.lr = learning_rate
        self.t0 = t0
        self.t1 = t1
        self.use_schedule = use_schedule
        self.t = 0  # イテレーション回数をカウントするカウンター

    def _learning_schedule(self):
        """イテレーションtに応じた学習率を返すプライベートメソッド"""
        # tが1から始まるため、t-1を使用
        return self.t0 / (self.t + self.t1)

    def update(self, params, grads):
        """
        パラメータの更新を行うメソッド
        :param params: 更新対象のパラメータ (重みとバイアス)
        :param grads: 勾配 (各パラメータの偏微分値)
        :return: 更新されたパラメータ
        """
        self.t += 1 # イテレーション回数をインクリメント

        current_lr = self.lr
        
        # 学習スケジュールを使用する場合、学習率を更新
        if self.use_schedule:
             current_lr = self._learning_schedule()

        # SGDの更新式: パラメータ = パラメータ - 現在の学習率 * 勾配
        params -= current_lr * grads
        return params

# ----------------------------------------------------
# 2. 線形モデルとデータ生成（元のコードから再掲）
# ----------------------------------------------------
class LinearRegressionModel:
    """線形回帰モデル (最小二乗法)"""
    def __init__(self, num_features):
        self.W = np.random.randn(num_features, 1)

    def predict(self, X):
        return X.dot(self.W)

    def loss_and_grad(self, X, y_true):
        m = len(X)
        y_pred = self.predict(X)
        loss = np.sum((y_pred - y_true)**2) / m
        grads = 2/m * X.T.dot(y_pred - y_true)
        return loss, grads

# データ生成
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

# ----------------------------------------------------
# 3. 訓練の実行
# ----------------------------------------------------

# ハイパーパラメータの設定
n_epochs = 50
n_batches = 10 # ミニバッチサイズ

# モデルとオプティマイザの初期化
model = LinearRegressionModel(num_features=X_b.shape[1])
# 学習率スケジュールを使用するように設定
optimizer = SGD(learning_rate=0.01, use_schedule=True) 

m = len(X_b)

print(f"--- 訓練開始 (SGD) ---")

for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    for i in range(0, m, n_batches):
        X_batch = X_shuffled[i:i + n_batches]
        y_batch = y_shuffled[i:i + n_batches]

        # 損失と勾配の計算
        loss, grads = model.loss_and_grad(X_batch, y_batch)
        
        # パラメータの更新 (SGDステップ) - 学習率の調整はupdate内で自動的に行われる
        model.W = optimizer.update(model.W, grads)
        
    if (epoch + 1) % 10 == 0:
        # 現在の学習率は、最後に update が呼ばれた時点の値です
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}, Iteration t: {optimizer.t}, Current LR: {optimizer._learning_schedule():.6f}")

# 結果の表示
print("\n--- 学習結果 ---")
print(f"学習後のパラメータ (W0(バイアス), W1(重み)):\n{model.W.flatten()}")
print(f"真のパラメータ (W0=2, W1=3) に近づきました。")