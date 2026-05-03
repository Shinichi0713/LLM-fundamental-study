import numpy as np
import matplotlib.pyplot as plt

# カルマンフィルタの実装
class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F  # 状態遷移行列
        self.H = H  # 観測行列
        self.Q = Q  # プロセスノイズ共分散
        self.R = R  # 観測ノイズ共分散
        self.x = x0 # 状態推定値
        self.P = P0 # 誤差共分散

    def predict(self):
        # 予測ステップ
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # 更新ステップ
        y = z - self.H @ self.x  # イノベーション（観測と予測の差）
        S = self.H @ self.P @ self.H.T + self.R  # イノベーション共分散
        K = self.P @ self.H.T @ np.linalg.inv(S)  # カルマンゲイン

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

# パラメータ設定
dt = 1.0  # サンプリング間隔
F = np.array([[1, dt],
              [0, 1]])  # 状態遷移行列: 等速運動
H = np.array([[1, 0]])  # 観測行列: 位置のみ観測
Q = np.array([[0.1, 0],
              [0, 0.1]])  # プロセスノイズ（小さめ）
R = np.array([[1.0]])     # 観測ノイズ（大きめ）

# 初期状態
x0 = np.array([0, 1.0])  # [位置, 速度]
P0 = np.eye(2) * 1.0    # 初期誤差共分散

# カルマンフィルタのインスタンス
kf = KalmanFilter(F, H, Q, R, x0, P0)

# データ生成
T = 50
true_positions = np.zeros(T)
observed_positions = np.zeros(T)
estimated_positions = np.zeros(T)

# 真の位置: 等速運動
for t in range(T):
    true_positions[t] = t * 1.0  # 速度 1.0

# 観測: 真の位置 + ノイズ
np.random.seed(42)
observed_positions = true_positions + np.random.normal(0, 1.0, T)

# カルマンフィルタの実行
for t in range(T):
    # 予測ステップ
    kf.predict()
    # 更新ステップ（観測値を用いて修正）
    kf.update(observed_positions[t:t+1])
    # 推定された位置を保存
    estimated_positions[t] = kf.x[0]

# 可視化
plt.figure(figsize=(10, 6))
plt.plot(true_positions, label='真の位置', linestyle='--', color='black')
plt.plot(observed_positions, label='観測値（ノイズあり）', marker='o', markersize=3, alpha=0.7)
plt.plot(estimated_positions, label='カルマンフィルタ推定値', linewidth=2)
plt.title('カルマンフィルタによる位置推定')
plt.xlabel('時間ステップ')
plt.ylabel('位置')
plt.legend()
plt.grid(True)
plt.show()