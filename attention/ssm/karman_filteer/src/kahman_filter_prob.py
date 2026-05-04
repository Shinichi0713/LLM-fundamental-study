import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ====================
# Kalman filter implementation (行列形式)
# ====================
class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Observation noise covariance
        self.x = x0 # State estimate
        self.P = P0 # Error covariance

    def predict(self):
        # Prediction step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # Update step
        y = z - self.H @ self.x  # Innovation (difference between observation and prediction)
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        # 状態次元に合わせて単位行列のサイズを自動設定
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P

# ====================
# パラメータ設定
# ====================
dt = 0.1  # サンプリング間隔 [s]
T = 100   # ステップ数

# 状態遷移行列 F (等速運動モデル)
F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# 観測行列 H (位置のみ観測)
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

# プロセスノイズ共分散 Q (加速度的な外乱)
q = 0.01
Q = np.diag([0, 0, q, q])  # 位置には直接ノイズを入れない簡略化

# 観測ノイズ共分散 R (GPS誤差)
r = 0.1
R = np.eye(2) * r

# ====================
# 真の軌道生成
# ====================
x_true = np.zeros((T, 4))
x_true[0] = [0, 0, 1, 0.5]  # 初期状態: (0,0) から (vx,vy)=(1,0.5) で移動

for t in range(1, T):
    x_true[t] = F @ x_true[t-1] + np.random.multivariate_normal([0,0,0,0], Q)

# ====================
# 観測生成 (GPS風センサ)
# ====================
observations = np.zeros((T, 2))
for t in range(T):
    observations[t] = H @ x_true[t] + np.random.multivariate_normal([0,0], R)

# ====================
# カルマンフィルタによる推定
# ====================
# 初期値
x0 = np.array([0, 0, 1, 0.5])  # 真の初期状態に近い値を与える
P0 = np.eye(4) * 0.1

kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

# 推定結果を保存する配列
x_upd = np.zeros((T, 4))
x_upd[0] = x0

for t in range(1, T):
    # 予測
    kf.predict()
    # 更新
    kf.update(observations[t])
    # 結果を保存
    x_upd[t] = kf.x

# ====================
# アニメーションの準備
# ====================
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('x position')
ax.set_ylabel('y position')
ax.set_title('Kalman Filter Demo: Noisy GPS vs KF Estimate (Animation)')
ax.grid(True)
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 10)

# プロット用の空ライン・点
true_line, = ax.plot([], [], 'g-', linewidth=2, label='True trajectory', alpha=0.8)
obs_scat = ax.scatter([], [], c='black', s=10, label='Raw observations (Agent A)', alpha=0.6)
kf_line, = ax.plot([], [], 'r-', linewidth=1.5, label='KF estimate (Agent B)')

ax.legend()

# アニメーション更新関数
def update(frame):
    # frame: 0..T-1
    t = frame
    
    # 真の軌跡 (0..t)
    true_line.set_data(x_true[:t+1, 0], x_true[:t+1, 1])
    
    # 観測値そのまま (0..t)
    obs_scat.set_offsets(observations[:t+1])
    
    # KF推定軌跡 (0..t)
    kf_line.set_data(x_upd[:t+1, 0], x_upd[:t+1, 1])
    
    return true_line, obs_scat, kf_line

# アニメーション生成
ani = FuncAnimation(fig, update, frames=T, interval=50, blit=True)

plt.tight_layout()
plt.show()

# 動画をファイルに保存したい場合は以下のコメントを外す（ffmpegが必要）
ani.save('kalman_filter_demo.mp4', writer='ffmpeg', fps=20)