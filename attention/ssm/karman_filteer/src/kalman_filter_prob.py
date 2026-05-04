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
# ローパスフィルタ（IIR）の実装
# ====================
class LowPassFilter:
    def __init__(self, alpha, dim=2):
        self.alpha = alpha  # 平滑化パラメータ (0 < alpha < 1)
        self.y = None       # フィルタ出力
        self.dim = dim

    def update(self, z):
        if self.y is None:
            self.y = z.copy()
        else:
            self.y = self.alpha * self.y + (1 - self.alpha) * z
        return self.y

# ====================
# パラメータ設定
# ====================
dt = 0.1  # サンプリング間隔 [s]
T = 200   # ステップ数（円運動を見るため少し長めに）

# 状態遷移行列 F (等速運動モデル) ※KFは依然として等速モデルを使う
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
# 真の軌道生成（円運動）
# ====================
# 円運動パラメータ
radius = 5.0      # 半径
omega = 0.1       # 角速度 [rad/s]

# 真の位置・速度を保存する配列
x_true = np.zeros((T, 4))

for t in range(T):
    theta = omega * t * dt
    # 位置
    x_true[t, 0] = radius * np.cos(theta)  # x
    x_true[t, 1] = radius * np.sin(theta)  # y
    # 速度（微分から計算）
    x_true[t, 2] = -radius * omega * np.sin(theta)  # vx
    x_true[t, 3] =  radius * omega * np.cos(theta)  # vy

# ====================
# 観測生成 (GPS風センサ)
# ====================
observations = np.zeros((T, 2))
for t in range(T):
    observations[t] = H @ x_true[t] + np.random.multivariate_normal([0,0], R)

# ====================
# カルマンフィルタによる推定
# ====================
# 初期値（真の初期状態に近い値を与える）
x0 = x_true[0].copy()
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
# ローパスフィルタによる平滑化
# ====================
alpha = 0.8  # ローパスフィルタのパラメータ（大きいほど平滑化が強い）
lpf = LowPassFilter(alpha=alpha, dim=2)

lpf_output = np.zeros((T, 2))
for t in range(T):
    lpf_output[t] = lpf.update(observations[t])

# ====================
# 誤差評価（KF vs ローパス）
# ====================
true_pos = x_true[:, :2]  # 真の位置
kf_pos = x_upd[:, :2]     # KF推定位置

# KFの位置誤差RMSE
kf_rmse = np.sqrt(np.mean(np.sum((kf_pos - true_pos)**2, axis=1)))

# ローパスフィルタの位置誤差RMSE
lpf_rmse = np.sqrt(np.mean(np.sum((lpf_output - true_pos)**2, axis=1)))

# KFの累積位置誤差
kf_integral_error = np.sum(np.sqrt(np.sum((kf_pos - true_pos)**2, axis=1))) * dt

# ローパスフィルタの累積位置誤差
lpf_integral_error = np.sum(np.sqrt(np.sum((lpf_output - true_pos)**2, axis=1))) * dt

print("=== 誤差評価結果 ===")
print(f"KF RMSE: {kf_rmse:.4f}")
print(f"LPF RMSE: {lpf_rmse:.4f}")
print(f"KF integral error: {kf_integral_error:.4f}")
print(f"LPF integral error: {lpf_integral_error:.4f}")

# ====================
# アニメーションの準備
# ====================
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlabel('x position')
ax.set_ylabel('y position')
ax.set_title('Kalman Filter vs Low-Pass Filter Demo (Circular Motion)')
ax.grid(True)
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_aspect('equal')

# プロット用の空ライン・点
true_line, = ax.plot([], [], 'g-', linewidth=2, label='True trajectory', alpha=0.8)
obs_scat = ax.scatter([], [], c='black', s=10, label='Raw observations', alpha=0.4)
lpf_line, = ax.plot([], [], 'b-', linewidth=1.5, label='Low-pass filter', alpha=0.8)
kf_line, = ax.plot([], [], 'r-', linewidth=1.5, label='Kalman filter', alpha=0.8)

ax.legend()

# アニメーション更新関数
def update(frame):
    # frame: 0..T-1
    t = frame
    
    # 真の軌跡 (0..t)
    true_line.set_data(x_true[:t+1, 0], x_true[:t+1, 1])
    
    # 観測値そのまま (0..t)
    obs_scat.set_offsets(observations[:t+1])
    
    # ローパスフィルタ出力 (0..t)
    lpf_line.set_data(lpf_output[:t+1, 0], lpf_output[:t+1, 1])
    
    # KF推定軌跡 (0..t)
    kf_line.set_data(x_upd[:t+1, 0], x_upd[:t+1, 1])
    
    return true_line, obs_scat, lpf_line, kf_line

# アニメーション生成
ani = FuncAnimation(fig, update, frames=T, interval=50, blit=True)

plt.tight_layout()
plt.show()

# 動画をファイルに保存したい場合は以下のコメントを外す（ffmpegが必要）
ani.save('kalman_vs_lowpass_circle.mp4', writer='ffmpeg', fps=20)