import numpy as np
import matplotlib.pyplot as plt

def f(x, dt, a, theta):
    """非線形状態遷移関数"""
    px, py, vx, vy = x
    return np.array([
        px + vx * dt,
        py + vy * dt,
        vx + a * np.cos(theta),
        vy + a * np.sin(theta)
    ])

def jacobian_f(x, dt, a, theta):
    """f(x) のヤコビアン ∂f/∂x"""
    # ここでは vx, vy に依存しない単純な例
    return np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ])

def extended_kalman_filter(observations, x0, P0, Q, R, dt, a, theta):
    """拡張カルマンフィルタ（EKF）の実装"""
    n_states = len(x0)
    T = len(observations)
    
    x_pred = np.zeros((T, n_states))
    x_upd = np.zeros((T, n_states))
    P_pred = np.zeros((T, n_states, n_states))
    P_upd = np.zeros((T, n_states, n_states))
    
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    
    # 初期化
    x_upd[0] = x0
    P_upd[0] = P0
    
    for t in range(1, T):
        # 予測ステップ
        x_pred[t] = f(x_upd[t-1], dt, a, theta)
        F = jacobian_f(x_upd[t-1], dt, a, theta)
        P_pred[t] = F @ P_upd[t-1] @ F.T + Q
        
        # 更新ステップ
        y_pred = H @ x_pred[t]
        innov = observations[t] - y_pred
        S = H @ P_pred[t] @ H.T + R
        K = P_pred[t] @ H.T @ np.linalg.inv(S)
        
        x_upd[t] = x_pred[t] + K @ innov
        P_upd[t] = (np.eye(n_states) - K @ H) @ P_pred[t]
    
    return x_upd, P_upd, x_pred, P_pred

# パラメータ設定
dt = 0.1
a = 0.5
theta = np.pi / 4
T = 100

# 真の状態生成（簡易シミュレーション）
x_true = np.zeros((T, 4))
x_true[0] = [0, 0, 1, 1]
for t in range(1, T):
    x_true[t] = f(x_true[t-1], dt, a, theta) + np.random.multivariate_normal([0,0,0,0], np.eye(4)*0.01)

# 観測生成（位置のみ）
H_obs = np.array([[1,0,0,0],[0,1,0,0]])
observations = H_obs @ x_true.T + np.random.multivariate_normal([0,0], np.eye(2)*0.1, size=T).T
observations = observations.T  # (T, 2)

# EKF実行
x0 = np.array([0, 0, 1, 1])
P0 = np.eye(4) * 0.1
Q = np.eye(4) * 0.01
R = np.eye(2) * 0.1

x_upd, P_upd, x_pred, P_pred = extended_kalman_filter(observations, x0, P0, Q, R, dt, a, theta)

# 可視化
plt.figure(figsize=(10, 6))
plt.plot(x_true[:, 0], x_true[:, 1], 'g-', label='True trajectory', alpha=0.7)
plt.plot(observations[:, 0], observations[:, 1], 'ko', markersize=2, label='Observations', alpha=0.5)
plt.plot(x_upd[:, 0], x_upd[:, 1], 'r-', label='EKF estimate')
plt.legend()
plt.xlabel('px')
plt.ylabel('py')
plt.title('Extended Kalman Filter (EKF) Example')
plt.grid(True)
plt.show()