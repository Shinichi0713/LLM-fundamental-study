import numpy as np
import matplotlib.pyplot as plt

# 1. 損失関数（山の地形）: f(x, y) = x^2 + y^2
# 目標は (0, 0) にたどり着くこと
def loss_function(x, y):
    return x**2 + y**2

# 2. 勾配の計算（足元の傾斜）
# f(x, y) を x と y でそれぞれ偏微分します
def calculate_gradient(x, y):
    grad_x = 2 * x  # x^2 の微分
    grad_y = 2 * y  # y^2 の微分
    return grad_x, grad_y

# 3. 勾配降下法の実装
def gradient_descent(start_x, start_y, learning_rate, iterations):
    # 履歴を保存するリスト（可視化用）
    path_x = [start_x]
    path_y = [start_y]
    
    x = start_x
    y = start_y
    
    for i in range(iterations):
        # 今の場所の傾きを計算
        grad_x, grad_y = calculate_gradient(x, y)
        
        # 傾きと逆方向へ「学習率」分だけ進む
        # x_new = x - (学習率 * 傾き)
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y
        
        path_x.append(x)
        path_y.append(y)
        
    return path_x, path_y

# --- 設定 ---
start_x, start_y = -8.0, 6.0  # スタート地点（山の上）
learning_rate = 0.1           # 歩幅（学習率）
iterations = 20               # 何歩進むか

# --- 実行 ---
path_x, path_y = gradient_descent(start_x, start_y, learning_rate, iterations)

# --- 可視化（等高線グラフ） ---
x_range = np.linspace(-10, 10, 100)
y_range = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = loss_function(X, Y)

plt.figure(figsize=(8, 6))

# 山の等高線を描く（青い色が濃いほど低い場所）
plt.contourf(X, Y, Z, levels=20, cmap='Blues_r')
plt.colorbar(label='Loss (Altitude)')

# 移動した経路を赤い点と線で描く
plt.plot(path_x, path_y, 'ro-', label='Path of Descent')
plt.scatter(path_x[0], path_y[0], color='green', s=100, label='Start') # スタート
plt.scatter(0, 0, color='yellow', marker='*', s=200, label='Goal (Minima)') # ゴール

plt.title(f'Gradient Descent Visualization\nLearning Rate: {learning_rate}')
plt.xlabel('Parameter X')
plt.ylabel('Parameter Y')
plt.legend()
plt.grid(True)
plt.show()


