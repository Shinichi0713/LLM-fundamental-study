import numpy as np
import matplotlib.pyplot as plt

def silu(x):
    return x * (1 / (1 + np.exp(-x)))

def simulate_selective_ssm(seq_len=50):
    # 1. 入力データの作成 (0は不要な情報、特定の場所だけ重要な値)
    u = np.zeros(seq_len)
    important_indices = [5, 25]  # ここに重要な情報が出現
    u[5] = 0.8  # ターゲット1
    u[25] = -0.6 # ターゲット2
    
    # SSMの基本パラメータ (固定)
    A = -0.5  # 安定系 (負の値)
    
    # 状態の初期化
    x = 0
    history_x = []
    history_delta = []
    history_B = []

    for k in range(seq_len):
        # 2. Selectiveな挙動の模倣 (本来はここを線形層で学習する)
        if abs(u[k]) > 0:
            # 重要な入力：書き込みを強く、時間は少し進める
            delta = 0.1
            B = 1.0
        else:
            # 不要な入力：書き込みをゼロに、時間は極めてゆっくり（保持）
            # ここで delta を大きくすると「すぐ忘れる」挙動になる
            delta = 0.001 
            B = 0.0
        
        # 3. 離散化
        # A_bar = exp(delta * A)
        # B_bar = (1/A) * (exp(delta * A) - 1) * B  (Zero-Order Hold)
        A_bar = np.exp(delta * A)
        B_bar = (1.0/A) * (np.exp(delta * A) - 1.0) * B
        
        # 4. 状態更新
        x = A_bar * x + B_bar * u[k]
        
        history_x.append(x)
        history_delta.append(delta)
        history_B.append(B)

    return u, history_x, history_delta

# --- 可視化 ---
u, x, deltas = simulate_selective_ssm()

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 入力信号
axes[0].stem(u, basefmt=" ")
axes[0].set_title("Input Sequence (u_k): Only peaks are 'Important'")
axes[0].set_ylabel("Value")

# Deltaの変化
axes[1].step(range(len(deltas)), deltas, where='post', color='orange')
axes[1].set_title("Dynamic Delta (Time Scale): Small = Remember, Large = Forget")
axes[1].set_ylabel("Delta")

# 内部状態（記憶）
axes[2].plot(x, marker='o', color='green')
axes[2].set_title("Internal State (x_k): Memory over time")
axes[2].set_ylabel("State Value")
axes[2].set_xlabel("Time Step (k)")

plt.tight_layout()
plt.show()