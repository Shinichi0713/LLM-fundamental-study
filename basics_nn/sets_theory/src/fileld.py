import numpy as np
import matplotlib.pyplot as plt

def plot_finite_field(p=5):
    """
    有限体 𝔽ₚ の加法・乗法表と逆元の分布を可視化
    """
    if not (isinstance(p, int) and p > 1):
        raise ValueError("p は 2 以上の整数で指定してください")

    values = list(range(p))
    size = p

    # 加法表 (mod p)
    add_table = np.zeros((size, size), dtype=int)
    for i, a in enumerate(values):
        for j, b in enumerate(values):
            add_table[i, j] = (a + b) % p

    # 乗法表 (mod p)
    mul_table = np.zeros((size, size), dtype=int)
    for i, a in enumerate(values):
        for j, b in enumerate(values):
            mul_table[i, j] = (a * b) % p

    # 逆元のチェック（0 以外）
    inverses = {}
    for a in range(1, p):
        for b in range(1, p):
            if (a * b) % p == 1:
                inverses[a] = b
                break

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 加法表
    im0 = axes[0].imshow(add_table, cmap='viridis', interpolation='nearest')
    axes[0].set_title(f'加法表 𝔽_{p} (mod {p})', fontsize=14)
    axes[0].set_xticks(range(size))
    axes[0].set_yticks(range(size))
    axes[0].set_xticklabels(values)
    axes[0].set_yticklabels(values)
    axes[0].set_xlabel('b')
    axes[0].set_ylabel('a')
    plt.colorbar(im0, ax=axes[0])

    # 乗法表（逆元を強調）
    im1 = axes[1].imshow(mul_table, cmap='viridis', interpolation='nearest')
    axes[1].set_title(f'乗法表 𝔽_{p} (逆元を強調)', fontsize=14)
    axes[1].set_xticks(range(size))
    axes[1].set_yticks(range(size))
    axes[1].set_xticklabels(values)
    axes[1].set_yticklabels(values)
    axes[1].set_xlabel('b')
    axes[1].set_ylabel('a')

    # 逆元の位置をマーキング（a の逆元が b なら (a,b) に印）
    for a, inv in inverses.items():
        axes[1].plot(inv, a, 'ro', markersize=8, markeredgecolor='white')

    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.show()

    # 逆元の情報をテキストで表示
    print(f"𝔽_{p} の乗法の逆元:")
    for a in range(1, p):
        print(f"  {a} の逆元: {inverses[a]} ({a} × {inverses[a]} ≡ 1 mod {p})")

# 実行例
plot_finite_field(p=5)  # 素数の例（体）
plot_finite_field(p=7)  # 別の素数の例

import matplotlib.pyplot as plt
import numpy as np

def plot_complex_field():
    """
    複素数体 ℂ の加法・乗法を幾何的に可視化
    """
    # 例として z = 1+2i, w = 2+1i を選ぶ
    z = complex(1, 2)
    w = complex(2, 1)

    # 加法: z + w
    add_result = z + w

    # 乗法: z * w
    mul_result = z * w

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 加法の可視化（ベクトルの和）
    axes[0].quiver(0, 0, z.real, z.imag, angles='xy', scale_units='xy', scale=1, color='blue', label=f'z = {z}')
    axes[0].quiver(z.real, z.imag, w.real, w.imag, angles='xy', scale_units='xy', scale=1, color='green', label=f'w = {w}')
    axes[0].quiver(0, 0, add_result.real, add_result.imag, angles='xy', scale_units='xy', scale=1, color='red', label=f'z+w = {add_result}')
    axes[0].set_xlim(-1, 5)
    axes[0].set_ylim(-1, 5)
    axes[0].set_aspect('equal')
    axes[0].grid(True)
    axes[0].set_title('Complex Addition (Vector Addition)', fontsize=14)
    axes[0].legend()

    # 乗法の可視化（極形式：絶対値と偏角）
    r_z, theta_z = np.abs(z), np.angle(z)
    r_w, theta_w = np.abs(w), np.angle(w)
    r_mul, theta_mul = np.abs(mul_result), np.angle(mul_result)

    # 極座標プロット
    angles = [theta_z, theta_w, theta_mul]
    radii = [r_z, r_w, r_mul]
    labels = [f'z (r={r_z:.2f}, θ={theta_z:.2f})',
              f'w (r={r_w:.2f}, θ={theta_w:.2f})',
              f'z×w (r={r_mul:.2f}, θ={theta_mul:.2f})']
    colors = ['blue', 'green', 'red']

    for i, (theta, r, label, color) in enumerate(zip(angles, radii, labels, colors)):
        axes[1].plot([0, r*np.cos(theta)], [0, r*np.sin(theta)], color=color, linewidth=2, label=label)

    axes[1].set_xlim(-1, 6)
    axes[1].set_ylim(-1, 6)
    axes[1].set_aspect('equal')
    axes[1].grid(True)
    axes[1].set_title('Complex Multiplication (Magnitude and Argument)', fontsize=14)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # テキストでの説明
    print("複素数体 ℂ の性質:")
    print(f"  加法: {z} + {w} = {add_result} (ベクトルの和)")
    print(f"  乗法: {z} × {w} = {mul_result}")
    print(f"    絶対値: |z|×|w| = {r_z:.2f}×{r_w:.2f} = {r_mul:.2f}")
    print(f"    偏角: arg(z)+arg(w) = {theta_z:.2f} + {theta_w:.2f} = {theta_mul:.2f} rad")
    print("  0 以外の複素数は逆元を持つ（例: 1/z など）")

# 実行例
plot_complex_field()
