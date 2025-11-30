import numpy as np

# --- 1. ReLU (Rectified Linear Unit) 関数の定義 ---
def relu(x):
    """
    ReLU関数を実装します。
    入力が0より大きければそのままの値を、0以下なら0を出力します。
    """
    return np.maximum(0, x)

# --- 2. Sigmoid 関数の定義 ---
def sigmoid(x):
    """
    Sigmoid関数を実装します。
    出力を常に0から1の間に収めます。
    """
    return 1 / (1 + np.exp(-x))

# ----------------------------------------------------
# 3. デモンストレーション
# ----------------------------------------------------

# (1) 仮想的な線形計算結果 (W*x + b) を用意
#    - 通常、活性化関数の入力はマイナスの値も含む「線形計算の結果」です。
input_data = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
print(f"入力データ (線形計算結果): {input_data}")
print("-" * 30)

# -----------------------------
# A. ReLUの適用
# -----------------------------
output_relu = relu(input_data)
print("A. ReLU 適用後の出力:")
print(f"  入力がマイナス値のとき: 0 になります (非線形性)")
print(f"  出力: {output_relu}")

# -----------------------------
# B. Sigmoidの適用
# -----------------------------
output_sigmoid = sigmoid(input_data)
print("\nB. Sigmoid 適用後の出力:")
print(f"  全ての出力が 0 から 1 の間に収束しました (非線形性)")
print(f"  出力: {output_sigmoid}")

