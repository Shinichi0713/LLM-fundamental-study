import numpy as np

# --- 1. 定義（PyTorchの重みとバイアスと等価なNumPy配列） ---

# PyTorchの fc_layer.weight.data = torch.tensor([[10.0, -5.0, -10.0]]) に対応
# NumPyでは形状 (1, 3) の配列として定義
W = np.array([[10.0, -5.0, -10.0]])

# PyTorchの fc_layer.bias.data = torch.tensor([50.0]) に対応
# NumPyでは形状 (1,) または (1, 1) の配列として定義（ブロードキャストのため）
b = np.array([50.0])


# --- 2. 入力データの準備 ---

# PyTorchの shop_a = torch.tensor([[10.0, 1.0, 1.0]]) に対応
# 形状 (1, 3) の配列として定義
shop_a_np = np.array([[10.0, 1.0, 1.0]])

# PyTorchの shop_b = torch.tensor([[3.0, 0.5, 0.0]]) に対応
shop_b_np = np.array([[3.0, 0.5, 0.0]])


# --- 3. 全結合層の計算ロジック（線形変換: Y = X @ W.T + b） ---

def fully_connected_forward(X_in, W_params, b_params):
    """
    全結合層の順伝播を計算します。
    X_in @ W_params.T + b_params
    """
    # 処理 1: 行列の積 (X と W の転置 W.T のドット積)
    # X_in (1, 3) と W_params.T (3, 1) の積は (1, 1) となります。
    # NumPyでは @ 演算子または np.dot を使用
    weighted_sum = X_in @ W_params.T 
    
    # 処理 2: バイアス b の加算
    # weighted_sum (1, 1) に b_params (1,) がブロードキャストされて加算されます。
    output = weighted_sum + b_params
    
    # スコア（要素を取り出す）
    return output.item()


# --- 4. 計算実行と結果表示 ---

score_a = fully_connected_forward(shop_a_np, W, b)
score_b = fully_connected_forward(shop_b_np, W, b)

print(f"A店の満足度: {score_a:.1f} 点")
print(f"B店の満足度: {score_b:.1f} 点")