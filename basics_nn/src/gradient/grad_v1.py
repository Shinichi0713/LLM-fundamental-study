class SimpleNode:
    def __init__(self):
        self.x = None
        self.w = None
        self.y = None
        self.t = None

    # 順伝播
    def forward(self, x, w, t):
        self.x = x
        self.w = w
        self.t = t
        
        # 1. 予測の計算 y = x * w
        self.y = self.x * self.w
        
        # 2. 損失の計算 L = (y - t)^2
        loss = (self.y - self.t) ** 2
        
        return loss

    # 逆伝播（勾配計算）
    def backward(self):
        # 連鎖律: dL/dw = (dL/dy) * (dy/dw)
        
        # 1. 後ろからの勾配 (dL/dy)
        # L = (y - t)^2 の微分 -> 2 * (y - t)
        grad_L_y = 2 * (self.y - self.t)
        
        # 2. 手前の勾配 (dy/dw)
        # y = x * w の微分 (wで微分) -> x
        grad_y_w = self.x
        
        # 3. 最終的な勾配 (dL/dw)
        grad_w = grad_L_y * grad_y_w
        
        return grad_w

# --- 実行 ---
# データ設定
x = 2.0
w = 3.0  # 初期重み
t = 10.0 # 正解

node = SimpleNode()

# 1. 順伝播でLossを計算
loss = node.forward(x, w, t)
print(f"Loss: {loss}")  # 結果: 16.0

# 2. 逆伝播で勾配を計算
gradient = node.backward()
print(f"Gradient (dL/dw): {gradient}") # 結果: -16.0

# 3. パラメータの更新 (学習率 lr = 0.1)
lr = 0.1
w_new = w - lr * gradient
print(f"Updated Weight: {w_new}") 
# 計算: 3.0 - 0.1 * (-16.0) = 3.0 + 1.6 = 4.6
# 重みが 3.0 -> 4.6 に増え、正解(10.0)を出すためにより適切な値に近づいた