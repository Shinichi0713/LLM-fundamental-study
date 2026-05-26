import numpy as np

def tucker_als(X, ranks, max_iter=100, tol=1e-6):
    """
    Tucker分解（3階テンソル）をALS（HOOI）で近似する修正版
    ranks = (R1, R2, R3)
    """
    I, J, K = X.shape
    R1, R2, R3 = ranks
    
    # 因子行列の初期化（ランダムな直交行列）
    U1, _ = np.linalg.qr(np.random.randn(I, R1))
    U2, _ = np.linalg.qr(np.random.randn(J, R2))
    U3, _ = np.linalg.qr(np.random.randn(K, R3))
    
    for it in range(max_iter):
        U1_old, U2_old, U3_old = U1.copy(), U2.copy(), U3.copy()
        
        # --- U1の更新 ---
        # Y1 = X ×2 U2^T ×3 U3^T  (形状: I, R2, R3)
        Y1 = np.einsum('ijk,jr,ks->irs', X, U2, U3)
        # モード1展開 (I, R2*R3)
        Y1_mode1 = Y1.reshape(I, -1)
        # SVDによる左特異ベクトルの抽出
        U, _, _ = np.linalg.svd(Y1_mode1, full_matrices=False)
        U1 = U[:, :R1]
        
        # --- U2の更新 ---
        # Y2 = X ×1 U1^T ×3 U3^T  (形状: R1, J, R3)
        Y2 = np.einsum('ijk,ir,ks->rjs', X, U1, U3)
        # モード2展開 (J, R1*R3)
        Y2_mode2 = Y2.transpose(1, 0, 2).reshape(J, -1)
        U, _, _ = np.linalg.svd(Y2_mode2, full_matrices=False)
        U2 = U[:, :R2]
        
        # --- U3の更新 ---
        # Y3 = X ×1 U1^T ×2 U2^T  (形状: R1, R2, K)
        Y3 = np.einsum('ijk,ir,js->rsk', X, U1, U2)
        # モード3展開 (K, R1*R2)
        Y3_mode3 = Y3.transpose(2, 0, 1).reshape(K, -1)
        U, _, _ = np.linalg.svd(Y3_mode3, full_matrices=False)
        U3 = U[:, :R3]
        
        # 収束判定（変化量の最大値が閾値未満か）
        diff_U1 = np.linalg.norm(U1 - U1_old)
        diff_U2 = np.linalg.norm(U2 - U2_old)
        diff_U3 = np.linalg.norm(U3 - U3_old)
        if max(diff_U1, diff_U2, diff_U3) < tol:
            print(f"収束しました（反復 {it+1} 回）")
            break
            
    # コアテンソルの計算
    # G = X ×1 U1^T ×2 U2^T ×3 U3^T
    G = np.einsum('ijk,ir,js,kt->rst', X, U1, U2, U3)
    
    return G, U1, U2, U3

def reconstruct_tucker(G, U1, U2, U3):
    """
    Tucker分解からテンソルを再構成する
    """
    return np.einsum('rst,ir,js,kt->ijk', G, U1, U2, U3, optimize=True)

# --- 実行テスト ---
# 小さなテンソルを作成（2x3x2）
X = np.array([
    [[1, 2],
     [3, 4],
     [5, 6]],
    [[7, 8],
     [9, 10],
     [11, 12]]
]).astype(float)  # SVD等の計算安定化のためfloat型を明示

print("元のテンソル X:")
print(X)
print("形状:", X.shape)

# Tucker分解（ランク (2,2,2)）
ranks = (2, 2, 2)
G, U1, U2, U3 = tucker_als(X, ranks=ranks)

print("\nコアテンソル G:")
print(G)
print("形状:", G.shape)

print("\n因子行列 U1 (I x R1):")
print(U1)
print("因子行列 U2 (J x R2):")
print(U2)
print("因子行列 U3 (K x R3):")
print(U3)

# 再構成
X_hat = reconstruct_tucker(G, U1, U2, U3)

print("\n再構成テンソル X_hat:")
print(np.round(X_hat, 4))  # 見やすくするため丸め処理

print("\n誤差（Frobeniusノルム）:", np.linalg.norm(X - X_hat))