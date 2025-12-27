import torch
import torch.nn.functional as F

def flash_attention_v2_simulation(Q, K, V, block_size_row=64, block_size_col=64):
    """
    Flash Attention v2 のタイリングアルゴリズムを再現したシミュレーション
    Q, K, V: (seq_len, head_dim) ※簡略化のため1ヘッド分
    """
    N, d = Q.shape
    scale = d ** -0.5
    
    # 1. 出力 O, 最大値行列 m, 指数和行列 l の初期化
    O = torch.zeros_like(Q)
    L = torch.zeros((N, 1))       # 各行の exp(x - m) の和
    M = torch.full((N, 1), -float('inf'))  # 各行の最大値

    # 外側のループ：K, V のブロック（列方向の分割）
    for j in range(0, N, block_size_col):
        K_j = K[j : j + block_size_col] # SRAMへロード
        V_j = V[j : j + block_size_col] # SRAMへロード

        # 内側のループ：Q のブロック（行方向の分割）
        for i in range(0, N, block_size_row):
            Q_i = Q[i : i + block_size_row]
            
            # ブロックごとのスコア計算 (S = QK^T)
            S_ij = torch.matmul(Q_i, K_j.T) * scale
            
            # --- オンラインSoftmaxアルゴリズム ---
            # 1. 現在のブロックの最大値を取得
            m_ij = torch.max(S_ij, dim=1, keepdim=True).values
            
            # 2. 最大値の更新と、既存値・新規値のリスケール係数の計算
            m_new = torch.max(M[i : i + block_size_row], m_ij)
            
            # 指数計算 (オーバーフロー防止のため新しい最大値を引く)
            P_ij = torch.exp(S_ij - m_new)
            l_ij = torch.sum(P_ij, dim=1, keepdim=True)
            
            # 3. 前ステップまでの累積値を新しい最大値に合わせて補正
            alpha = torch.exp(M[i : i + block_size_row] - m_new)
            
            # 4. 出力 O と統計量 L の更新
            # 既存の O を補正し、新しいブロックの寄与を加算
            O[i : i + block_size_row] = O[i : i + block_size_row] * alpha + torch.matmul(P_ij, V_j)
            L[i : i + block_size_row] = L[i : i + block_size_row] * alpha + l_ij
            M[i : i + block_size_row] = m_new

    # 最後に累積した指数和 L で割って正規化を完了させる
    return O / L

# --- 動作確認 ---
seq_len, head_dim = 256, 64
Q = torch.randn(seq_len, head_dim)
K = torch.randn(seq_len, head_dim)
V = torch.randn(seq_len, head_dim)

# 1. 通常の Attention (比較用)
S_ref = torch.matmul(Q, K.T) * (head_dim ** -0.5)
O_ref = torch.matmul(F.softmax(S_ref, dim=-1), V)

# 2. Flash Attention シミュレーション
O_flash = flash_attention_v2_simulation(Q, K, V, block_size_row=32, block_size_col=32)

# 誤差の確認
print(f"Max Difference: {torch.max(torch.abs(O_ref - O_flash)).item():.2e}")