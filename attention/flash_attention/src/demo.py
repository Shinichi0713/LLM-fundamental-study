import torch
import torch.nn.functional as F

# [Batch, Heads, Seq_len, Head_dim]
q, k, v = torch.randn(8, 12, 1024, 64).cuda().half(), ... 

# Flash Attentionが利用可能な環境なら自動で適用される
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    output = F.scaled_dot_product_attention(q, k, v)

# 概念的なタイリングの実装イメージ
def flash_attention_step(Q, K, V, block_size):
    # 出力 O, 最大値 m, 和 l を初期化
    O = torch.zeros_like(Q)
    m = torch.full((N,), -torch.inf)
    l = torch.zeros((N,))

    # 外側のループ（K, Vのブロック）
    for j in range(0, N, block_size):
        K_j, V_j = K[j:j+block_size], V[j:j+block_size]
        
        # 内側のループ（Qのブロック）
        for i in range(0, N, block_size):
            Q_i = Q[i:i+block_size]
            
            # SRAM内でのスコア計算
            S_ij = torch.matmul(Q_i, K_j.T)
            
            # --- オンラインSoftmaxの更新ロジック ---
            m_new = max(m[i], row_max(S_ij))
            l_new = exp(m[i] - m_new) * l[i] + row_sum(exp(S_ij - m_new))
            
            # 出力の補正と加算
            O[i] = (l[i] * exp(m[i] - m_new) * O[i] + exp(S_ij - m_new) * V_j) / l_new
            
            m[i], l[i] = m_new, l_new
            
    return O

