def local_attention(q, k, v, window):
    """
    q, k, v: (B, H, T, D)
    window: int（左右何トークンを見るか）
    """
    B, H, T, D = q.shape

    outputs = []
    for t in range(T):
        left = max(0, t - window)
        right = min(T, t + window + 1)

        # shape: (B,H,1,D) @ (B,H,D,W) → (B,H,1,W)
        q_t = q[:, :, t:t+1, :]               # 1 token
        k_slice = k[:, :, left:right, :]     # window range
        v_slice = v[:, :, left:right, :]

        attn = torch.matmul(q_t, k_slice.transpose(-2, -1)) / (D ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_slice)    # (B,H,1,D)

        outputs.append(out)

    # concat each timestep output
    outputs = torch.cat(outputs, dim=2)  # (B,H,T,D)
    return outputs
