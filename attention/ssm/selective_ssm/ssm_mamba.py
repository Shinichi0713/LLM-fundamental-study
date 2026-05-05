import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
    """
    Selective SSM の参照実装（PyTorchのみ）
    公式実装の selective_scan_ref を参考にした簡易版です。

    引数:
        u:     入力シーケンス (batch, dim, seqlen)
        delta: 時間ステップ (batch, dim, seqlen)
        A:     状態遷移行列 (dim, d_state)
        B:     入力行列 (batch, d_state, seqlen) または (dim, d_state)
        C:     出力行列 (batch, d_state, seqlen) または (dim, d_state)
        D:     スキップ接続 (dim,) または None
        z:     ゲート (batch, dim, seqlen) または None
        delta_bias: delta のバイアス (dim,)
        delta_softplus: delta に softplus を適用するかどうか

    戻り値:
        y: 出力 (batch, dim, seqlen)
    """
    batch, dim, seqlen = u.shape
    d_state = A.shape[-1]

    # delta の処理
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    # B, C の形状を統一（入力依存か静的か）
    if B.dim() == 2:
        # 静的 B: (dim, d_state) -> (batch, d_state, seqlen)
        B = B.t().unsqueeze(0).expand(batch, -1, -1).transpose(1, 2)  # (batch, d_state, seqlen)
    if C.dim() == 2:
        # 静的 C: (dim, d_state) -> (batch, d_state, seqlen)
        C = C.t().unsqueeze(0).expand(batch, -1, -1).transpose(1, 2)  # (batch, d_state, seqlen)

    # A は (dim, d_state) の対角行列と仮定（簡略化）
    # 実際の実装では A はブロック対角など構造化行列ですが、ここでは単純化
    A = -torch.exp(A)  # 安定性のため負の指数

    # 離散化: A_bar = exp(Δ A), B_bar = (exp(Δ A) - I) A^{-1} B
    # ここでは A が対角と仮定して要素ごとに計算
    deltaA = torch.exp(delta.transpose(1, 2).unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))  # (batch, seqlen, dim, d_state)
    # A_bar = deltaA
    # B_bar = (deltaA - 1) * (1/A) * B
    inv_A = 1.0 / A.unsqueeze(0).unsqueeze(2)  # (1, 1, dim, d_state)
    B_bar = (deltaA - 1.0) * inv_A * B.transpose(1, 2).unsqueeze(-1)  # (batch, seqlen, dim, d_state)

    # 状態更新（逐次スキャン）
    # h: (batch, dim, d_state)
    h = torch.zeros(batch, dim, d_state, device=u.device)
    ys = []
    for t in range(seqlen):
        # u_t: (batch, dim)
        u_t = u[:, :, t]
        # h_t = A_bar_t * h_{t-1} + B_bar_t * u_t
        h = deltaA[:, t, :, :] * h + B_bar[:, t, :, :] * u_t.unsqueeze(-1)
        # y_t = C_t * h_t
        y_t = (C[:, :, t].unsqueeze(-1) * h).sum(dim=-2)  # (batch, dim)
        if D is not None:
            y_t = y_t + D.view(1, -1) * u_t
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)  # (batch, dim, seqlen)

    # ゲート z がある場合
    if z is not None:
        y = y * F.silu(z)

    return y