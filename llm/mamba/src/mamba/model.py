class SimpleSSM(nn.Module):
    """
    単純な離散化SSM（状態空間モデル）の実装
    - A: 状態遷移行列 (d_state, d_state)
    - B: 入力行列 (d_inner, d_state)
    - C: 出力行列 (d_state, d_inner)
    - Δ: 離散化ステップ（時間刻み） (d_inner,)
    """
    def __init__(self, d_inner, d_state):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # パラメータ初期化
        self.A_log = nn.Parameter(torch.log(torch.ones(d_state)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.B = nn.Parameter(torch.randn(d_inner, d_state) / d_state**0.5)
        self.C = nn.Parameter(torch.randn(d_state, d_inner) / d_inner**0.5)

        # Δ を出力する線形層（論文の sΔ に相当）
        self.delta_proj = nn.Linear(d_inner, d_inner, bias=True)

    def discretize(self, delta):
        """
        A, B を離散化: A_bar = exp(Δ A), B_bar = (exp(Δ A) - I) A^{-1} B
        簡略化のため A は対角行列と仮定（Mambaの設計に近い）
        """
        A = -torch.exp(self.A_log.float())  # (d_state,)
        # A_bar = exp(Δ A)
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))  # (B, L, d_state)
        # B_bar = (exp(Δ A) - I) * (A)^{-1} B
        inv_A = 1.0 / A  # (d_state,)
        B_bar = (A_bar - 1.0) * inv_A.unsqueeze(0) @ self.B.T  # (B, L, d_inner)
        return A_bar, B_bar

    def forward(self, u):
        """
        u: (batch, length, d_inner)
        戻り値: y: (batch, length, d_inner)
        """
        batch, L, d = u.shape
        assert d == self.d_inner

        # Δ を計算
        delta = F.softplus(self.delta_proj(u))  # (B, L, d_inner)

        # A_bar, B_bar を離散化
        A_bar, B_bar = self.discretize(delta)  # A_bar: (B, L, d_state), B_bar: (B, L, d_inner)

        # 状態 h の初期化
        h = torch.zeros(batch, self.d_state, device=u.device)  # (B, d_state)

        outputs = []
        for t in range(L):
            # 入力 u_t: (B, d_inner)
            u_t = u[:, t, :]
            # 状態更新: h_{t} = A_bar_t * h_{t-1} + B_bar_t * u_t
            h = A_bar[:, t, :] * h + B_bar[:, t, :] * u_t.unsqueeze(-1)  # (B, d_state)
            # 出力: y_t = C * h_t + D * u_t
            y_t = (self.C @ h.unsqueeze(-1)).squeeze(-1) + self.D * u_t  # (B, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        return y