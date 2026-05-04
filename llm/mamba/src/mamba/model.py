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
    
class MambaBlock(nn.Module):
    """
    Mambaブロックの簡易実装
    - in_proj: d_model -> d_inner
    - conv1d: 局所依存性のモデリング
    - ssm: SSM層
    - out_proj: d_inner -> d_model
    - residual connection
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # 入力投影（in_proj）とゲート用の線形層
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 1D causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal padding
            groups=self.d_inner,  # depthwise conv
            bias=False,
        )

        # SSM層
        self.ssm = SimpleSSM(d_inner=self.d_inner, d_state=d_state)

        # 出力投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (batch, length, d_model)
        戻り値: (batch, length, d_model)
        """
        residual = x  # 残差接続用

        # 入力投影 + ゲート
        x = self.in_proj(x)  # (B, L, 2*d_inner)
        x, gate = x.chunk(2, dim=-1)  # それぞれ (B, L, d_inner)

        # 1D causal conv（depthwise）
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)
        x = x[:, :, :residual.size(1)]  # causal padding の調整
        x = x.transpose(1, 2)  # (B, L, d_inner)

        # SSM
        x = self.ssm(x)  # (B, L, d_inner)

        # ゲート適用（SiLUなど）
        x = F.silu(x) * gate

        # 出力投影
        x = self.out_proj(x)  # (B, L, d_model)

        # 残差接続
        x = x + residual
        return x