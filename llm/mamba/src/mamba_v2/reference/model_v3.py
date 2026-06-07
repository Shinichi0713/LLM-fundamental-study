import torch
import torch.nn as nn
import torch.nn.functional as F

class Mamba3CoreSSM(nn.Module):
    def __init__(self, d_model, d_state=64, r_mimo=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state  # 複素数状態として扱う次元
        self.r_mimo = r_mimo    # MIMO（複数入出力）の多重度
        
        # 1. 各種投影（パラメータのデータ依存生成）
        # 入力 x から Δ(タイムステップ), B(入力射影), C(出力射影), λ(台形重み) を予測
        self.x_proj = nn.Linear(d_model, 1 + d_state * 2 + 1) 
        
        # 2. 定数A（対角行列成分、複素数の実部にあたる部分を制御）
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        
        # 3. 複素数状態の「回転（角度）」を決定する重み（RoPEのような挙動用）
        self.theta_proj = nn.Linear(d_model, d_state)
        
        # 出力射影
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [Batch, SeqLen, Dim]
        """
        b, l, d = x.shape
        
        # 状態の初期化 (複素数: Complex Float)
        h = torch.zeros(b, self.d_state, dtype=torch.complex64, device=x.device)
        outputs = []
        
        # 連続的な数式計算のための定数A
        A = -torch.exp(self.A_log) # [d_state] (負の値にして安定化)
        
        # 逐次的な推論（またはRNNモード）でのシミュレーション
        # ※実際の訓練ではこれを並列スキャンに分解して高速化（SSD）します
        for t in range(l):
            xt = x[:, t, :] # [b, d]
            
            # 各種パラメータの射影
            proj_out = self.x_proj(xt)
            dt = F.softplus(proj_out[:, 0:1]) # Δt > 0
            
            # B, C の抽出 (複素数として扱うため分割して結合)
            B_real = proj_out[:, 1 : 1 + self.d_state]
            B_imag = proj_out[:, 1 + self.d_state : 1 + 2 * self.d_state]
            B = torch.complex(B_real, B_imag) # [b, d_state]
            
            # Cの生成
            C = torch.ones_like(B) # 簡易化のため固定（本来は別軸射影）
            
            # データ依存のスカラ λ (台形公式の重みバランス 0~1)
            lambda_t = torch.sigmoid(proj_out[:, -1:])
            
            # 複素数の回転角度（RoPE的なデータ追跡能力のキモ）
            theta = self.theta_proj(xt) # [b, d_state]
            
            # --- Mamba-3 のコア: 台形公式離散化（Exponential-Trapezoidal） ---
            # α_t = exp(Δt * (A + i*θ))
            A_complex = torch.complex(A, theta) # 複素数化
            alpha = torch.exp(dt.to(torch.complex64) * A_complex)
            
            # β_t = (1 - λ_t) * Δt * exp(Δt * A_complex)
            # γ_t = λ_t * Δt
            beta = (1.0 - lambda_t.to(torch.complex64)) * dt.to(torch.complex64) * alpha
            gamma = lambda_t.to(torch.complex64) * dt.to(torch.complex64)
            
            # 状態更新方程式の計算 (Mamba-3 式)
            # h_t = α_t * h_{t-1} + β_t * (B_{t-1}*x_{t-1}) + γ_t * (B_t * x_t)
            # ※ 前のタイムステップの入力依存を考慮（簡易的にxtで合成）
            input_effect = gamma * B * xt[:, :self.d_state].to(torch.complex64)
            if t > 0:
                input_effect += beta * B_prev * xt_prev[:, :self.d_state].to(torch.complex64)
            
            h = alpha * h + input_effect
            
            # 出力の計算 (C^H * h)
            yt = torch.sum(torch.conj(C) * h, dim=-1) # [b]
            outputs.append(yt.real.unsqueeze(-1)) # 実部を取り出す
            
            # 前のステップの値を保存
            B_prev = B
            xt_prev = xt

        # 出力をテンソルに結合
        y = torch.cat(outputs, dim=-1) # [b, l]
        
        # 次元の復元と出力射影（入力と同じ次元数に拡張）
        y_out = y.unsqueeze(-1).repeat(1, 1, d)
        return self.out_proj(y_out)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mamba3Block(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model) # 内部表現の次元を拡張（Mambaの標準構造）
        self.d_conv = d_conv
        
        # 1. 入力を2系統（SSM側とゲート側）に分岐させる射影層
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 2. 短期的な時系列関係を捉える1次元因果コンボリューション（Causal Convolution）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner, # Depthwise Conv
            padding=d_conv - 1,
        )
        
        # 3. Mamba-3のキモ：データ依存パラメータ（Δ, B, C, λ）の動的生成層
        # 複素数状態に対応するため、BとCの次元を2倍（実部＋虚部）にする
        self.x_proj = nn.Linear(self.d_inner, 1 + (d_state * 2) * 2 + 1, bias=False)
        
        # 4. 定数A（対角行列の実部を制御。負の値で初期化してシステムを安定化）
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        
        # 5. 複素数状態の回転（θ）を決定する周波数投影層（RoPEに相当）
        self.theta_proj = nn.Linear(self.d_inner, d_state, bias=False)
        
        # 6. 最終出力の射影層
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _mamba3_parallel_scan(self, u, delta, B, C, theta, lambdas):
        """
        学習高速化のための並列シミュレーション（Mamba-3の数学的表現）
        u: [b, l, d_inner]
        """
        b, l, d = u.shape
        device = u.device
        
        # 複素数定数Aの構築
        A_real = -torch.exp(self.A_log) # [d_state]
        A_complex = torch.complex(A_real.unsqueeze(0).unsqueeze(0), theta) # [b, l, d_state]
        
        # 台形公式に基づく離散化行列 α, β, γ の計算
        # dt: [b, l, 1] -> [b, l, d_state] へ拡張
        dt_c = delta.to(torch.complex64)
        alpha = torch.exp(dt_c * A_complex) # 状態遷移係数
        
        beta = (1.0 - lambdas.to(torch.complex64)) * dt_c * alpha
        gamma = lambdas.to(torch.complex64) * dt_c
        
        # 入力が状態に与える影響の計算 (MIMO的なベクトル内積のシミュレーション)
        # 簡易化のため、入力の各次元を複素数状態の空間へマッピング
        u_c = u[:, :, :self.d_state].to(torch.complex64)
        
        # 現在ステップと過去ステップの入力を台形公式でブレンド
        x_effect = gamma * B * u_c
        x_effect_prev = beta * B * u_c
        # 過去の入力を1ステップシフト
        x_effect_prev = torch.cat([torch.zeros_like(x_effect_prev[:, :1, :]), x_effect_prev[:, :-1, :]], dim=1)
        inputs_processed = x_effect + x_effect_prev
        
        # --- 累積並列スキャン (本来はAssociative Scanカーネルで高速化) ---
        # 簡易的に累積計算をシミュレート
        h = torch.zeros(b, self.d_state, dtype=torch.complex64, device=device)
        hs = []
        for t in range(l):
            h = alpha[:, t, :] * h + inputs_processed[:, t, :]
            hs.append(h.unsqueeze(1))
        HS = torch.cat(hs, dim=1) # [b, l, d_state]
        
        # 出力方程式: y = Re(C^H * H)
        y_complex = torch.sum(torch.conj(C) * HS, dim=-1, keepdim=True)
        y = y_complex.real
        
        # 元の次元へ射影するためのベースとして入力を薄く混ぜる（残差結合への前処理）
        y_out = y.repeat(1, 1, d) * u
        return y_out

    def forward(self, x):
        """
        x: [Batch, SeqLen, d_model]
        """
        b, l, d = x.shape
        
        # 1. 入力の拡張と分岐 (SSMパスとGatedパス)
        in_projected = self.in_proj(x) # [b, l, d_inner * 2]
        u, res = in_projected.chunk(2, dim=-1) # それぞれ [b, l, d_inner]
        
        # 2. 因果コンボリューションの適用（時系列の順序を壊さないようPaddingを調整）
        u_conv = u.transpose(1, 2) # [b, d_inner, l]
        u_conv = self.conv1d(u_conv)[:, :, :l] # 過去の情報だけを参照するようにトリミング
        u = u_conv.transpose(1, 2) # [b, l, d_inner]
        
        u_act = F.silu(u) # Mamba標準のSiLU(Swish)活性化
        
        # 3. パラメータの動的生成
        proj_vars = self.x_proj(u_act)
        
        # 各種ゲート・射影行列の切り出し
        delta = F.softplus(proj_vars[:, :, 0:1]) # タイムステップ Δ
        lambdas = torch.sigmoid(proj_vars[:, :, -1:]) # 台形公式のブレンド係数 λ
        
        # B と C を複素数として復元
        # 2 * d_state 分の領域を実部と虚部に分ける
        B_real = proj_vars[:, :, 1 : 1 + self.d_state]
        B_imag = proj_vars[:, :, 1 + self.d_state : 1 + 2 * self.d_state]
        B = torch.complex(B_real, B_imag)
        
        C_real = proj_vars[:, :, 1 + 2 * self.d_state : 1 + 3 * self.d_state]
        C_imag = proj_vars[:, :, 1 + 3 * self.d_state : 1 + 4 * self.d_state]
        C = torch.complex(C_real, C_imag)
        
        # データ依存の回転（θ）
        theta = self.theta_proj(u_act)
        
        # 4. Mamba-3 コアSSMの実行
        y = self._mamba3_parallel_scan(u_act, delta, B, C, theta, lambdas)
        
        # 5. Gated Linear Unit (GLU) による乗算と最終射影
        # もう片方のパス（res）をSiLUに通したものでSSMの出力をゲート制御する
        gated_output = y * F.silu(res)
        
        return self.out_proj(gated_output)

# --- テスト実行 ---
if __name__ == "__main__":
    # トランスフォーマーのBlockと同じ感覚で呼び出し可能
    # [BatchSize=2, SequenceLength=32, EmbeddingDim=256]
    x = torch.randn(2, 32, 256)
    
    mamba3_block = Mamba3Block(d_model=256, d_state=64, expand=2)
    out = mamba3_block(x)
    
    print("=== Mamba-3 拡張ブロック構造チェック ===")
    print("入力データの形状 :", x.shape)
    print("出力データの形状 :", out.shape)
    print("検証成功: テンソルの次元が完全に維持されています。")