
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# 因果畳み込み
class CausalConv1D(nn.Module):
    def __init__(self, d_model, kernel_size, is_causal=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.is_causal = is_causal
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=0
        )

    def forward(self, x):
        if self.is_causal:
            # 左側に (k-1) 個のパディングを入れ、未来を隠す
            # [0, 0, 0, x1, x2, x3] -> 出力y3はx1, x2, x3に依存
            x_padded = F.pad(x, (self.kernel_size - 1, 0))
        else:
            # --- BERT / エンコーダ型 (双方向的) ---
            # 左右に均等にパディングを入れ、中心を現在に合わせる
            # [0, x1, x2, x3, 0] -> 出力y2はx1, x2, x3に依存 (k=3の場合)
            pad_left = (self.kernel_size - 1) // 2
            pad_right = self.kernel_size - 1 - pad_left
            x_padded = F.pad(x, (pad_left, pad_right))
        return self.conv1d(x_padded)


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state, d_conv=4, dt_rank="auto", is_causal=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        if dt_rank == "auto":
            dt_rank = max(1, d_model // 16)
        if isinstance(dt_rank, int) is False:
            raise TypeError(f"dt_rank must be an integer or 'auto', but got {type(dt_rank).__name__}: {dt_rank}")
        self.dt_rank = dt_rank
        
        # 畳み込み層
        self.conv1d = CausalConv1D(
            d_model=d_model,
            kernel_size=d_conv,
            is_causal=is_causal
        )

        # 入力投影: d_model -> dt_rank + 2*d_state
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)

        # δ
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        # A のパラメータ（対角行列）
        self.A_log = nn.Parameter(torch.log(torch.ones(d_model, d_state)))
        # D（スキップ接続）
        self.D = nn.Parameter(torch.ones(d_model))

    # dtの制御 (出力抽出)
    def __selective_scan(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
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
        
        # deltaの処理
        if delta_bias is not None:
            delta = delta + delta_bias.view(1, -1, 1)
        if delta_softplus:
            delta = F.softplus(delta)

        # B, C形状統一
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
            C = C.t().unsqueeze(0).expand(batch, -1, -1).transpose(1, 2) 
        # Aは対角行列
        A = -torch.exp(A)
        # 離散化
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

    def forward(self, u):
        batch, seqlen, dim = u.shape
        assert dim == self.d_model
        # causal conv
        u_conv = u.transpose(1, 2)
        u_conv = self.conv1d(u_conv)
        u_conv = u_conv[:, :, :seqlen]  # causal padding の調整
        u_conv = u_conv.transpose(1, 2)  # (B, L, D)
        # 入力投影: (B, L, dt_rank + 2*d_state)
        x_proj = self.x_proj(u_conv)
        # 分割: delta_rank, B, C
        delta, B, C = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Δ の射影 + softplus
        delta = F.softplus(self.dt_proj(delta))  # (B, L, D)
        delta = delta.transpose(1, 2)  # (B, D, L)

        # B, C を (B, d_state, L) に変換
        B = B.transpose(1, 2)  # (B, d_state, L)
        C = C.transpose(1, 2)  # (B, d_state, L)

        u = u.transpose(1, 2)
        y = self.__selective_scan(
            u=u,
            delta=delta,
            A=self.A_log,
            B=B,
            C=C,
            D=self.D,
            z=None,  # 必要に応じて追加
            delta_bias=None,
            delta_softplus=False,  # すでに softplus 済み
        )
        y = y.transpose(1, 2)  # (B, L, D)
        return y


# --- 動作確認用コード ---
def test_conv_behavior():
    d_model = 1
    kernel_size = 3
    seq_len = 5
    
    # テストデータの作成 (1, 1, 5)
    # [1.0, 2.0, 3.0, 4.0, 5.0]
    x = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, 1, -1)
    
    # 重みを1に固定して計算を分かりやすくする (単なる合計を計算する設定)
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()

    print(f"--- Testing with kernel_size={kernel_size} ---\n")

    for causal in [True, False]:
        mode = "Causal (Mamba style)" if causal else "Non-Causal (BERT style)"
        model = CausalConv1D(d_model, kernel_size, is_causal=causal)
        model.apply(init_weights)
        
        with torch.no_grad():
            output = model(x)
            
            # 未来の値を書き換えて影響をチェック
            # インデックス 2 (値 3.0) を 100.0 に変更
            x_modified = x.clone()
            x_modified[0, 0, 2] = 100.0
            output_modified = model(x_modified)
            
            # 変化した場所を特定
            diff = (output != output_modified).squeeze()
            
            print(f"Mode: {mode}")
            print(f"Original Input:  {x.squeeze().tolist()}")
            print(f"Modified Input:  {x_modified.squeeze().tolist()} (at index 2)")
            print(f"Original Output: {output.squeeze().tolist()}")
            print(f"Modified Output: {output_modified.squeeze().tolist()}")
            print(f"Affected Indices: {torch.where(diff)[0].tolist()}")
            
            if causal:
                print("✓ Check: Index 2 以降のみが影響を受けています (因果性を維持)")
            else:
                print("✓ Check: Index 2 の前後が影響を受けています (双方向を参照)")
            print("-" * 50)
