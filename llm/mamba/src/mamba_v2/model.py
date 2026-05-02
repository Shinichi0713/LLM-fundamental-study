import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import AutoTokenizer

class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.num_heads = num_heads
        assert self.d_inner % self.num_heads == 0, "d_inner must be divisible by num_heads"
        self.d_head = self.d_inner // self.num_heads  # 各headが担当する次元

        # 入力投影: [z, x_inner, dt, B, C] をまとめて出す
        total_dim = (
            self.d_inner +          # z
            self.d_inner +          # x_inner
            self.num_heads +        # dt
            self.num_heads * self.d_state +  # B
            self.num_heads * self.d_state    # C
        )
        self.in_proj = nn.Linear(d_model, total_dim, bias=False)

        # 1D depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=False,
        )

        # dt のバイアス（headごと）
        self.dt_bias = nn.Parameter(torch.randn(num_heads))

        # A は対角行列（headごとに異なる減衰係数）
        self.A_log = nn.Parameter(
            torch.log(1e-3 + torch.arange(1, num_heads + 1, dtype=torch.float32))
        )

        # 出力投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 正規化（RMSNorm がなければ LayerNorm）
        try:
            from torch.nn import RMSNorm
            self.norm = RMSNorm(d_model)
        except ImportError:
            self.norm = nn.LayerNorm(d_model)

    def discretize(self, dt, A):
        """
        dt: [B, L, H]
        A:  [H, N]  (N = d_state)
        ΔA = exp(Δt A) の簡易実装
        """
        dt = dt.unsqueeze(-1)  # [B, L, H, 1]
        A = A.unsqueeze(0).unsqueeze(0)  # [1, 1, H, N]
        dA = torch.exp(dt * A)  # [B, L, H, N]
        return dA

    def forward(self, x):
        """
        x: [B, L, D]
        """
        residual = x
        x = self.norm(x)

        batch, seq_len, _ = x.size()

        # in_proj で [z, x_inner, dt, B, C] をまとめて計算
        z_x_dt_bc = self.in_proj(x)  # [B, L, total_dim]

        # 分割する次元を明示
        split_sizes = [
            self.d_inner,           # z
            self.d_inner,           # x_inner
            self.num_heads,         # dt
            self.num_heads * self.d_state,  # B
            self.num_heads * self.d_state,  # C
        ]
        z, x_inner, dt, B, C = torch.split(z_x_dt_bc, split_sizes, dim=-1)

        # 1. Conv1d による局所混合
        x_inner = x_inner.transpose(1, 2)  # [B, D_inner, L]
        x_inner = self.conv1d(x_inner)[:, :, :seq_len]
        x_inner = x_inner.transpose(1, 2)  # [B, L, D_inner]
        x_inner = F.silu(x_inner)

        # 2. dt のスケーリングとバイアス追加
        dt = dt + self.dt_bias.view(1, 1, -1)  # [B, L, H]
        dt = F.softplus(dt)

        # 3. A の取得（headごとの対角行列）
        A = -torch.exp(self.A_log)  # [H]
        A = A.unsqueeze(-1).expand(-1, self.d_state)  # [H, N]

        # 4. 離散化: ΔA, ΔB を計算（簡易版）
        dA = self.discretize(dt, A)  # [B, L, H, N]
        dB = dt.unsqueeze(-1)  # [B, L, H, 1]

        # B, C を head と state に合わせて整形
        B = B.view(batch, seq_len, self.num_heads, self.d_state)  # [B, L, H, N]
        C = C.view(batch, seq_len, self.num_heads, self.d_state)  # [B, L, H, N]

        # x_inner を head 方向に分割（各headが d_head 次元を担当）
        # x_inner: [B, L, D_inner] -> [B, L, H, D_head]
        x_inner = x_inner.view(batch, seq_len, self.num_heads, self.d_head)

        # 5. recurrent scan（簡易実装）
        h = torch.zeros(batch, self.num_heads, self.d_state, device=x.device)  # [B, H, N]
        outputs = []
        for t in range(seq_len):
            # x_inner_t: [B, H, D_head]
            x_t = x_inner[:, t]  # [B, H, D_head]

            # 各headごとに、x_t と B_t を掛け合わせて SSM への入力を形成
            # ここでは簡易に、x_t の平均を取ってスカラーに縮約する例
            # （本来は線形変換などで d_head -> 1 に投影するのが望ましい）
            x_input = x_t.mean(dim=-1, keepdim=True)  # [B, H, 1]

            # 状態更新: h_t = ΔA_t * h_{t-1} + ΔB_t * x_input * B_t
            h = dA[:, t] * h + dB[:, t] * x_input * B[:, t]

            # 出力: y_t = h_t * C_t
            y_t = (h * C[:, t]).sum(dim=-1)  # [B, H]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, L, H]

        # head を結合して出力次元に戻す
        y = y.reshape(batch, seq_len, self.num_heads)  # [B, L, H]
        # 必要に応じて線形変換で d_inner に合わせる（簡易対応）
        if y.size(-1) != self.d_inner:
            y = nn.Linear(y.size(-1), self.d_inner, bias=False).to(y.device)(y)

        # 6. ゲート z と組み合わせて出力
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y + residual


class SimpleSSM(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.linear1 = nn.Linear(d_model, self.d_inner)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 複数のカーネルサイズを使う
        self.conv3 = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner)
        self.conv5 = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=5, padding=2, groups=self.d_inner)
        self.conv7 = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=7, padding=3, groups=self.d_inner)
        
        self.mix = nn.Linear(self.d_inner * 3, self.d_inner)  # 3つの畳み込み結果を混合
        self.linear2 = nn.Linear(self.d_inner, d_model)

        self.gate = nn.Linear(d_model, d_model)  # 残差のゲート

    def forward(self, x):
        residual = x
        x = self.norm1(x)  # 入力の正規化
        x = self.linear1(x)
        x = F.silu(x)

        # 時間方向処理（マルチスケール）
        x = x.transpose(1, 2)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat([x3, x5, x7], dim=1)  # [B, D*3, L]
        x = x.transpose(1, 2)  # [B, L, D*3]
        x = self.mix(x)        # [B, L, D]
        x = F.silu(x)

        x = self.linear2(x)
        
        # ゲートで残差の重みを調整
        gate = torch.sigmoid(self.gate(residual))
        return gate * x + (1 - gate) * residual


class MambaLikeLM(nn.Module):
    def __init__(self, vocab_size, tokenizer=None, d_model=512, n_layers=6, use_mamba_block=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.use_mamba_block = use_mamba_block

        if use_mamba_block:
            self.layers = nn.ModuleList([
                Mamba2Block(d_model) for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                SimpleSSM(d_model) for _ in range(n_layers)
            ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

    def generate(self, prompt, max_len=50, temperature=0.7, top_k=None, top_p=None):
        self.eval()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        for _ in range(max_len):
            with torch.no_grad():
                logits = self(input_ids)

            next_token_logits = logits[:, -1, :] / temperature

            # top-k / top-p フィルタリング（オプション）
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("inf")
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float("inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return self.tokenizer.decode(input_ids[0])

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, strict=True):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict, strict=strict)
        print(f"Model loaded from {path}")