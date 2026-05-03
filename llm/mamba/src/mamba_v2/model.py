import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import AutoTokenizer

from causal_conv1d import CausalConv1d  # 因果性を保証するconv1d
from einops import rearrange  # 形状操作を簡潔にする

class CausalConv1dEquivalent(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = kernel_size - 1  # 左側のみに padding を適用

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        x: [B, C, L]
        """
        # 左側に padding を適用（F.pad で明示的に左側のみパディング）
        x_padded = F.pad(x, (self.padding, 0))  # (left, right)
        # 通常の conv1d を適用
        x_conv = self.conv(x_padded)
        # 出力を入力長 L にスライス
        x_causal = x_conv[:, :, :x.size(-1)]
        return x_causal

class Mamba2Block(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        nheads=8,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_init_range=(0.9, 0.999),
        use_mem_eff_path=False,
        chunk_size=None,
        layer_idx=None,
        **factory_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.nheads = nheads
        self.ngroups = ngroups
        self.use_mem_eff_path = use_mem_eff_path
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # head次元（簡易版では固定）
        self.headdim = self.d_inner // self.nheads
        assert self.d_inner % self.nheads == 0, "d_inner must be divisible by nheads"

        # in_proj の出力次元: [z0, x0, z, xBC, dt]
        # z0, x0: MLP的なゲート・スキップ用（簡易版では同じ次元数と仮定）
        d_mlp = self.d_inner  # 簡易版では d_mlp = d_inner と仮定
        d_in_proj = (
            d_mlp +                     # z0
            d_mlp +                     # x0
            self.d_inner +               # z
            self.d_inner +               # xBC (d_ssm + 2*ngroups*d_state)
            self.nheads                 # dt
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)

        # CausalConv1d（kernel_size=4 がMamba-2標準）
        self.conv1d = CausalConv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
        )

        self.act = nn.SiLU()

        # dt のバイアス（Mamba-2風の初期化）
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplusの逆関数
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True  # 重み減衰をかけない（Mamba-2風）

        # A の初期化（Mamba-2風）
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D（skipパラメータ）
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        # RMSNormGated の代わりに LayerNorm + ゲート（簡易版）
        self.norm = nn.LayerNorm(self.d_inner)
        self.gate_proj = nn.Linear(self.d_inner, self.d_inner)

        # 出力投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # 推論用状態キャッシュ（簡易版）
        self.conv_state = None
        self.ssm_state = None

    def discretize(self, dt, A):
        """
        dt: [B, L, H]
        A:  [H, N]
        ΔA = exp(Δt A) の簡易実装
        """
        dt = dt.unsqueeze(-1)  # [B, L, H, 1]
        A = A.unsqueeze(0).unsqueeze(0)  # [1, 1, H, N]
        dA = torch.exp(dt * A)  # [B, L, H, N]
        return dA

    def forward(self, u, inference_params=None):
        """
        u: [B, L, D]
        inference_params: 簡易版では未使用（本格実装では状態キャッシュに使う）
        """
        batch, seqlen, _ = u.shape

        # in_proj で [z0, x0, z, xBC, dt] を計算
        zxbcdt = self.in_proj(u)  # [B, L, d_in_proj]

        # 分割（簡易版では d_mlp = d_inner と仮定）
        d_mlp = self.d_inner
        split_sizes = [
            d_mlp,                     # z0
            d_mlp,                     # x0
            self.d_inner,               # z
            self.d_inner,               # xBC
            self.nheads,                # dt
        ]
        z0, x0, z, xBC, dt = torch.split(zxbcdt, split_sizes, dim=-1)

        # dt のスケーリング（softplus）
        dt = dt + self.dt_bias.view(1, 1, -1)
        dt = F.softplus(dt)

        # A の取得
        A = -torch.exp(self.A_log)  # [H]
        A = A.unsqueeze(-1).expand(-1, self.d_state)  # [H, N]

        # Conv1d による局所混合（CausalConv1d）
        xBC = xBC.transpose(1, 2)  # [B, D, L]
        xBC = self.conv1d(xBC)[:, :, :seqlen]  # 出力を入力長にスライス
        xBC = self.act(xBC)
        xBC = xBC.transpose(1, 2)  # [B, L, D]

        # xBC を [x, B, C] に分割
        # 簡易版では xBC の次元を d_inner とし、その中から分割
        d_ssm = self.d_inner - 2 * self.ngroups * self.d_state
        assert d_ssm > 0, "d_inner must be larger than 2*ngroups*d_state"
        x, BC = torch.split(xBC, [d_ssm, 2 * self.ngroups * self.d_state], dim=-1)
        B, C = torch.split(BC, [self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        # 形状を head と state に合わせる
        x = rearrange(x, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
        dt = rearrange(dt, "b l h -> b l h 1")  # [B, L, H, 1]

        # SSM scan（簡易版：forループ）
        h = torch.zeros(batch, self.nheads, self.headdim, self.d_state, device=u.device)
        outputs = []
        for t in range(seqlen):
            # 離散化: ΔA, ΔB
            dA = torch.exp(dt[:, t] * A.unsqueeze(0))  # [B, H, N]
            dB = dt[:, t]  # [B, H, 1]

            # 状態更新: h_t = ΔA_t * h_{t-1} + ΔB_t * x_t * B_t
            h = dA * h + dB * x[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)

            # 出力: y_t = h_t * C_t + D * x_t
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # [B, H, P]
            y_t = y_t + self.D.view(1, self.nheads, 1) * x[:, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, L, H, P]
        y = rearrange(y, "b l h p -> b l (h p)")  # [B, L, D_inner]

        # 正規化＋ゲート（RMSNormGated の簡易版）
        y = self.norm(y)
        gate = torch.sigmoid(self.gate_proj(y))
        y = gate * y

        # MLP的スキップ接続（z0, x0）
        if d_mlp > 0:
            z0_act = F.silu(z0)
            mlp_skip = z0_act * x0
            y = torch.cat([mlp_skip, y], dim=-1)

        # 出力投影
        out = self.out_proj(y)

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        """
        簡易版のstep関数（1トークンずつの推論用）
        本格実装では causal_conv1d_update や selective_state_update を使うが、
        ここでは概念実装として簡易に実装。
        """
        batch, seqlen, _ = hidden_states.shape
        assert seqlen == 1, "Only support decoding with 1 token at a time"

        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # [B, d_in_proj]
        d_mlp = self.d_inner
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_inner, self.d_inner, self.nheads],
            dim=-1
        )

        # Conv step（簡易版）
        if conv_state is not None:
            # 状態を更新（簡易実装）
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = xBC.transpose(1, 2)
            xBC = torch.sum(conv_state * self.conv1d.weight.view(self.d_inner, self.d_conv), dim=-1)
            xBC = self.act(xBC)
        else:
            # 初回は通常のconv1d（簡易版）
            xBC = xBC.transpose(1, 2)
            xBC = self.conv1d(xBC)[:, :, :1]
            xBC = self.act(xBC)
            xBC = xBC.transpose(1, 2)

        # xBC を [x, B, C] に分割
        d_ssm = self.d_inner - 2 * self.ngroups * self.d_state
        x, BC = torch.split(xBC, [d_ssm, 2 * self.ngroups * self.d_state], dim=-1)
        B, C = torch.split(BC, [self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        # 形状整形
        x = rearrange(x, "b (h p) -> b h p", h=self.nheads, p=self.headdim)
        B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
        dt = F.softplus(dt + self.dt_bias).unsqueeze(-1)  # [B, H, 1]

        # A の取得
        A = -torch.exp(self.A_log)  # [H]
        A = A.unsqueeze(-1).expand(-1, self.d_state)  # [H, N]

        # SSM step（簡易版）
        if ssm_state is None:
            ssm_state = torch.zeros(batch, self.nheads, self.headdim, self.d_state, device=hidden_states.device)

        dA = torch.exp(dt * A.unsqueeze(0))  # [B, H, N]
        dB = dt  # [B, H, 1]

        # 状態更新
        ssm_state = dA * ssm_state + dB * x.unsqueeze(-1) * B.unsqueeze(1)
        y = (ssm_state * C.unsqueeze(1)).sum(dim=-1)  # [B, H, P]
        y = y + self.D.view(1, self.nheads, 1) * x

        y = rearrange(y, "b h p -> b (h p)")  # [B, D_inner]

        # 正規化＋ゲート
        y = self.norm(y)
        gate = torch.sigmoid(self.gate_proj(y))
        y = gate * y

        # MLP的スキップ
        if d_mlp > 0:
            z0_act = F.silu(z0)
            mlp_skip = z0_act * x0
            y = torch.cat([mlp_skip, y], dim=-1)

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


class SimpleSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, ngroups=1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.ngroups = ngroups

        self.d_inner = d_model * expand
        self.d_ssm = self.d_inner - 2 * ngroups * d_state  # SSM 用次元

        # 入力正規化
        self.norm = nn.LayerNorm(d_model)

        # in_proj: [z0, x0, z, xBC, dt] に分割
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + 1  # dt はスカラー
        self.in_proj = nn.Linear(d_model, d_in_proj)

        # SSM 用パラメータ
        self.A = nn.Parameter(torch.randn(ngroups, d_state))
        self.D = nn.Parameter(torch.ones(self.d_ssm))

        # 因果畳み込み（SSM 前の前処理）
        self.conv = CausalConv1dEquivalent(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
        )

        # 出力射影とゲート
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        residual = x
        B, L, D = x.shape

        # 1. 入力正規化
        x_norm = self.norm(x)  # [B, L, D]

        # 2. in_proj で分割
        in_proj_out = self.in_proj(x_norm)  # [B, L, d_in_proj]
        # 分割: [z0, x0, z, xBC, dt]
        z0 = in_proj_out[:, :, :self.d_inner]                          # [B, L, d_inner]
        x0 = in_proj_out[:, :, self.d_inner:2*self.d_inner]            # [B, L, d_inner]
        z = in_proj_out[:, :, 2*self.d_inner:2*self.d_inner+self.d_ssm] # [B, L, d_ssm]
        xBC = in_proj_out[:, :, 2*self.d_inner+self.d_ssm:2*self.d_inner+self.d_ssm+2*self.ngroups*self.d_state]  # [B, L, 2*ngroups*d_state]
        dt = in_proj_out[:, :, -1:]  # [B, L, 1]

        # 3. MLP 的スキップ（z0, x0）
        mlp_skip = F.silu(z0) * x0  # [B, L, d_inner]

        # 4. SSM 用の前処理（causal conv）
        x_ssm = torch.cat([z, xBC], dim=-1)  # [B, L, d_ssm + 2*ngroups*d_state]
        x_ssm = x_ssm.transpose(1, 2)        # [B, d_ssm+..., L]
        x_ssm = self.conv(x_ssm)             # [B, d_ssm+..., L]（因果性保証）
        x_ssm = F.silu(x_ssm)
        x_ssm = x_ssm.transpose(1, 2)        # [B, L, d_ssm+...]

        # 5. SSM scan（簡易版）
        # x_ssm を [x_ssm_part, B, C] に分割
        x_ssm_part = x_ssm[:, :, :self.d_ssm]                           # [B, L, d_ssm]
        B_param = x_ssm[:, :, self.d_ssm:self.d_ssm+self.ngroups*self.d_state]  # [B, L, ngroups*d_state]
        C_param = x_ssm[:, :, self.d_ssm+self.ngroups*self.d_state:]    # [B, L, ngroups*d_state]

        # dt を適切な範囲に制限
        dt = F.softplus(dt)  # [B, L, 1]

        # SSM 状態 h の初期化
        h = torch.zeros(B, self.ngroups, self.d_state, device=x.device)  # [B, ngroups, d_state]

        # 簡易 scan（for ループ）
        y_ssm_list = []
        for t in range(L):
            # 離散化: A_bar = exp(A * dt), B_bar = (exp(A*dt) - I) / A * B
            A_bar = torch.exp(self.A.unsqueeze(0) * dt[:, t:t+1, None])  # [B, ngroups, d_state]
            B_bar = (A_bar - 1.0) / (self.A.unsqueeze(0) + 1e-8) * B_param[:, t:t+1].view(B, self.ngroups, self.d_state)  # [B, ngroups, d_state]

            # 状態更新: h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar * h + B_bar * x_ssm_part[:, t:t+1].view(B, self.d_ssm // self.ngroups, self.d_state).transpose(1, 2)  # [B, ngroups, d_state]

            # 出力: y_t = C_t * h_t + D * x_t
            y_t = torch.sum(C_param[:, t:t+1].view(B, self.ngroups, self.d_state) * h, dim=-1)  # [B, ngroups]
            y_t = y_t.view(B, 1, self.ngroups * self.d_state)  # [B, 1, ngroups*d_state]
            y_t = y_t + self.D * x_ssm_part[:, t:t+1]  # [B, 1, d_ssm]
            y_ssm_list.append(y_t)

        y_ssm = torch.cat(y_ssm_list, dim=1)  # [B, L, d_ssm]

        # 6. SSM 出力と MLP スキップを統合
        ssm_out = torch.cat([y_ssm, mlp_skip], dim=-1)  # [B, L, d_inner]
        ssm_out = self.out_proj(ssm_out)  # [B, L, D]

        # 7. ゲート付き残差
        gate = torch.sigmoid(self.gate(residual))
        out = gate * ssm_out + (1 - gate) * residual
        return out


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