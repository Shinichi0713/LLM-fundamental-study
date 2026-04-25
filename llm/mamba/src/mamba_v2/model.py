import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# ここでは GPT-2 のトークナイザを使う例（vocab_size=50257）
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = tokenizer.vocab_size

# 上で実装した MambaLM を再定義（必要に応じて別ファイルから import しても可）

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand  # 拡張後の次元

        # 入力投影（チャネル拡張）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D因果畳み込み
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=False,
        )

        # SSMパラメータ（d_inner と d_state に合わせる）
        # x_proj: d_inner -> (d_state + d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state + self.d_inner, bias=False)

        # A_log: (d_inner, d_state)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))

        # D: (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 出力投影（d_inner -> d_model）
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        residual = x

        # LayerNorm
        x = self.norm(x)

        # 入力投影
        x = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, z = x.chunk(2, dim=-1)  # 各 (B, L, d_inner)

        # 1D因果畳み込み
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :residual.size(1)]  # causal conv
        x = x.transpose(1, 2)  # (B, L, d_inner)

        # SSMパラメータ計算
        ssm_params = self.x_proj(x)  # (B, L, d_state + d_inner)
        B, C = ssm_params.split([self.d_state, self.d_inner], dim=-1)
        # B: (B, L, d_state)
        # C: (B, L, d_inner)

        # Aを対数スケールで保持
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # SSM更新（簡略化）
        # x: (B, L, d_inner)
        # A: (d_inner, d_state)
        # B: (B, L, d_state)
        # C: (B, L, d_inner)
        # D: (d_inner,)

        # Ax: (B, L, d_inner) × (d_inner, d_state) -> (B, L, d_state)
        Ax = torch.einsum("bli,id->bld", x, A)

        # Bx: (B, L, d_state) × (B, L, d_inner) -> (B, L, d_inner)
        Bx = torch.einsum("bld,bli->bli", B, x)

        # h = Ax + Bx
        # Ax を (B, L, d_inner) に拡張してから加算
        Ax_expanded = torch.einsum("bld,id->bli", Ax, A.transpose(0, 1))
        h = Ax_expanded + Bx  # (B, L, d_inner)

        # y = C h + D x
        y = torch.einsum("bli,bli->bli", C, h) + self.D * x

        # ゲート適用
        y = y * F.silu(z)

        # 出力投影
        y = self.out_proj(y)  # (B, L, d_model)

        # 残差接続
        y = y + residual

        return y


class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, d_state, d_conv, expand=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    
if __name__ == "__main__":
    # ハイパーパラメータ（例：Mamba-130M相当の小さめ設定）
    vocab_size = 50257  # GPT-2トークナイザなど
    d_model = 768
    n_layer = 12
    d_state = 16
    d_conv = 4
    expand = 2

    model = MambaLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )

    # ダミー入力（バッチサイズ2, シーケンス長128）
    input_ids = torch.randint(0, vocab_size, (2, 128))

    # 順伝播
    logits = model(input_ids)  # (2, 128, vocab_size)

    print("Logits shape:", logits.shape)
    # 出力例: Logits shape: torch.Size([2, 128, 50257])