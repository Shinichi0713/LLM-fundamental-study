import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # 入力投影（チャネル拡張）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 短い因果畳み込み（local convolution）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,  # causal padding
            bias=False,
        )

        # SSMパラメータ（簡略化のため線形層で代用）
        self.x_proj = nn.Linear(self.d_inner, d_state + d_model, bias=False)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        # 出力投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 正規化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        residual = x

        # LayerNorm
        x = self.norm(x)

        # 入力投影
        x = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, z = x.chunk(2, dim=-1)  # ゲート用とSSM用に分割

        # 1D因果畳み込み
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :residual.size(1)]  # causal conv
        x = x.transpose(1, 2)  # (B, L, d_inner)

        # SSM部分（簡略化：実際はselective scan＋SSDアルゴリズム）
        ssm_params = self.x_proj(x)  # (B, L, d_state + d_model)
        B, C = ssm_params.split([self.d_state, self.d_model], dim=-1)

        # Aを対数スケールで保持
        A = -torch.exp(self.A_log)  # (d_model, d_state)

        # 簡略化のため、単純な線形変換＋活性化で代用
        h = torch.einsum("bln,nd->bld", x, A) + torch.einsum("bln,bln->bld", B, x)
        y = torch.einsum("bld,nd->bln", h, C) + self.D * x

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

        # トークン埋め込み
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Mambaブロックのスタック
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layer)
        ])

        # 最終LayerNorm
        self.ln_f = nn.LayerNorm(d_model)

        # LMヘッド（言語モデルヘッド）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 埋め込みとLMヘッドの重みを共有（任意）
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)

        # 埋め込み
        x = self.embedding(input_ids)  # (B, L, d_model)

        # Mambaブロックを順に適用
        for block in self.blocks:
            x = block(x)

        # 最終LayerNorm
        x = self.ln_f(x)

        # LMヘッドで次トークンロジットを計算
        logits = self.lm_head(x)  # (B, L, vocab_size)

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