import torch
import torch.nn as nn
from mamba_ssm import Mamba
from transformers import AutoTokenizer

class MambaLM(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size
def generate(model, prompt, max_len=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(input_ids)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0])

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, nheads=8, headdim=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.nheads = nheads
        self.headdim = headdim
        
        # Mamba-2では、dt, B, C, X, Z を一つの巨大なLinear層で一気に投影します
        # これにより計算効率が大幅に向上します
        self.d_in_proj = self.d_inner * 2 + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, self.d_in_proj, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSD (Selective State Space Duality) のためのパラメータ
        self.A_log = nn.Parameter(torch.log(torch.ones(self.nheads)))
        self.dt_bias = nn.Parameter(torch.ones(self.nheads))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        batch, seqlen, _ = x.shape

        # 1. 入力投影 (一括処理)
        projected = self.in_proj(x) # (B, L, d_in_proj)
        
        # 分割 (Mamba-2の構成要素)
        # z: ゲート, x: SSM入力, B/C: 選択行列, dt: 時間刻み
        z, x_ssm, B, C, dt = torch.split(
            projected, 
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads], 
            dim=-1
        )

        # 2. 畳み込みパス
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[:, :, :seqlen].transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        # 3. SSDアルゴリズム (簡略化した行列形式)
        # Mamba-2の本質は、SSMをAttentionのように「Head」で分けることです
        # ここでは headdim ごとに分割して処理します
        y = self.ssd_kernel(x_ssm, dt, A=self.A_log, B=B, C=C)

        # 4. ゲート結合と出力
        y = y * F.silu(z)
        return self.out_proj(y)

    def ssd_kernel(self, x, dt, A, B, C):
        """
        SSD (Selective State Space Duality) のコア。
        行列の積としてSSMを計算することで、GPUのテンソルコアを利用可能にします。
        """
        batch, seqlen, d_inner = x.shape
        # 各ヘッドへの分配
        x = rearrange(x, "b l (h p) -> b h l p", h=self.nheads) # p = headdim
        
        # A, dt の適用 (離散化)
        dt = F.softplus(dt + self.dt_bias) # (b, l, h)
        dt = rearrange(dt, "b l h -> b h l 1")
        A = -torch.exp(A).view(1, -1, 1, 1) # (1, h, 1, 1)
        
        # 簡易版SSD: 実際の実装ではここでブロック行列乗算（Semiseparable Matrix）を用います
        # ここでは論理的な流れを再現するため、スキャン形式を維持したマルチヘッド処理を行います
        dA = torch.exp(dt * A) 
        
        # 状態更新のループ (マルチヘッド並列)
        h = torch.zeros(batch, self.nheads, self.d_state, self.headdim, device=x.device)
        ys = []
        
        # B, Cの調整
        B = rearrange(B, "b l d -> b 1 l d")
        C = rearrange(C, "b l d -> b 1 l d")
        
        for t in range(seqlen):
            # 状態更新: h = dA * h + B * x
            # (b, h, d_state, headdim)
            h = dA[:, :, t:t+1] * h + B[:, :, t:t+1].transpose(-1, -2) @ x[:, :, t:t+1]
            # 出力計算: y = C * h
            y_t = C[:, :, t:t+1] @ h
            ys.append(y_t)
            
        y = torch.cat(ys, dim=2)
        return rearrange(y, "b h l p -> b l (h p)")


model = MambaLM(vocab_size=vocab_size)
print("=== BEFORE ===")
print(generate(model, "Machine learning is"))