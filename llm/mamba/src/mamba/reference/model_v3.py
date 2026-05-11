import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model) # 内部次元を拡張 (通常2倍)

        # 入力を内部次元の2倍に投影（SSMパス用とゲートパス用）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Selective SSM (先ほど実装したクラス)
        # d_model の代わりに d_inner を使用する点に注意
        self.ssm = SelectiveSSM(
            d_model=self.d_inner, 
            d_state=d_state, 
            d_conv=d_conv, 
            is_causal=True
        )

        # 最終的な投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (Batch, SeqLen, d_model)
        """
        # 1. 入力投影と分割
        # (B, L, D) -> (B, L, 2*D_inner)
        combined_proj = self.in_proj(x)
        u, z = torch.split(combined_proj, [self.d_inner, self.d_inner], dim=-1)

        # 2. SSM パス (Selective SSM)
        # 内部で CausalConv1d -> Selective Scan が行われる
        y = self.ssm(u)

        # 3. ゲートパスとの結合 (Mamba の核心的な非線形処理)
        # y: SSMの出力, z: ゲート入力
        # y * silu(z)
        y = y * F.silu(z)

        # 4. 出力投影
        out = self.out_proj(y)

        return out

class Mamba(nn.Module):
    def __init__(self, d_model, n_layers, d_state=16, expand=2):
        super().__init__()
        # 通常、各ブロックの前に RMSNorm を配置するのが標準的
        self.layers = nn.ModuleList([
            nn.ModuleList([
                RMSNorm(d_model),
                MambaBlock(d_model, d_state=d_state, expand=expand)
            ])
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)

    def forward(self, x):
        for norm, block in self.layers:
            # 残差接続 (Residual Connection)
            x = x + block(norm(x))
        return self.norm_f(x)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight
    

import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_mamba_influence(model, input_text_embeddings, tokens=None):
    """
    input_text_embeddings: (1, L, D) のテンソル
    tokens: 表示用のトークンリスト（任意）
    """
    model.eval()
    input_text_embeddings.requires_grad_(True)

    # 1. 順伝播
    output = model(input_text_embeddings)
    
    # 2. 「各時刻の出力の強さ」の合計をターゲットにする
    # 本来のアテンションに近づけるため、全チャネル・全時刻のエネルギーを計算
    loss = output.pow(2).sum()
    loss.backward()

    # 3. 勾配を取得 (Batch, SeqLen, Dim)
    # これが「どの入力トークンがどれだけ出力に影響したか」の指標（等価的なアテンション）になる
    gradients = input_text_embeddings.grad.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    
    # 4. 正規化 (0-1)
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min())

    # 5. 可視化用の行列作成 (SeqLen, SeqLen) 
    # Mambaは因果的なので、下三角行列として表現（t番目の入力はt以降にしか影響しない）
    seq_len = len(gradients)
    influence_matrix = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        # 簡易的に、i番目の入力の寄与度をi番目以降の行に配置
        influence_matrix[i:, i] = gradients[i]

    # ヒートマップの描画
    plt.figure(figsize=(10, 8))
    sns.heatmap(influence_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title("Mamba Implicit Attention (Input Influence Map)")
    plt.xlabel("Input Tokens")
    plt.ylabel("Affected Output Timesteps")
    plt.show()

    return output