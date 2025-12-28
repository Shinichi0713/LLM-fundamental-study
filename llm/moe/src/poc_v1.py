import torch
import torch.nn as nn

# --- 1. 標準的な Dense 層 (比較用) ---
class DenseLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. MoE (Mixture of Experts) 層 ---
class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=8):
        super().__init__()
        # 8人の専門家 (全員合わせると巨大)
        self.experts = nn.ModuleList([DenseLayer(dim) for _ in range(num_experts)])
        # どの専門家を使うか決める「ルーター」
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        # 1. ルーターが各トークンに対して最適な専門家を計算
        gate_logits = self.gate(x)
        # 2. 上位1名の専門家だけを選ぶ (Top-1 Routing)
        weights, selected_experts = torch.topk(gate_logits, 1, dim=-1)
        
        # 本来は並列処理しますが、概念理解のためループで記述
        # 実際には選ばれた専門家だけの計算しか走らないため、
        # パラメータが8倍あっても計算量はDense1回分とほぼ同じ
        output = torch.zeros_like(x)
        # (実際の実装ではここで選ばれたExpertのみを効率的に計算します)
        return output

# メモリ上のパラメータ数の違いをシミュレーション
dim = 512
dense = DenseLayer(dim)
moe = MoELayer(dim, num_experts=8)

print(f"Dense Params: {sum(p.numel() for p in dense.parameters()):,}")
print(f"MoE Total Params: {sum(p.numel() for p in moe.parameters()):,}")
print("Note: MoE has 8x params, but compute per token is nearly the same as Dense.")