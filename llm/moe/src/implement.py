import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
    

class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: [batch, seq, d_model]
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        return probs
    
class MoEFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden) for _ in range(num_experts)]
        )
        self.router = Router(d_model, num_experts)

    def forward(self, x):
        # x: [batch, seq, d_model]
        gate_probs = self.router(x)  # [B, S, E]

        # Top-1 routing
        expert_idx = torch.argmax(gate_probs, dim=-1)  # [B, S]

        output = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                expert_input = x[mask]
                expert_output = expert(expert_input)
                output[mask] = expert_output

        return output
    
class TransformerBlockMoE(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden, num_experts):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.moe_ffn = MoEFFN(d_model, d_hidden, num_experts)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.moe_ffn(x)
        x = self.norm2(x + ffn_out)

        return x