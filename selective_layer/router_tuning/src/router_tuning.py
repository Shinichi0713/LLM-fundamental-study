import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class MoELayerWithRouterTuning(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 入力からどのエキスパートに送るかを決定するルーター（Gating Network）
        self.router = nn.Linear(d_model, num_experts)
        
        # 複数のエキスパート層
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # [B * S, D]
        
        # 1. ルーティングロジックの計算
        router_logits = self.router(x_flat) # [B * S, Num_Experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k の選択（各トークンに対して影響力の大きい上位 k 個のエキスパートを選ぶ）
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1) # [B * S, K]
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True) # 重みの正規化
        
        # 2. ロードバランス損失（Auxiliary Loss）の計算
        # ルーターが特定の専門家に偏るのを防ぎ、均等にルーティングさせるためのチューニング項
        expert_mask = F.one_hot(topk_indices, self.num_experts).float() # [B * S, K, E]
        expert_gate_load = router_probs.mean(dim=0) # [E] (ルーティング確率の平均)
        expert_hit_load = expert_mask.sum(dim=(0, 1)) / (batch_size * seq_len * self.top_k) # [E] (実際に選ばれた割合)
        
        # スイッチトランスフォーマー等で使われるバランス損失の定式化
        load_balance_loss = self.num_experts * torch.sum(expert_gate_load * expert_hit_load)

        # 3. エキスパートへのディスパッチと重み付き結合
        final_output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            # このエキスパートが選ばれているトークンを特定
            matches = (topk_indices == i) # [B * S, K]
            if not matches.any():
                continue
            
            token_indices, k_indices = torch.where(matches)
            if len(token_indices) == 0:
                continue
                
            selected_tokens = x_flat[token_indices]
            expert_out = expert(selected_tokens)
            
            # ルーターの重みを掛けて累積加算
            weights = topk_weights[token_indices, k_indices].unsqueeze(1)
            final_output.index_add_(0, token_indices, expert_out * weights)
            
        final_output = final_output.view(batch_size, seq_len, d_model)
        return final_output, load_balance_loss

# 動作確認・トレーニングループのサンプル
if __name__ == "__main__":
    d_model = 64
    d_ff = 256
    num_experts = 4
    top_k = 2
    
    model = MoELayerWithRouterTuning(d_model, d_ff, num_experts, top_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # ダミーデータの生成 (Batch=2, SeqLen=16, Dim=64)
    x = torch.randn(2, 16, d_model)
    target = torch.randn(2, 16, d_model)
    
    # 学習ステップ
    optimizer.zero_grad()
    output, aux_loss = model(x)
    
    # メインのタスク損失（例としてMSE） + ルーターの偏りを防ぐためのペナルティ項（係数 0.01）
    task_loss = F.mse_loss(output, target)
    total_loss = task_loss + 0.01 * aux_loss
    
    total_loss.backward()
    optimizer.step()
    
    print(f"Task Loss: {task_loss.item():.4f}")
    print(f"Load Balance Auxiliary Loss: {aux_loss.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")

