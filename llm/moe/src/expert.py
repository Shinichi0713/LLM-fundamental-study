import torch
import torch.nn as nn
import torch.nn.functional as F

# デバイス設定 (CPUまたはGPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================================================================
# 1. Expert (エキスパート) クラスの定義
# ====================================================================

class Expert(nn.Module):
    """ MoEにおける個々のフィードフォワードネットワーク（FFN）を担うクラス """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # シンプルな2層FFN
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ====================================================================
# 2. MoELayer (メイン) クラスの定義
# ====================================================================

class MoELayer(nn.Module):
    """ MoEレイヤー本体: ルーティング、エキスパート選択、出力結合、損失計算を行う """
    def __init__(self, dim, num_experts, top_k, expert_hidden_dim=None):
        """
        :param dim: 入力/出力の次元数 (モデルのd_model)
        :param num_experts: エキスパートの総数
        :param top_k: 活性化するエキスパートの数
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_dim = expert_hidden_dim if expert_hidden_dim is not None else dim * 2
        
        # エキスパート群の定義
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_experts)])
        
        # ゲート（ルーター）の定義: トークンをnum_experts個のlogitに出力
        self.gate = nn.Linear(dim, num_experts)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        original_shape = x.shape
        
        # トークンレベルでの処理のため、(N_tokens, dim) にフラット化
        x = x.view(-1, original_shape[-1])
        N_tokens = x.size(0)
        
        # --- 1. ゲートによるルーティング ---
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1) # 確率重み: (N_tokens, num_experts)
        
        # --- 2. Top-K 選択 ---
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        
        # 選択されたK個の重みを再正規化 (合計が1になるように)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 最終出力を保持するテンソルを初期化
        final_output = torch.zeros_like(x)
        
        # --- 3. ロードバランシング損失の計算 (Auxiliary Loss) ---
        # aux_lossをメインの損失に加えることで、均等なエキスパート利用を促す
        
        # Expert Usage: 各トークンがどのエキスパートにルーティングされたか (N_tokens, num_experts)
        expert_usage_one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1).float()
        
        # ロードバランシング損失の計算式 (簡略版)
        expert_router_prob = gate_weights.sum(dim=0) / N_tokens
        expert_fraction_routed = expert_usage_one_hot.sum(dim=0) / N_tokens
        load_balancing_loss = (expert_router_prob * expert_fraction_routed).sum()
        
        # --- 4. Dispatch (転送) と Combination (結合) ---
        
        for k in range(self.top_k):
            expert_index = top_k_indices[:, k] # k番目に選ばれたエキスパートのID
            weight = top_k_weights[:, k]       # k番目に選ばれたエキスパートの重み
            
            for i in range(self.num_experts):
                # 現在のエキスパート i にルーティングされたトークンをマスクで選択
                mask = (expert_index == i) 
                
                if not mask.any():
                    continue
                
                # 選択された入力トークン (N_i, dim)
                expert_input = x[mask]
                
                # エキスパート i で計算
                expert_output = self.experts[i](expert_input)
                
                # ゲート重みを適用し、最終出力に加算
                weighted_output = expert_output * weight[mask].unsqueeze(1)
                final_output[mask] += weighted_output

        # 出力を元の形状に戻す: (batch_size, seq_len, dim)
        final_output = final_output.view(original_shape)
        
        return final_output, load_balancing_loss

# ====================================================================
# 3. 使用例と訓練フローへの組み込み
# ====================================================================

# 設定値
DIM = 128         # トークン埋め込み次元
NUM_EXPERTS = 8   # エキスパートの総数
TOP_K = 2         # 活性化するエキスパートの数
BATCH_SIZE = 4
SEQ_LEN = 10

# MoEレイヤーの初期化
moe_layer = MoELayer(DIM, NUM_EXPERTS, TOP_K).to(device)
optimizer = torch.optim.Adam(moe_layer.parameters(), lr=1e-4)

# ダミー入力データ (例: 4つのシーケンス、各10トークン)
dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, DIM).to(device)
dummy_target = torch.randn(BATCH_SIZE, SEQ_LEN, DIM).to(device) # ダミーターゲット

# 訓練ステップの模擬
EPOCHS = 3
print("-" * 40)
print(f"MoE Training Simulation ({EPOCHS} steps)")

for epoch in range(EPOCHS):
    moe_layer.train()
    optimizer.zero_grad()
    
    # 順伝播
    output, aux_loss = moe_layer(dummy_input)
    
    # 1. メインのタスク損失 (ここではMSEを使用)
    main_task_loss = F.mse_loss(output, dummy_target)
    
    # 2. ロードバランシング損失を結合 (係数 0.01 はハイパーパラメータ)
    MOE_LOSS_COEF = 0.01
    total_loss = main_task_loss + MOE_LOSS_COEF * aux_loss
    
    # 逆伝播
    total_loss.backward()
    optimizer.step()
    
    print(f"Step {epoch+1}: Total Loss={total_loss.item():.6f} (Main Loss={main_task_loss.item():.6f}, Aux Loss={aux_loss.item():.6f})")

print("-" * 40)
print("MoE実装と訓練シミュレーション完了。")