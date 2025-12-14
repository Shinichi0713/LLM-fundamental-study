import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================================================================
# 1. Expert (エキスパート) クラス
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
# 2. MoELayer (メイン) クラス
# ====================================================================

class MoELayer(nn.Module):
    """ MoEレイヤー本体: ルーティング、エキスパート選択、出力結合、損失計算を行う """
    def __init__(self, dim, num_experts, top_k):
        """
        :param dim: 入力/出力の次元数 (例: 128)
        :param num_experts: エキスパートの総数 (例: 4)
        :param top_k: 活性化するエキスパートの数 (例: 2)
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # エキスパート群: num_experts個のExpertモジュールをリスト化
        self.experts = nn.ModuleList([Expert(dim, dim * 2) for _ in range(num_experts)])
        
        # ゲート（ルーター）: トークンを入力として受け取り、num_experts個のlogitを出力
        self.gate = nn.Linear(dim, num_experts)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        original_shape = x.shape
        
        # トークンレベルでの処理のため、(N_tokens, dim) にフラット化
        x = x.view(-1, original_shape[-1])
        N_tokens = x.size(0)
        
        # --- 1. ゲートによるルーティング ---
        # gate_logits: (N_tokens, num_experts)
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1) # 確率重み
        
        # --- 2. Top-K 選択 ---
        # top_k_weights: 選択されたK個のエキスパートの確率 (N_tokens, top_k)
        # top_k_indices: 選択されたK個のエキスパートのID (N_tokens, top_k)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        
        # 重みを再正規化 (Top-Kの合計が1になるように)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 最終的な出力を保持するテンソル
        final_output = torch.zeros_like(x)
        
        # --- 3. ロードバランシング損失の計算 (Auxiliary Loss) ---
        # エキスパートが均等に使われるよう促すための損失項
        
        # a) Expert Usage: 各トークンがどのエキスパートにルーティングされたか (N_tokens, num_experts)
        expert_usage_one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1).float()
        
        # b) ルーター確率の平均 (P_i = sum(Prob_i) / N_tokens)
        expert_router_prob = gate_weights.sum(dim=0) / N_tokens
        
        # c) ルーティングされたサンプルの割合 (F_i = sum(Usage_i) / N_tokens)
        expert_fraction_routed = expert_usage_one_hot.sum(dim=0) / N_tokens
        
        # ロードバランシング損失 (最小化すべき値): P_i と F_i の積の合計
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
        
        # 最終的な出力と補助損失（ロードバランシングロス）を返す
        return final_output, load_balancing_loss

# ====================================================================
# 3. 使用例
# ====================================================================

# 設定値
DIM = 128         # トークン埋め込み次元
NUM_EXPERTS = 4   # エキスパートの総数
TOP_K = 2         # 活性化するエキスパートの数
BATCH_SIZE = 2
SEQ_LEN = 10

# MoEレイヤーの初期化
moe_layer = MoELayer(DIM, NUM_EXPERTS, TOP_K).to(device)

# ダミー入力データ (例: 2つのシーケンス、各10トークン)
dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, DIM).to(device)

# 推論実行 (訓練時は aux_loss をメインの損失に加算して使用)
moe_layer.train() # 訓練モード

output, aux_loss = moe_layer(dummy_input)

print("-" * 40)
print("MoE Layer 実行結果:")
print(f"入力形状: {dummy_input.shape}")
print(f"出力形状: {output.shape}")
print(f"ロードバランシング損失 (訓練に加算すべき値): {aux_loss.item():.6f}")

# 損失計算の例
# main_task_loss = some_criterion(output, target)
# total_loss = main_task_loss + 0.01 * aux_loss # 0.01 は MoE損失の重み
