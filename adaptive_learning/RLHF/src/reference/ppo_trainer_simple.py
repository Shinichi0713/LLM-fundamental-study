import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplePPOTrainer:
    def __init__(self, model, optimizer, eps_clip=0.2, gamma=0.99):
        self.model = model
        self.optimizer = optimizer
        self.eps_clip = eps_clip  # 更新の制限幅
        self.gamma = gamma        # 割引率（将来の報酬の重視度）
        self.mse_loss = nn.MSELoss()

    def compute_ppo_loss(self, old_log_probs, states, actions, advantages, rewards):
        """
        PPOの核心となる損失計算
        - old_log_probs: 更新前モデルでの行動確率（対数）
        - advantages: その行動が平均よりどれだけ良かったか（アドバンテージ）
        - rewards: 実際の報酬
        """
        # 1. 現在のモデルでの行動確率と価値予測を取得
        # ※ modelは Policy (確率) と Value (価値) の両方を返す想定
        new_log_probs, state_values = self.model.evaluate(states, actions)

        # 2. 確率比 (Probability Ratio) r(t) の計算
        # exp(new - old) で比率を出す
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 3. クリップされた目的関数 (Surrogate Objective)
        # 良い行動の確率を上げたいが、上げすぎないように制限をかける
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        
        # 損失の計算（マイナスを付けて最小化問題にする）
        policy_loss = -torch.min(surr1, surr2).mean()

        # 4. 価値損失 (Value Loss)
        # 報酬予測の精度を上げるための損失
        value_loss = self.mse_loss(state_values, rewards)

        # 5. 合計損失
        return policy_loss + 0.5 * value_loss

    def step(self, transitions):
        """
        1ステップの学習実行
        """
        # transitionsからデータを展開
        states, actions, old_log_probs, advantages, rewards = transitions

        loss = self.compute_ppo_loss(old_log_probs, states, actions, advantages, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()