class MAPPO_Net(nn.Module):
    def __init__(self, in_channels, out_dim, is_critic=False):
        super().__init__()
        # CNN: (C, 84, 84) -> (64, 7, 7)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, out_dim)

    def forward(self, x):
        feat = self.fc(self.cnn(x))
        return self.head(feat)

def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    returns = []
    gae = 0
    # 逆順に計算
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns.insert(0, gae + values[i])
    return torch.tensor(returns)


def ppo_update(actor, critic, obs, state, actions, old_log_probs, returns, advantages):
    # 1. 現在の確率と価値を取得
    new_logits = actor(obs)
    dist = torch.distributions.Categorical(logits=new_logits)
    new_log_probs = dist.log_prob(actions)
    new_values = critic(state)
    
    # 2. PPO Clipping
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    
    # 3. Value Clipping
    critic_loss = F.mse_loss(new_values, returns)
    
    return actor_loss + 0.5 * critic_loss

import torch
import numpy as np

class SharedRolloutBuffer:
    def __init__(self, num_agents, episode_length, obs_shape, state_shape):
        self.obs = torch.zeros((episode_length, num_agents, *obs_shape))
        self.state = torch.zeros((episode_length, num_agents, *state_shape))
        self.actions = torch.zeros((episode_length, num_agents, 1))
        self.values = torch.zeros((episode_length, num_agents, 1))
        self.returns = torch.zeros((episode_length, num_agents, 1))
        self.log_probs = torch.zeros((episode_length, num_agents, 1))
        self.rewards = torch.zeros((episode_length, num_agents, 1))
        self.masks = torch.ones((episode_length, num_agents, 1))
        self.step = 0

    def insert(self, obs, state, action, value, log_prob, reward, mask):
        self.obs[self.step] = obs
        self.state[self.step] = state
        self.actions[self.step] = action
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.masks[self.step] = mask
        self.step = (self.step + 1) % self.obs.size(0)

    def compute_returns(self, next_value, gamma=0.99, gae_lambda=0.95):
        # GAE計算: ライブラリなしで再帰的に計算
        gae = 0
        for i in reversed(range(self.obs.size(0))):
            delta = self.rewards[i] + gamma * next_value * self.masks[i] - self.values[i]
            gae = delta + gamma * gae_lambda * self.masks[i] * gae
            self.returns[i] = gae + self.values[i]


import torch.nn as nn

class MAPPO_Core(nn.Module):
    def __init__(self, obs_channels, state_channels, action_dim):
        super().__init__()
        # Actor/Critic 共通のCNNバックボーンを定義可能だが、入力chが異なる
        self.actor_cnn = self._build_cnn(obs_channels)
        self.critic_cnn = self._build_cnn(state_channels)
        
        self.actor_head = nn.Linear(3136, action_dim)
        self.critic_head = nn.Linear(3136, 1)

    def _build_cnn(self, in_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, obs, state):
        # 実行時は obs, 学習時は state (Critic用)
        a_feat = self.actor_cnn(obs)
        c_feat = self.critic_cnn(state)
        return self.actor_head(a_feat), self.critic_head(c_feat)


def update_step(ac, buffer, optimizer, clip_param=0.2):
    # バッファからデータを取り出し、平坦化 (Batch)
    obs = buffer.obs.view(-1, *obs_shape)
    state = buffer.state.view(-1, *state_shape)
    actions = buffer.actions.view(-1, 1)
    old_log_probs = buffer.log_probs.view(-1, 1)
    returns = buffer.returns.view(-1, 1)
    
    # PPO損失計算
    logits, values = ac(obs, state)
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
    
    ratio = torch.exp(new_log_probs - old_log_probs)
    adv = returns - values.detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8) # 正規化
    
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1-clip_param, 1+clip_param) * adv
    
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = nn.functional.mse_loss(values, returns)
    
    total_loss = actor_loss + 0.5 * critic_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(ac.parameters(), 0.5) # 勾配爆発防止
    optimizer.step()
