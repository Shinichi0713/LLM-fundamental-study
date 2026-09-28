import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ----------------------------------------------------------------------
# 1. Vision Model (VAE Encoder & Decoder)
# ----------------------------------------------------------------------
class Encoder(nn.Module):
    """画像観測 x_t -> 潜在特徴表現へのエンコード"""
    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(256 * 2 * 2, embed_dim) # 64x64 入力を想定

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class Decoder(nn.Module):
    """潜在状態 (h_t, z_t) -> 画像観測 x_t の再構成"""
    def __init__(self, feature_dim: int, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 256 * 2 * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256 * 2 * 2, 128, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=6, stride=2), nn.Sigmoid()
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        x = self.fc(feature).view(-1, 256 * 2 * 2, 1, 1)
        return self.deconv(x)


# ----------------------------------------------------------------------
# 2. Recurrent State Space Model (RSSM Core)
# ----------------------------------------------------------------------
class RSSM(nn.Module):
    """
    確定的な状態 (h_t: GRU) と 確率的な状態 (z_t: Normal) を併用するワールドモデルの核心部
    """
    def __init__(self, action_dim: int, stoch_dim: int = 30, deter_dim: int = 200, embed_dim: int = 256):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        # Prior (事前分布 Predictor): (h_t) -> z_t の平均・分散
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, 256), nn.ReLU(),
            nn.Linear(256, 2 * stoch_dim)
        )

        # Posterior (事後分布 Representation): (h_t, embed_t) -> z_t の平均・分散
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 2 * stoch_dim)
        )

        # Recurrent Model (h_t 遷移): (h_{t-1}, z_{t-1}, a_{t-1}) -> h_t
        self.rnn_cell = nn.GRUCell(stoch_dim + action_dim, deter_dim)

    def get_dist(self, stats: torch.Tensor) -> Normal:
        """平均と標準偏差から正規分布を生成"""
        mean, std_logits = stats.chunk(2, dim=-1)
        std = F.softplus(std_logits) + 0.1
        return Normal(mean, std)

    def observe(self, embed: torch.Tensor, action: torch.Tensor, state=None):
        """学習時: 観測埋め込み embed_t と行動 a_t を受け取り、事後分布と事前分布を計算"""
        if state is None:
            state = self.init_state(embed.size(0), embed.device)
            
        h, z = state
        # GRU で確定状態 h_t を更新
        rnn_in = torch.cat([z, action], dim=-1)
        h_next = self.rnn_cell(rnn_in, h)

        # Prior & Posterior の計算
        prior_stats = self.prior_net(h_next)
        post_stats = self.post_net(torch.cat([h_next, embed], dim=-1))

        prior_dist = self.get_dist(prior_stats)
        post_dist = self.get_dist(post_stats)

        # Reparameterization Trick によるサンプリング
        z_next = post_dist.rsample()

        return (h_next, z_next), prior_dist, post_dist

    def imagine(self, action: torch.Tensor, state):
        """夢（想像）の中での予測: 観測なしで Prior から将来状態を遷移"""
        h, z = state
        rnn_in = torch.cat([z, action], dim=-1)
        h_next = self.rnn_cell(rnn_in, h)

        prior_stats = self.prior_net(h_next)
        prior_dist = self.get_dist(prior_stats)
        z_next = prior_dist.rsample()

        return (h_next, z_next), prior_dist

    def init_state(self, batch_size: int, device: torch.device):
        return (torch.zeros(batch_size, self.deter_dim, device=device),
                torch.zeros(batch_size, self.stoch_dim, device=device))


# ----------------------------------------------------------------------
# 3. Full World Model
# ----------------------------------------------------------------------
class WorldModel(nn.Module):
    def __init__(self, action_dim: int, stoch_dim: int = 30, deter_dim: int = 200):
        super().__init__()
        self.encoder = Encoder(in_channels=3, embed_dim=256)
        self.rssm = RSSM(action_dim, stoch_dim, deter_dim, embed_dim=256)
        self.decoder = Decoder(feature_dim=deter_dim + stoch_dim, out_channels=3)
        
        # 報酬予測モデル
        self.reward_pred = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor, state=None):
        """
        obs: (B, C, H, W)
        action: (B, action_dim)
        """
        embed = self.encoder(obs)
        (h, z), prior_dist, post_dist = self.rssm.observe(embed, action, state)
        
        feat = torch.cat([h, z], dim=-1)
        recon_obs = self.decoder(feat)
        pred_reward = self.reward_pred(feat)

        return (h, z), recon_obs, pred_reward, prior_dist, post_dist


# ----------------------------------------------------------------------
# 4. 動作検証コード
# ----------------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 4
    action_dim = 6 # 例: 6自由度のアクション空間
    
    # ワールドモデルの構築
    world_model = WorldModel(action_dim=action_dim)

    # ダミー観測画像 (B, C, H, W) と 行動 (B, action_dim)
    obs = torch.randn(batch_size, 3, 64, 64)
    action = torch.randn(batch_size, action_dim)

    # 1ステップ順伝播 (観測からの環境状態推定)
    state, recon_obs, pred_reward, prior_dist, post_dist = world_model(obs, action)

    # KL ダイバージェンス (事後分布を事前分布に近づける Loss)
    kl_loss = torch.distributions.kl.kl_divergence(post_dist, prior_dist).sum(-1).mean()
    recon_loss = F.mse_loss(recon_obs, obs)

    print("State (h) shape    :", state[0].shape) # (B, deter_dim)
    print("State (z) shape    :", state[1].shape) # (B, stoch_dim)
    print("Reconstructed Obs  :", recon_obs.shape) # (B, 3, 64, 64)
    print("Predicted Reward   :", pred_reward.shape) # (B, 1)
    print("KL Loss            :", kl_loss.item())
    print("Reconstruction Loss:", recon_loss.item())

    # --- ワールドモデル内部での「夢（ロールアウト）」生成例 ---
    print("\n--- Imaginary Rollout (夢の中での未来予測) ---")
    imag_state = state
    for t in range(5): # 5ステップ未来を頭の中でシミュレーション
        imag_action = torch.randn(batch_size, action_dim) # Policyが決定した行動想定
        imag_state, _ = world_model.rssm.imagine(imag_action, imag_state)
        
        imag_feat = torch.cat(imag_state, dim=-1)
        imag_reward = world_model.reward_pred(imag_feat)
        print(f"Step {t+1} imagined reward shape:", imag_reward.shape)

import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence

# 先ほど定義した WorldModel クラスを使用
# from world_model import WorldModel

# ----------------------------------------------------------------------
# 1. 時系列データ用リプレイバッファ (Replay Buffer)
# ----------------------------------------------------------------------
class SequenceReplayBuffer:
    """ワールドモデルの時系列学習に必要な (obs, action, reward) のシーケンスを保持"""
    def __init__(self, capacity: int = 1000, seq_len: int = 50):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, episode_obs, episode_actions, episode_rewards):
        """1エピソード全体の軌跡を追加 (obs: T+1, actions: T, rewards: T)"""
        self.buffer.append({
            'obs': torch.tensor(episode_obs, dtype=torch.float32),
            'action': torch.tensor(episode_actions, dtype=torch.float32),
            'reward': torch.tensor(episode_rewards, dtype=torch.float32).unsqueeze(-1)
        })

    def sample(self, batch_size: int):
        """ランダムなエピソードから固定長 seq_len のシーケンスをサンプリング"""
        batch_obs, batch_actions, batch_rewards = [], [], []

        while len(batch_obs) < batch_size:
            ep = random.choice(self.buffer)
            ep_len = len(ep['action'])
            if ep_len >= self.seq_len:
                start = random.randint(0, ep_len - self.seq_len)
                end = start + self.seq_len

                batch_obs.append(ep['obs'][start:end + 1])     # (seq_len + 1, C, H, W)
                batch_actions.append(ep['action'][start:end]) # (seq_len, action_dim)
                batch_rewards.append(ep['reward'][start:end]) # (seq_len, 1)

        # 形状変換: (B, T, ...)
        return (torch.stack(batch_obs), 
                torch.stack(batch_actions), 
                torch.stack(batch_rewards))

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------------------------
# 2. ワールドモデルの学習ループ (Trainer)
# ----------------------------------------------------------------------
class WorldModelTrainer:
    def __init__(self, model: nn.Module, lr: float = 1e-4, kl_scale: float = 1.0):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.kl_scale = kl_scale

    def train_step(self, obs_seq: torch.Tensor, action_seq: torch.Tensor, reward_seq: torch.Tensor):
        """
        obs_seq:    (B, T+1, C, H, W)
        action_seq: (B, T, action_dim)
        reward_seq: (B, T, 1)
        """
        self.model.train()
        self.optimizer.zero_grad()

        B, T, _ = action_seq.shape
        device = obs_seq.device

        state = self.model.rssm.init_state(B, device)

        recon_loss = 0.0
        reward_loss = 0.0
        kl_loss = 0.0

        # 時系列に沿って展開して損失を計算 (BPTT: Backpropagation Through Time)
        for t in range(T):
            obs_t = obs_seq[:, t]
            next_obs = obs_seq[:, t + 1]
            act_t = action_seq[:, t]
            rew_t = reward_seq[:, t]

            # 1ステップ順伝播
            state, recon_obs, pred_reward, prior_dist, post_dist = self.model(obs_t, act_t, state)

            # 1. 再構成損失 (Reconstruction Loss) - 画像復元精度
            recon_loss += F.mse_loss(recon_obs, obs_t)
            
            # 2. 報酬予測損失 (Reward Loss)
            reward_loss += F.mse_loss(pred_reward, rew_t)

            # 3. KL ダイバージェンス (Prior と Posterior の差を最小化)
            kl_t = kl_divergence(post_dist, prior_dist).sum(-1).mean()
            # Free Bits (最小KL補正) で過度な正則化を防ぐ
            kl_loss += torch.clamp(kl_t, min=1.0)

        # タイムステップ数 T で正規化
        recon_loss /= T
        reward_loss /= T
        kl_loss /= T

        # 全体損失 (Total Loss)
        total_loss = recon_loss + reward_loss + self.kl_scale * kl_loss

        total_loss.backward()
        # 勾配クリッピング (RNN の勾配爆発防止)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'reward_loss': reward_loss.item(),
            'kl_loss': kl_loss.item()
        }


# ----------------------------------------------------------------------
# 3. 潜在空間での推論・未来予測 (Inference & Imagination)
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_future(model: nn.Module, initial_obs: torch.Tensor, action_plan: torch.Tensor):
    """
    初期観測 1 枚から始めて、モデル内部の「夢」の中で将来の状況・画像・報酬を予測する

    initial_obs: (1, C, H, W) 初期観測
    action_plan: (H, action_dim) 未来に実行予定の行動計画 (H: Horizon)
    """
    model.eval()
    device = initial_obs.device
    horizon = action_plan.size(0)

    # 1. 初期観測をエンコードして初期状態を作成
    embed = model.encoder(initial_obs)
    dummy_action = torch.zeros(1, action_plan.size(-1), device=device)
    state, _, _, _ = model.rssm.observe(embed, dummy_action)

    predicted_images = []
    predicted_rewards = []

    # 2. 観測を得ずに内部モデル (Imagine) だけを回して未来を予測
    for t in range(horizon):
        act_t = action_plan[t].unsqueeze(0) # (1, action_dim)
        
        # Prior（事前分布）のみで次の隠れ状態を想像
        state, prior_dist = model.rssm.imagine(act_t, state)

        # 隠れ状態 (h_t, z_t) から画像と報酬を予測
        feat = torch.cat(state, dim=-1)
        pred_img = model.decoder(feat)
        pred_rew = model.reward_pred(feat)

        predicted_images.append(pred_img)
        predicted_rewards.append(pred_rew.item())

    return torch.cat(predicted_images, dim=0), predicted_rewards


# ----------------------------------------------------------------------
# 4. 動作検証メイン処理
# ----------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    action_dim = 6
    seq_len = 10
    batch_size = 4

    # モデルおよびバッファ、トレーナーの構築
    from world_model import WorldModel
    model = WorldModel(action_dim=action_dim).to(device)
    buffer = SequenceReplayBuffer(capacity=100, seq_len=seq_len)
    trainer = WorldModelTrainer(model, lr=1e-3)

    # --- ダミーデータ作成 & バッファ挿入 ---
    print("--- 1. ダミー軌跡データの蓄積 ---")
    for _ in range(10):
        ep_obs = torch.randn(20, 3, 64, 64)      # 20ステップの観測画像
        ep_act = torch.randn(20, action_dim)      # 行動
        ep_rew = torch.randn(20)                  # 報酬
        buffer.push(ep_obs, ep_act, ep_rew)
    print(f"バッファ内のエピソード数: {len(buffer)}")

    # --- 学習ループ実行 ---
    print("\n--- 2. ワールドモデルの学習ループ ---")
    for epoch in range(3):
        obs_batch, act_batch, rew_batch = buffer.sample(batch_size)
        metrics = trainer.train_step(
            obs_batch.to(device), 
            act_batch.to(device), 
            rew_batch.to(device)
        )
        print(f"Epoch {epoch+1} | "
              f"Total Loss: {metrics['total_loss']:.4f} | "
              f"Recon Loss: {metrics['recon_loss']:.4f} | "
              f"KL Loss: {metrics['kl_loss']:.4f}")

    # --- 未来予測 (推論) のテスト ---
    print("\n--- 3. 未来予測（イマジネーション推論）の実行 ---")
    init_obs = torch.randn(1, 3, 64, 64).to(device)
    future_actions = torch.randn(5, action_dim).to(device) # 5ステップ先の行動計画

    pred_imgs, pred_rews = predict_future(model, init_obs, future_actions)
    print(f"予測画像の形状 (Horizon=5) : {pred_imgs.shape}") # (5, 3, 64, 64)
    print(f"予測された各ステップの報酬   : {[round(r, 3) for r in pred_rews]}")