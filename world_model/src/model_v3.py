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