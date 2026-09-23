import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ------------------------------------------------------------------
# 1. Vision Model (VAE)
# ------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2), # (32, 31, 31)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # (64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),          # (128, 6, 6)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),         # (256, 2, 2)
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
        
        # Decoder
        self.decoder_dense = nn.Linear(latent_dim, 256 * 2 * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 * 2 * 2, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=6, stride=2),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        d = self.decoder_dense(z).view(-1, 256 * 2 * 2, 1, 1)
        recon_x = self.decoder(d)
        return recon_x, mu, logvar, z


# ------------------------------------------------------------------
# 2. Memory Model (MDN-RNN)
# ------------------------------------------------------------------
class MDNRNN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=256, num_gaussians=5):
        super(MDNRNN, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        
        self.rnn = nn.LSTMCell(latent_dim + action_dim, hidden_dim)
        
        # GMM Outputs: pi (weights), mu (means), sigma (stds) for next z
        self.fc_pi = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        
        # Reward prediction
        self.fc_reward = nn.Linear(hidden_dim, 1)

    def forward(self, z, action, hidden_state):
        # z: (batch, latent_dim), action: (batch, action_dim)
        rnn_in = torch.cat([z, action], dim=-1)
        h, c = self.rnn(rnn_in, hidden_state)
        
        pi = self.fc_pi(h).view(-1, self.num_gaussians, self.latent_dim)
        pi = F.softmax(pi, dim=1)
        
        mu = self.fc_mu(h).view(-1, self.num_gaussians, self.latent_dim)
        sigma = torch.exp(self.fc_sigma(h)).view(-1, self.num_gaussians, self.latent_dim)
        
        reward = self.fc_reward(h)
        return (pi, mu, sigma), reward, (h, c)


# ------------------------------------------------------------------
# 3. Controller (Policy)
# ------------------------------------------------------------------
class Controller(nn.Module):
    def __init__(self, latent_dim, hidden_dim, action_dim):
        super(Controller, self).__init__()
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z, h):
        # z: VAE latent, h: RNN hidden state
        state = torch.cat([z, h], dim=-1)
        action = torch.tanh(self.fc(state)) # 連続値行動空間 (-1, 1)
        return action


# ------------------------------------------------------------------
# 4. Integrated World Model Wrapper (Dream Rollout Example)
# ------------------------------------------------------------------
class WorldModel(nn.Module):
    def __init__(self, vae, rnn, controller):
        super(WorldModel, self).__init__()
        self.vae = vae
        self.rnn = rnn
        self.controller = controller

    def dream_rollout(self, initial_z, initial_h, steps=50):
        """環境を介さず、頭の中（モデル内部）のみで未来を予測・行動決定するループ"""
        z = initial_z
        h_state = initial_h
        
        rewards = []
        sampled_zs = [z]

        for _ in range(steps):
            # 1. Controllerにより行動決定
            action = self.controller(z, h_state[0])
            
            # 2. Mモデルにより次の潜在表現の分布と報酬を予測
            (pi, mu, sigma), pred_reward, h_state = self.rnn(z, action, h_state)
            
            # 3. ガウス混合分布から次の z_next をサンプリング
            # (簡略化のため、最も確率の高いガウス成分からサンプリング)
            best_k = torch.argmax(pi, dim=1)
            batch_idx = torch.arange(z.size(0))
            
            selected_mu = mu[batch_idx, best_k[batch_idx, 0]]
            selected_sigma = sigma[batch_idx, best_k[batch_idx, 0]]
            
            eps = torch.randn_like(selected_mu)
            z = selected_mu + eps * selected_sigma
            
            rewards.append(pred_reward)
            sampled_zs.append(z)
            
        return sampled_zs, torch.stack(rewards, dim=1)


if __name__ == "__main__":
    # パラメータ設定
    BATCH_SIZE = 4
    LATENT_DIM = 32
    ACTION_DIM = 3
    HIDDEN_DIM = 256
    
    # モデル初期化
    vae = VAE(latent_dim=LATENT_DIM)
    mdn_rnn = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM)
    controller = Controller(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM)
    
    wm = WorldModel(vae, mdn_rnn, controller)
    
    # ダミー入力（64x64 画像）
    dummy_img = torch.randn(BATCH_SIZE, 3, 64, 64)
    
    # 観測画像のエンコード
    _, mu, logvar, z_0 = vae(dummy_img)
    
    # 初期隠れ状態
    h_0 = (torch.zeros(BATCH_SIZE, HIDDEN_DIM), torch.zeros(BATCH_SIZE, HIDDEN_DIM))
    
    # 「夢（Dream）」ロールアウトの実行 (50ステップ先までの世界モデル内シミュレーション)
    dream_zs, predicted_rewards = wm.dream_rollout(z_0, h_0, steps=50)
    
    print(f"Dream rollout successful.")
    print(f"Generated latent trajectories length: {len(dream_zs)}")
    print(f"Predicted rewards shape: {predicted_rewards.shape}")