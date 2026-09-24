import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, KullbackLeibler

class RSSM(nn.Module):
    def __init__(self, action_dim=3, stochastic_dim=30, deterministic_dim=200, embed_dim=1024):
        super(RSSM, self).__init__()
        self.stoch_dim = stochastic_dim
        self.det_dim = deterministic_dim
        
        # 1. Recurrent Model (決定論的遷移: GRUCell)
        # Input: [stochastic_state_(t-1), action_(t-1)]
        self.cell = nn.GRUCell(stochastic_dim + action_dim, deterministic_dim)
        
        # 2. Transition Prior (環境の観察なしで次のzを予測)
        # Input: deterministic_state_t -> Prior (mu, std)
        self.fc_prior = nn.Sequential(
            nn.Linear(deterministic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, stochastic_dim * 2) # mean and log_std
        )
        
        # 3. Representation Posterior (観測画像情報を取り込んで正確なzを推定)
        # Input: [deterministic_state_t, image_embed_t] -> Posterior (mu, std)
        self.fc_posterior = nn.Sequential(
            nn.Linear(deterministic_dim + embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, stochastic_dim * 2) # mean and log_std
        )

    def _get_dist(self, stats):
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(log_std) + 0.1 # 数値安定化のためのオフセット
        return Normal(mean, std)

    def observe(self, embed, action, state=None):
        """
        学習時: 実際の観測系列(embed)を受け取り、Posterior(事後分布)とPrior(事前分布)の双方を追跡
        """
        batch_size, seq_len, _ = embed.shape
        if state is None:
            state = self.init_state(batch_size, embed.device)
            
        det, stoch = state
        
        prior_dists = []
        post_dists = []
        post_samples = []
        dets = []

        for t in range(seq_len):
            # 1. 決定論的状態 h_t の更新
            x = torch.cat([stoch, action[:, t]], dim=-1)
            det = self.cell(x, det)
            
            # 2. 事前分布 P(z_t | h_t) の予測
            prior_dist = self._get_dist(self.fc_prior(det))
            
            # 3. 事後分布 Q(z_t | h_t, e_t) の推定 (観測埋め込み e_t を活用)
            post_input = torch.cat([det, embed[:, t]], dim=-1)
            post_dist = self._get_dist(self.fc_posterior(post_input))
            
            # Reparameterization trick によるサンプリング
            stoch = post_dist.rsample()
            
            prior_dists.append(prior_dist)
            post_dists.append(post_dist)
            post_samples.append(stoch)
            dets.append(det)

        # (batch, seq_len, dim) に整形
        dets = torch.stack(dets, dim=1)
        post_samples = torch.stack(post_samples, dim=1)
        
        return (dets, post_samples), (prior_dists, post_dists)

    def imagine(self, action, state):
        """
        夢（Imagination）モード: 観測画像なしでPriorのみを使って未来の状態系列を生成
        """
        det, stoch = state
        seq_len = action.shape[1]
        
        imag_dets = []
        imag_stochs = []

        for t in range(seq_len):
            x = torch.cat([stoch, action[:, t]], dim=-1)
            det = self.cell(x, det)
            
            prior_dist = self._get_dist(self.fc_prior(det))
            stoch = prior_dist.rsample() # 観測なしで自己予測からサンプリング
            
            imag_dets.append(det)
            imag_stochs.append(stoch)

        return torch.stack(imag_dets, dim=1), torch.stack(imag_stochs, dim=1)

    def init_state(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.det_dim, device=device),
            torch.zeros(batch_size, self.stoch_dim, device=device)
        )


class RSSMWorldModel(nn.Module):
    def __init__(self, action_dim=3, stochastic_dim=30, deterministic_dim=200):
        super(RSSMWorldModel, self).__init__()
        
        # 観測画像のエンコーダ & デコーダ
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 1024)
        )
        
        self.rssm = RSSM(action_dim, stochastic_dim, deterministic_dim, embed_dim=1024)
        
        # 状態 (det + stoch) から観測画像を再構成するデコーダ
        self.decoder = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, 128 * 6 * 6),
            nn.Unflatten(1, (128, 6, 6)),
            nn.ConvTranspose2d(128, 64, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, 2), nn.Sigmoid()
        )
        
        # 報酬予測ネットワーク
        self.reward_predictor = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def compute_loss(self, obs_seq, action_seq, reward_seq):
        """
        obs_seq: (batch, seq_len, 3, 64, 64)
        action_seq: (batch, seq_len, action_dim)
        reward_seq: (batch, seq_len, 1)
        """
        batch_size, seq_len, c, h, w = obs_seq.shape
        
        # 1. 全タイムステップの画像をエンコード
        flat_obs = obs_seq.view(batch_size * seq_len, c, h, w)
        embed = self.encoder(flat_obs).view(batch_size, seq_len, -1)
        
        # 2. RSSMの系列処理 (Observe)
        (dets, post_samples), (prior_dists, post_dists) = self.rssm.observe(embed, action_seq)
        
        # 3. 状態表現の結合 [det, stoch]
        feat = torch.cat([dets, post_samples], dim=-1)
        
        # 4. 観測画像の再構成損失 (Reconstruction Loss)
        flat_feat = feat.view(batch_size * seq_len, -1)
        recon_obs = self.decoder(flat_feat).view(batch_size, seq_len, c, h, w)
        recon_loss = F.mse_loss(recon_obs, obs_seq, reduction='sum') / batch_size
        
        # 5. 報酬の予測損失
        pred_rewards = self.reward_predictor(flat_feat).view(batch_size, seq_len, 1)
        reward_loss = F.mse_loss(pred_rewards, reward_seq, reduction='sum') / batch_size
        
        # 6. KLダイバージェンス損失 (PriorとPosteriorの距離を近づける)
        kl_loss = 0.0
        for t in range(seq_len):
            kl = torch.distributions.kl.kl_divergence(post_dists[t], prior_dists[t])
            # KL Balancing / Clipping (自由度確保のためのテクニック)
            kl_loss += torch.mean(torch.max(kl.sum(dim=-1), torch.tensor(3.0, device=obs_seq.device)))
            
        total_loss = recon_loss + reward_loss + kl_loss
        return total_loss, recon_loss.item(), reward_loss.item(), kl_loss.item()


if __name__ == "__main__":
    BATCH_SIZE = 4
    SEQ_LEN = 10
    ACTION_DIM = 3
    
    model = RSSMWorldModel(action_dim=ACTION_DIM)
    
    # ダミー系列データ (Batch, SeqLen, Channel, Height, Width)
    dummy_obs = torch.randn(BATCH_SIZE, SEQ_LEN, 3, 64, 64)
    dummy_actions = torch.randn(BATCH_SIZE, SEQ_LEN, ACTION_DIM)
    dummy_rewards = torch.randn(BATCH_SIZE, SEQ_LEN, 1)
    
    # 損失計算
    loss, recon, reward, kl = model.compute_loss(dummy_obs, dummy_actions, dummy_rewards)
    
    print(f"RSSM World Model Forward Pass Complete.")
    print(f"Total Loss: {loss.item():.4f} (Recon: {recon:.2f}, Reward: {reward:.2f}, KL: {kl:.2f})")