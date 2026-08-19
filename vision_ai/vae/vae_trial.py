"""
simple_vae_fc.py
VAEの最もシンプルなPyTorch実装（全結合層ベース）
MNISTなどの小さな画像向け。VAEの本質を理解するためのコード。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# 1. モデル定義
# ============================================================

class SimpleVAE(nn.Module):
    """
    全結合層のみのシンプルなVAE。
    入力: フラットな画像ベクトル（例: MNIST 28x28=784）
    潜在次元: 任意（デフォルト20）
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(SimpleVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 入力 → 隠れ層
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 潜在空間のパラメータ（平均と対数分散）
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: 潜在変数 → 隠れ層 → 再構成
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """画像 → μ, logvar"""
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        再パラメータ化トリック。
        z = μ + σ × ε  （ε ~ N(0, I)）
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """潜在変数 → 再構成画像"""
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        """順伝播: 入力 → 再構成"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# ============================================================
# 2. 損失関数
# ============================================================

def loss_function(x_recon, x_original, mu, logvar):
    """
    VAEの損失 = 再構成誤差 + KLダイバージェンス

    Args:
        x_recon:     再構成画像 [B, input_dim]
        x_original:  元画像 [B, input_dim]
        mu:          潜在変数の平均 [B, latent_dim]
        logvar:      潜在変数の対数分散 [B, latent_dim]
    """
    # 再構成誤差（二値交差エントロピー）
    BCE = F.binary_cross_entropy(x_recon, x_original, reduction='sum')

    # KLダイバージェンス
    # KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


# ============================================================
# 3. 学習
# ============================================================

def train(epoch, model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # MNIST: [B, 1, 28, 28] → [B, 784]
        data = data.view(data.size(0), -1)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item() / len(data):.4f} (BCE: {bce.item() / len(data):.2f}, '
                  f'KLD: {kld.item() / len(data):.4f})')

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss


# ============================================================
# 4. テスト & 可視化
# ============================================================

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device).view(data.size(0), -1)
            recon, mu, logvar = model(data)
            loss, _, _ = loss_function(recon, data, mu, logvar)
            test_loss += loss.item()

    avg_loss = test_loss / len(test_loader.dataset)
    print(f'====> Test set loss: {avg_loss:.4f}')
    return avg_loss


def visualize_reconstruction(model, test_loader, device, save_path="vae_recon.png"):
    """元画像と再構成画像を並べて表示"""
    import matplotlib.pyplot as plt
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:8].to(device).view(8, -1)
        recon, _, _ = model(data)

        data = data.cpu().view(8, 1, 28, 28)
        recon = recon.cpu().view(8, 1, 28, 28)

        fig, axes = plt.subplots(2, 8, figsize=(12, 3))
        for i in range(8):
            axes[0, i].imshow(data[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
        axes[0, 0].set_title("Original", fontsize=10, loc='left')
        axes[1, 0].set_title("Reconstructed", fontsize=10, loc='left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"Saved: {save_path}")


def generate_from_random(model, device, n=8, save_path="vae_gen.png"):
    """ランダムな潜在変数から新しい画像を生成"""
    import matplotlib.pyplot as plt
    model.eval()
    with torch.no_grad():
        # 標準正規分布からサンプリング
        z = torch.randn(n, model.latent_dim).to(device)
        samples = model.decode(z).cpu().view(n, 1, 28, 28)

        fig, axes = plt.subplots(1, n, figsize=(12, 2))
        for i in range(n):
            axes[i].imshow(samples[i].squeeze(), cmap='gray')
            axes[i].axis('off')
        plt.suptitle("Generated from random latent vectors", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"Saved: {save_path}")


# ============================================================
# 5. メイン
# ============================================================

if __name__ == "__main__":
    # デバイス
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ハイパーパラメータ
    BATCH_SIZE = 128
    EPOCHS = 10
    LATENT_DIM = 20

    # MNISTデータセット
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1] に正規化
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # モデル
    model = SimpleVAE(input_dim=784, hidden_dim=400, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Model: SimpleVAE(latent_dim={LATENT_DIM})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 学習ループ
    for epoch in range(1, EPOCHS + 1):
        train(epoch, model, train_loader, optimizer, device)
        test(model, test_loader, device)

    # 可視化
    print("\n--- Reconstruction ---")
    visualize_reconstruction(model, test_loader, device, save_path="/home/user/outputs/vae_recon.png")

    print("\n--- Generation ---")
    generate_from_random(model, device, n=8, save_path="/home/user/outputs/vae_gen.png")
