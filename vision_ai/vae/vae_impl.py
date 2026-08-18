"""
vae_impl.py
畳み込みベース VAE の PyTorch 実装（Stable Diffusion 系 VAE と同じ設計思想）

構成:
- Encoder: 画像 → 潜在表現 (mu, logvar)
- Reparameterization Trick: サンプリング（勾配を通せる）
- Decoder: 潜在表現 → 画像
- Loss: MSE再構成損失 + KLダイバージェンス

潜在空間のチャネル数: 4（SD 1.x と同じ）
ダウンサンプリング率: 8（512x512 → 64x64）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ============================================================
# 1. 基本ブロック
# ============================================================

class ResnetBlock(nn.Module):
    """
    ResNet風ブロック。
    畳み込み + GroupNorm + SiLU を2回繰り返し、スキップ接続で加算。
    """
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # チャネル数が変わる場合のスキップ接続
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.skip(x) + h


class Downsample(nn.Module):
    """空間解像度を半分に（stride=2畳み込み）"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """空間解像度を2倍に（nearest neighbor + 畳み込み）"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    Self-Attention ブロック（空間方向）。
    潜在空間ではピクセル数が少ない（64x64）ので計算コストが低い。
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # [B, 3*C, H, W]
        q, k, v = qkv.chunk(3, dim=1)

        # [B, C, H, W] → [B, H*W, C]
        q = q.view(B, C, H * W).transpose(1, 2)
        k = k.view(B, C, H * W).transpose(1, 2)
        v = v.view(B, C, H * W).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(C)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, H*W, H*W]
        attn = F.softmax(attn, dim=-1)

        h = torch.bmm(attn, v)  # [B, H*W, C]
        h = h.transpose(1, 2).view(B, C, H, W)
        h = self.proj(h)
        return x + h


# ============================================================
# 2. Encoder（画像 → 潜在表現）
# ============================================================

class Encoder(nn.Module):
    """
    入力: [B, 3, H, W] の画像
    出力: [B, 2*z_channels, H//8, W//8] （mu と logvar をチャネル方向に連結）

    構造: Conv → ResNetBlocks ×3 → Downsample → ... → Attention → ResNet → Conv
    """
    def __init__(
        self,
        in_channels=3,
        ch=128,               # ベースチャネル数
        ch_mult=(1, 2, 4, 4), # 各ステージのチャネル倍率
        num_res_blocks=2,     # 各解像度でのResNetブロック数
        z_channels=4,         # 潜在空間のチャネル数
        dropout=0.0,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # 初期畳み込み
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)

        # ダウンサンプリングステージ
        self.down = nn.ModuleList()
        in_ch = ch
        for i_level in range(self.num_resolutions):
            blocks = nn.ModuleList()
            attn = nn.ModuleList()
            out_ch = ch * ch_mult[i_level]

            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(in_ch, out_ch, dropout=dropout))
                in_ch = out_ch
                # 最後の解像度で Attention を挿入
                if i_level == self.num_resolutions - 1:
                    attn.append(AttentionBlock(in_ch))

            down_block = nn.Module()
            down_block.blocks = blocks
            down_block.attn = attn

            # 最後以外は Downsample
            if i_level != self.num_resolutions - 1:
                down_block.downsample = Downsample(in_ch)
            else:
                down_block.downsample = nn.Identity()

            self.down.append(down_block)

        # ミドルブロック
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_ch, in_ch, dropout=dropout)
        self.mid.attn_1 = AttentionBlock(in_ch)
        self.mid.block_2 = ResnetBlock(in_ch, in_ch, dropout=dropout)

        # 終了ブロック
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.conv_out = nn.Conv2d(in_ch, 2 * z_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)

        # ダウンサンプリング
        for down_block in self.down:
            for block in down_block.blocks:
                h = block(h)
            for attn in down_block.attn:
                h = attn(h)
            h = down_block.downsample(h)

        # ミドル
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # 終了
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


# ============================================================
# 3. Decoder（潜在表現 → 画像）
# ============================================================

class Decoder(nn.Module):
    """
    入力: [B, z_channels, H//8, W//8] の潜在表現
    出力: [B, 3, H, W] の再構成画像
    """
    def __init__(
        self,
        out_channels=3,
        ch=128,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        z_channels=4,
        dropout=0.0,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # 計算された最終チャネル数
        in_ch = ch * ch_mult[-1]

        # 初期畳み込み
        self.conv_in = nn.Conv2d(z_channels, in_ch, kernel_size=3, padding=1)

        # ミドルブロック
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_ch, in_ch, dropout=dropout)
        self.mid.attn_1 = AttentionBlock(in_ch)
        self.mid.block_2 = ResnetBlock(in_ch, in_ch, dropout=dropout)

        # アップサンプリングステージ
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            blocks = nn.ModuleList()
            attn = nn.ModuleList()
            out_ch = ch * ch_mult[i_level]

            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(in_ch, out_ch, dropout=dropout))
                in_ch = out_ch
                # 最後の解像度で Attention
                if i_level == self.num_resolutions - 1:
                    attn.append(AttentionBlock(in_ch))

            up_block = nn.Module()
            up_block.blocks = blocks
            up_block.attn = attn

            # 最初以外は Upsample
            if i_level != 0:
                up_block.upsample = Upsample(in_ch)
            else:
                up_block.upsample = nn.Identity()

            self.up.insert(0, up_block)  # 逆順に挿入

        # 終了ブロック
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)

        # ミドル
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # アップサンプリング
        for up_block in self.up:
            for block in up_block.blocks:
                h = block(h)
            for attn in up_block.attn:
                h = attn(h)
            h = up_block.upsample(h)

        # 終了
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


# ============================================================
# 4. VAE 本体（Encoder + Reparameterization + Decoder）
# ============================================================

class VAE(nn.Module):
    """
    完全なVAEモデル。

    使い方:
        vae = VAE(z_channels=4)
        # エンコード
        z, mu, logvar = vae.encode(x)  # x: [B, 3, 256, 256]
        # デコード
        x_recon = vae.decode(z)
        # 順伝播（学習時）
        x_recon, mu, logvar = vae.forward(x)
    """
    def __init__(
        self,
        in_channels=3,
        ch=128,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        z_channels=4,
        dropout=0.0,
    ):
        super().__init__()
        self.z_channels = z_channels
        self.encoder = Encoder(
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            dropout=dropout,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            dropout=dropout,
        )

    def encode(self, x):
        """
        画像 → 潜在変数 z（再パラメータ化後）
        戻り値: z, mu, logvar
        """
        h = self.encoder(x)  # [B, 2*z_channels, H//8, W//8]
        # mu  logvar に分割
        mu, logvar = h.split(self.z_channels, dim=1)
        # 再パラメータ化トリック
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def decode(self, z):
        """
        潜在変数 → 再構成画像
        """
        return self.decoder(z)

    def forward(self, x):
        """
        学習時の順伝播。
        戻り値: 再構成画像, mu, logvar
        """
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# ============================================================
# 5. 損失関数
# ============================================================

def vae_loss(x_recon, x_target, mu, logvar, kl_weight=1.0):
    """
    VAEの損失関数。

    Args:
        x_recon:  再構成画像 [B, 3, H, W]
        x_target: 目標画像 [B, 3, H, W]
        mu:       潜在変数の平均 [B, z_channels, h, w]
        logvar:   潜在変数の対数分散 [B, z_channels, h, w]
        kl_weight: KL項の重み（β-VAEとして調整可能）

    Returns:
        loss:  合計損失
        recon: 再構成損失（MSE）
        kl:    KLダイバージェンス
    """
    # 再構成損失（ピクセル単位のMSE）
    recon = F.mse_loss(x_recon, x_target, reduction="sum") / x_target.size(0)

    # KLダイバージェンス（潜在空間の各ピクセル位置で計算）
    # mu, logvar: [B, z_channels, h, w]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_target.size(0)

    loss = recon + kl_weight * kl
    return loss, recon, kl


# ============================================================
# 6. 学習ループの例
# ============================================================

def train_vae_demo():
    """
    ダミーデータでVAEを学習するデモ。
    実際の画像データセットに置き換えて使用してください。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # モデル構築
    vae = VAE(
        in_channels=3,
        ch=64,               # 小さめに設定（デモ用）
        ch_mult=(1, 2, 4),   # 3ステージ（ダウンサンプリング3回 → 1/8）
        num_res_blocks=2,
        z_channels=4,
        dropout=0.0,
    ).to(device)

    total_params = sum(p.numel() for p in vae.parameters())
    print(f"Total parameters: {total_params:,}")

    # オプティマイザ
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    # ダミーデータセット（実際は ImageFolder などに置き換え）
    # 512x512 のランダム画像を 100枚
    dummy_images = torch.randn(100, 3, 512, 512)
    dataset = TensorDataset(dummy_images)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 学習ループ
    vae.train()
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for batch in dataloader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = vae(x)
            loss, recon, kl = vae_loss(x_recon, x, mu, logvar, kl_weight=0.001)
            loss.backward()
            optimizer.step()

            epoch_recon += recon.item()
            epoch_kl += kl.item()
            num_batches += 1

        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}: recon={avg_recon:.4f}, KL={avg_kl:.4f}, total={avg_recon + 0.001*avg_kl:.4f}")

    # 推論モードでテスト
    vae.eval()
    with torch.no_grad():
        x_test = dummy_images[:4].to(device)
        x_recon, mu, logvar = vae(x_test)
        z, _, _ = vae.encode(x_test)
        print(f"\nInput shape:  {x_test.shape}")
        print(f"Latent shape: {z.shape}  (ダウンサンプリング率: {x_test.shape[-1] // z.shape[-1]}x)")
        print(f"Recon shape:  {x_recon.shape}")

    return vae


# ============================================================
# 7. 拡散モデルとの接続（エンコード/デコードのみ）
# ============================================================

class VAEForDiffusion(nn.Module):
    """
    拡散モデル用のVAEラッパー。
    学習済みVAEのエンコーダ・デコーダのみを使用し、
    KL損失なしで画像↔潜在空間の変換を行う。
    """
    def __init__(self, vae_model, scale_factor=0.18215):
        """
        Args:
            vae_model: 学習済みVAE
            scale_factor: 潜在変数のスケーリング係数（Stable Diffusion 1.x は 0.18215）
        """
        super().__init__()
        self.encoder = vae_model.encoder
        self.decoder = vae_model.decoder
        self.scale_factor = scale_factor
        self.z_channels = vae_model.z_channels

    def encode_to_latent(self, x):
        """
        画像 → スケーリング済み潜在表現
        戻り値: z_scaled, mu, logvar
        """
        h = self.encoder(x)
        mu, logvar = h.split(self.z_channels, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z_scaled = z * self.scale_factor
        return z_scaled, mu, logvar

    def decode_from_latent(self, z_scaled):
        """
        スケーリング済み潜在表現 → 画像
        """
        z = z_scaled / self.scale_factor
        return self.decoder(z)

    def forward(self, x):
        """エンコード → デコード（エンドツーエンド）"""
        z_scaled, _, _ = self.encode_to_latent(x)
        return self.decode_from_latent(z_scaled)


# ============================================================
# 8. 実行エントリポイント
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VAE Implementation Demo")
    print("=" * 60)

    # モデル構造の確認
    vae = VAE(ch=64, ch_mult=(1, 2, 4), z_channels=4)
    total = sum(p.numel() for p in vae.parameters())
    enc_params = sum(p.numel() for p in vae.encoder.parameters())
    dec_params = sum(p.numel() for p in vae.decoder.parameters())
    print(f"Total params:     {total:,}")
    print(f"Encoder params:   {enc_params:,}")
    print(f"Decoder params:   {dec_params:,}")

    # テスト順伝播
    x = torch.randn(2, 3, 512, 512)
    x_recon, mu, logvar = vae(x)
    z, _, _ = vae.encode(x)

    print(f"\nInput:  {x.shape}")
    print(f"Latent: {z.shape}  (縮小率: {x.shape[-1] // z.shape[-1]}x)")
    print(f"Recon:  {x_recon.shape}")

    # 損失計算テスト
    loss, recon, kl = vae_loss(x_recon, x, mu, logvar, kl_weight=0.001)
    print(f"\nLoss: recon={recon.item():.4f}, KL={kl.item():.4f}, total={loss.item():.4f}")

    # 学習デモ（ダミーデータ）
    print("\n" + "=" * 60)
    print("Training Demo (dummy data)")
    print("=" * 60)
    # train_vae_demo()  # 必要に応じてコメントアウトを外して実行
