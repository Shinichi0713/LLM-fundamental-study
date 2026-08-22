#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vae_full.py
===========
CIFAR-10 / MNIST 対応の本格畳み込みVAE実装（PyTorch）

【含まれる機能】
- ResNet-style Encoder / Decoder（GroupNorm + SiLU）
- Self-Attention（ボトルネック層）
- EMA（Exponential Moving Average）モデル
- 混合精度学習（torch.cuda.amp）
- チェックポイント保存・再開（latest / best）
- TensorBoard ロギング（Loss / KL / Recon / LR / FID）
- 画像サンプリング（再構成・ランダム生成・潜在空間補間）
- FID スコア計算（InceptionV3 特徴量ベース・簡易版）
- コマンドライン引数で全設定変更可能

【推奨実行例】
  # CIFAR-10 で学習（デフォルト）
  python vae_full.py --dataset cifar10 --epochs 100 --batch_size 128 --z_channels 4

  # MNIST で学習
  python vae_full.py --dataset mnist --image_size 28 --channels 64 --z_channels 2

  # 学習再開
  python vae_full.py --resume ./checkpoints/vae_best.pt
"""

import argparse
import os
import time
import math
from copy import deepcopy
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils
from torchvision.models import inception_v3

# ------------------------------------------------------------------------------
# 0. ユーティリティ：FID 計算（scipy が無くても動作するフォールバック付き）
# ------------------------------------------------------------------------------

try:
    from scipy import linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[Warning] scipy not found. FID calculation will use numpy fallback.")


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Frechet Inception Distance (FID) を計算。
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

    diff = mu1 - mu2

    if HAS_SCIPY:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    else:
        # numpy fallback（固有値分解で平方根）
        eigvals, eigvecs = np.linalg.eigh(sigma1.dot(sigma2))
        eigvals = np.maximum(eigvals, eps)
        covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ np.linalg.inv(eigvecs)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# ------------------------------------------------------------------------------
# 1. モデル構成要素
# ------------------------------------------------------------------------------

class ResnetBlock(nn.Module):
    """ResNetブロック（GroupNorm → SiLU → Conv × 2 + Skip）"""
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """空間方向 Self-Attention（解像度が低いボトルネックで使用）"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.view(B, C, H * W).transpose(1, 2)
        k = k.view(B, C, H * W).transpose(1, 2)
        v = v.view(B, C, H * W).transpose(1, 2)
        scale = 1.0 / math.sqrt(C)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        h = torch.bmm(attn, v).transpose(1, 2).view(B, C, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ------------------------------------------------------------------------------
# 2. Encoder / Decoder / VAE
# ------------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    畳み込みEncoder。
    入力: [B, 3, H, W]  →  出力: [B, 2*z_channels, H//8, W//8]
    """
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,2,4,4),
                 num_res_blocks=2, z_channels=4, dropout=0.0):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down = nn.ModuleList()
        in_ch = ch
        for i_level in range(len(ch_mult)):
            blocks = nn.ModuleList()
            attn = nn.ModuleList()
            out_ch = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
                if i_level == len(ch_mult) - 1:
                    attn.append(AttentionBlock(in_ch))
            down = nn.Module()
            down.blocks = blocks
            down.attn = attn
            down.downsample = Downsample(in_ch) if i_level != len(ch_mult)-1 else nn.Identity()
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_ch, in_ch, dropout)
        self.mid.attn_1 = AttentionBlock(in_ch)
        self.mid.block_2 = ResnetBlock(in_ch, in_ch, dropout)
        self.norm_out = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.conv_out = nn.Conv2d(in_ch, 2 * z_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for down in self.down:
            for block in down.blocks:
                h = block(h)
            for attn in down.attn:
                h = attn(h)
            h = down.downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    """
    畳み込みDecoder。
    入力: [B, z_channels, H//8, W//8]  →  出力: [B, 3, H, W]
    """
    def __init__(self, out_channels=3, ch=128, ch_mult=(1,2,4,4),
                 num_res_blocks=2, z_channels=4, dropout=0.0):
        super().__init__()
        in_ch = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, in_ch, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_ch, in_ch, dropout)
        self.mid.attn_1 = AttentionBlock(in_ch)
        self.mid.block_2 = ResnetBlock(in_ch, in_ch, dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(len(ch_mult))):
            blocks = nn.ModuleList()
            attn = nn.ModuleList()
            out_ch = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
                if i_level == len(ch_mult) - 1:
                    attn.append(AttentionBlock(in_ch))
            up = nn.Module()
            up.blocks = blocks
            up.attn = attn
            up.upsample = Upsample(in_ch) if i_level != 0 else nn.Identity()
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for up in self.up:
            for block in up.blocks:
                h = block(h)
            for attn in up.attn:
                h = attn(h)
            h = up.upsample(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return torch.sigmoid(h)


class VAE(nn.Module):
    def __init__(self, image_channels=3, ch=128, ch_mult=(1,2,4,4),
                 num_res_blocks=2, z_channels=4, dropout=0.0):
        super().__init__()
        self.z_channels = z_channels
        self.encoder = Encoder(image_channels, ch, ch_mult, num_res_blocks, z_channels, dropout)
        self.decoder = Decoder(image_channels, ch, ch_mult, num_res_blocks, z_channels, dropout)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.split(self.z_channels, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar


# ------------------------------------------------------------------------------
# 3. EMA（Exponential Moving Average）
# ------------------------------------------------------------------------------

class EMA:
    """モデルパラメータの指数移動平均。生成品質向上に有効。"""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                s.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ------------------------------------------------------------------------------
# 4. 損失関数
# ------------------------------------------------------------------------------

def vae_loss(x_recon, x_target, mu, logvar, kl_weight=1.0):
    recon = F.mse_loss(x_recon, x_target, reduction="sum") / x_target.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_target.size(0)
    return recon + kl_weight * kl, recon, kl


# ------------------------------------------------------------------------------
# 5. データ読み込み
# ------------------------------------------------------------------------------

def get_dataloader(dataset_name="cifar10", image_size=32, batch_size=128, num_workers=4):
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1ch → 3ch
        ])
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    else:  # cifar10
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]) if image_size == 32 else transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, test_loader


# ------------------------------------------------------------------------------
# 6. FID 計算（InceptionV3 特徴量）
# ------------------------------------------------------------------------------

class InceptionFeatureExtractor(nn.Module):
    """InceptionV3 の最終プーリング層出力を特徴量として抽出。"""
    def __init__(self, device):
        super().__init__()
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity()
        inception.eval()
        self.model = inception.to(device)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        # x: [B, 3, 32, 32] → Inceptionは299x299を期待
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # InceptionV3 forward は logits を返すが、aux を無効化し fc=Identity にしている
        features = self.model(x)
        if isinstance(features, tuple):
            features = features[0]
        return features


def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(real_images, fake_images, extractor, device, batch_size=64):
    """
    real_images, fake_images: Tensor [N, 3, H, W] in [0,1]
    """
    def get_feats(imgs):
        feats = []
        for i in range(0, imgs.size(0), batch_size):
            batch = imgs[i:i+batch_size].to(device)
            f = extractor(batch)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    real_feats = get_feats(real_images)
    fake_feats = get_feats(fake_images)
    mu1, sigma1 = compute_statistics(real_feats)
    mu2, sigma2 = compute_statistics(fake_feats)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


# ------------------------------------------------------------------------------
# 7. 可視化
# ------------------------------------------------------------------------------

def save_image_grid(images, path, nrow=8):
    """images: Tensor [N, C, H, W] in [0,1]"""
    utils.save_image(images, path, nrow=nrow, padding=2)


def interpolate_latent(vae, device, n_rows=8, n_cols=16, save_path="interp.png"):
    """潜在間の2点間補間グリッドを生成。"""
    vae.eval()
    with torch.no_grad():
        z1 = torch.randn(1, vae.z_channels, 4, 4, device=device)
        z2 = torch.randn(1, vae.z_channels, 4, 4, device=device)
        alphas = torch.linspace(0, 1, n_cols, device=device).view(n_cols, 1, 1, 1, 1)
        z_interp = (1 - alphas) * z1 + alphas * z2  # [n_cols, 1, C, 4, 4]
        z_interp = z_interp.squeeze(1)  # [n_cols, C, 4, 4]
        # 複数行（異なるペア）
        all_rows = []
        for _ in range(n_rows):
            z1 = torch.randn(1, vae.z_channels, 4, 4, device=device)
            z2 = torch.randn(1, vae.z_channels, 4, 4, device=device)
            alphas = torch.linspace(0, 1, n_cols, device=device).view(n_cols, 1, 1, 1, 1)
            z_row = (1 - alphas) * z1 + alphas * z2
            z_row = z_row.squeeze(1)
            x_row = vae.decode(z_row)
            all_rows.append(x_row)
        grid = torch.cat(all_rows, dim=0)
        save_image_grid(grid, save_path, nrow=n_cols)
    print(f"[Saved interpolation] {save_path}")


# ------------------------------------------------------------------------------
# 8. 学習 / 評価
# ------------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, args, ema=None):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    start = time.time()

    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.amp):
            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, args.kl_weight)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        if batch_idx % args.log_interval == 0:
            print(f"  Epoch {epoch:3d} [{batch_idx:4d}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f})")

    n = len(dataloader)
    avg_loss = total_loss / n
    avg_recon = total_recon / n
    avg_kl = total_kl / n
    elapsed = time.time() - start
    print(f"====> Epoch {epoch} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Time: {elapsed:.1f}s")
    return avg_loss, avg_recon, avg_kl


def evaluate(model, dataloader, device, args, epoch=0, save_path=None, fixed_batch=None):
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                recon, mu, logvar = model(data)
                loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, args.kl_weight)
            total_loss += loss.item() * data.size(0)
            total_recon += recon_loss.item() * data.size(0)
            total_kl += kl_loss.item() * data.size(0)

    n = len(dataloader.dataset)
    avg_loss = total_loss / n
    avg_recon = total_recon / n
    avg_kl = total_kl / n
    print(f"====> Test       | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

    # 再構成サンプル保存
    if save_path and fixed_batch is not None:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                recon_batch, _, _ = model(fixed_batch[:64])
            comparison = torch.cat([fixed_batch[:8], recon_batch[:8]], dim=0)
            save_image_grid(comparison, save_path, nrow=8)
        print(f"[Saved reconstruction] {save_path}")

    return avg_loss


# ------------------------------------------------------------------------------
# 9. チェックポイント
# ------------------------------------------------------------------------------

def save_checkpoint(model, ema, optimizer, scaler, epoch, best_loss, args, is_best=False):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
        "args": vars(args),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    path = os.path.join(args.checkpoint_dir, "vae_latest.pt")
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(args.checkpoint_dir, "vae_best.pt")
        torch.save(state, best_path)
        print(f"[Saved best checkpoint] {best_path}")


def load_checkpoint(path, model, ema, optimizer, scaler, device):
    print(f"[Resuming from] {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if "ema" in state and ema is not None:
        ema.load_state_dict(state["ema"])
    if "scaler" in state and scaler is not None:
        scaler.load_state_dict(state["scaler"])
    return state["epoch"], state.get("best_loss", float("inf"))


# ------------------------------------------------------------------------------
# 10. メイン
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full Convolutional VAE Training")
    # データ
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    # モデル
    parser.add_argument("--channels", type=int, default=128, help="base channel width")
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1,2,4,4])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--z_channels", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--kl_weight", type=float, default=1.0, help="beta-VAE weight")
    # 学習
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true", help="mixed precision training")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    # ログ・保存
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--sample_interval", type=int, default=10, help="epochs between image samples")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--sample_dir", type=str, default="./samples")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--resume", type=str, default=None)
    # FID
    parser.add_argument("--compute_fid", action="store_true")
    parser.add_argument("--fid_batch", type=int, default=64)
    parser.add_argument("--fid_samples", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {vars(args)}")

    # データ
    train_loader, test_loader = get_dataloader(
        args.dataset, args.image_size, args.batch_size, args.num_workers
    )
    image_channels = 3

    # モデル
    model = VAE(
        image_channels=image_channels,
        ch=args.channels,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        z_channels=args.z_channels,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # オプティマイザ
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    start_epoch = 1
    best_loss = float("inf")

    # 再開
    if args.resume:
        start_epoch, best_loss = load_checkpoint(args.resume, model, ema, optimizer, scaler, device)
        start_epoch += 1

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # 固定バッチ（再構成サンプル用）
    fixed_batch = next(iter(test_loader))[0][:64].to(device)

    # FID用特徴量抽出器
    fid_extractor = None
    if args.compute_fid:
        fid_extractor = InceptionFeatureExtractor(device)
        print("FID computation enabled (InceptionV3).")

    os.makedirs(args.sample_dir, exist_ok=True)

    # 学習ループ
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        loss, recon, kl = train_epoch(model, train_loader, optimizer, scaler, device, epoch, args, ema)
        test_loss = evaluate(model, test_loader, device, args, epoch,
                             save_path=os.path.join(args.sample_dir, f"recon_epoch{epoch:03d}.png"),
                             fixed_batch=fixed_batch)

        # TensorBoard
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Recon/train", recon, epoch)
        writer.add_scalar("KL/train", kl, epoch)

        # サンプル生成
        if epoch % args.sample_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # ランダム生成
                z = torch.randn(64, args.z_channels, args.image_size//8, args.image_size//8, device=device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    samples = model.decode(z)
                save_image_grid(samples, os.path.join(args.sample_dir, f"random_epoch{epoch:03d}.png"), nrow=8)
                print(f"[Saved random samples] epoch {epoch}")

                # 潜在空間補間
                interpolate_latent(model if ema is None else ema.shadow, device, n_rows=8, n_cols=16,
                                   save_path=os.path.join(args.sample_dir, f"interp_epoch{epoch:03d}.png"))

            # FID 計算（EMAモデル使用）
            if args.compute_fid and fid_extractor is not None:
                print("Computing FID...")
                eval_model = model if ema is None else ema.shadow
                eval_model.eval()
                with torch.no_grad():
                    z = torch.randn(args.fid_samples, args.z_channels,
                                    args.image_size//8, args.image_size//8, device=device)
                    fake_images = []
                    for i in range(0, args.fid_samples, args.batch_size):
                        z_batch = z[i:i+args.batch_size]
                        with torch.cuda.amp.autocast(enabled=args.amp):
                            x_batch = eval_model.decode(z_batch)
                        fake_images.append(x_batch.cpu())
                    fake_images = torch.cat(fake_images, dim=0)

                    # 本物画像を取得
                    real_images = []
                    for data, _ in test_loader:
                        real_images.append(data)
                        if sum(t.size(0) for t in real_images) >= args.fid_samples:
                            break
                    real_images = torch.cat(real_images, dim=0)[:args.fid_samples]

                fid_score = compute_fid(real_images, fake_images, fid_extractor, device, args.fid_batch)
                writer.add_scalar("FID", fid_score, epoch)
                print(f"====> FID: {fid_score:.2f}")

        # チェックポイント
        is_best = test_loss < best_loss
        if is_best:
            best_loss = test_loss
        save_checkpoint(model, ema, optimizer, scaler, epoch, best_loss, args, is_best=is_best)

    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
