**vae_impl.py** を保存しました。

## 実装の構成

### 1. 基本ブロック

| クラス | 役割 |
|--------|------|
| `ResnetBlock` | 畳み込み + GroupNorm + SiLU + スキップ接続 |
| `Downsample` | stride=2畳み込みで解像度を半分に |
| `Upsample` | nearest neighbor + 畳み込みで解像度を2倍に |
| `AttentionBlock` | 空間方向のSelf-Attention（潜在空間では計算コストが低い） |

### 2. Encoder（画像 → 潜在表現）

- 入力: `[B, 3, 512, 512]`
- 出力: `[B, 8, 64, 64]`（muとlogvarをチャネル方向に連結）
- 構造: `Conv → ResNetBlocks ×3 → Downsample → ... → Attention → Conv`
- **3回ダウンサンプリング** → 縮小率 **8x**（512→64）

### 3. Decoder（潜在表現 → 画像）

- 入力: `[B, 4, 64, 64]`
- 出力: `[B, 3, 512, 512]`
- 構造: `Conv → ResNetBlocks → Upsample → ... → Conv`

### 4. VAE 本体

```python
vae = VAE(ch=128, ch_mult=(1, 2, 4, 4), z_channels=4)

# エンコード（再パラメータ化トリック付き）
z, mu, logvar = vae.encode(x)

# デコード
x_recon = vae.decode(z)

# 学習時の順伝播
x_recon, mu, logvar = vae(x)
```

### 5. 損失関数

```python
loss, recon, kl = vae_loss(x_recon, x, mu, logvar, kl_weight=0.001)
```

- **再構成損失**: MSE（ピクセル単位）
- **KLダイバージェンス**: 潜在空間の各ピクセル位置で計算
- `kl_weight` で β-VAE として調整可能

### 6. 拡散モデルとの接続

```python
vae_diff = VAEForDiffusion(vae, scale_factor=0.18215)

# 画像 → スケーリング済み潜在表現
z_scaled, mu, logvar = vae_diff.encode_to_latent(x)

# 潜在表現 → 画像
x_recon = vae_diff.decode_from_latent(z_scaled)
```

`scale_factor=0.18215` は Stable Diffusion 1.x の標準値です。

### 実行方法

```bash
python vae_impl.py
```

ダミーデータでモデル構造と順伝播を確認できます。実際の画像データセットに置き換えて `train_vae_demo()` を実行すれば学習も可能です。