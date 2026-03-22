import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

def simulate_anyres(image_path, patch_size=224):
    # 1. 画像の読み込み
    raw_image = Image.open(image_path).convert('RGB')
    
    # 2. 全体像（サムネイル）の作成
    # どんなに大きくても一旦 patch_size に縮小
    thumbnail_transform = T.Compose([
        T.Resize((patch_size, patch_size)),
        T.ToTensor()
    ])
    thumbnail = thumbnail_transform(raw_image) # [3, 224, 224]

    # 3. 高解像度パッチへの分割 (2x2 グリッドを想定)
    # 元画像を 448x448 にリサイズしてから 224x224 の4枚に分ける
    high_res_transform = T.Compose([
        T.Resize((patch_size * 2, patch_size * 2)),
        T.ToTensor()
    ])
    high_res_img = high_res_transform(raw_image) # [3, 448, 448]
    
    # [3, 448, 448] -> [3, 2, 224, 2, 224] -> [4, 3, 224, 224] に変換
    patches = high_res_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    
    # 4. 可視化 (何が起きているか確認)
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    axes[0].imshow(thumbnail.permute(1, 2, 0))
    axes[0].set_title("Thumbnail")
    for i in range(4):
        axes[i+1].imshow(patches[i].permute(1, 2, 0))
        axes[i+1].set_title(f"Patch {i+1}")
    plt.show()

    return thumbnail, patches

# 実行例 (Colabに画像をアップロードしてパスを指定してください)
thumbnail, patches = simulate_anyres("/content/feature-3.png")