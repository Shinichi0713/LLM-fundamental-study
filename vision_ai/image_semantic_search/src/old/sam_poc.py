import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM2ImageEncoder(nn.Module):
    """
    階層的な特徴（Multi-scale Features）を出力するイメージエンコーダの簡易実装。
    Decoder側で高解像度特徴をスキップ接続として利用します。
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Stem (初期ダウンスキャン)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 浅い層（高解像度特徴）
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # 深い層（低解像度・高次元特徴）
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        feat_high_res = self.stem(x)         # [B, 64, H/4, W/4]
        feat_mid = self.layer1(feat_high_res) # [B, 128, H/4, W/4]
        feat_low_res = self.layer2(feat_mid)  # [B, 256, H/8, W/8]
        
        # 深い特徴量とスキップ接続用の高解像度特徴量を返す
        return feat_low_res, [feat_high_res, feat_mid]

class MemoryAttentionBlock(nn.Module):
    """
    現在フレームの特徴量に対し、自己アテンションと記憶領域へのクロスアテンションを適用するブロック
    """
    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_memory = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, memory_bank=None):
        # x: [B, N, C] (現在の画像特徴ベクトル列)
        # memory_bank: [B, M, C] (過去フレームから蓄積された記憶トークン列)
        
        # 1. Self-Attention (現フレーム内の空間的文脈の強調)
        q = x + self.self_attn(x, x, x)[0]
        q = self.norm1(q)
        
        # 2. Cross-Attention with Memory (過去記憶の参照)
        if memory_bank is not None and memory_bank.shape[1] > 0:
            memory_out = self.cross_attn_memory(query=q, key=memory_bank, value=memory_bank)[0]
            q = self.norm2(q + memory_out)
        
        # 3. FFN
        out = self.norm3(q + self.mlp(q))
        return out

class SAM2PromptEncoder(nn.Module):
    """
    ポイント座標・マスク入力をベクトル埋め込みに変換
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        # 前景/背景ポイント判別用の学習可能埋め込み
        self.fg_point_embed = nn.Embedding(1, embed_dim)
        self.bg_point_embed = nn.Embedding(1, embed_dim)
        
        # マスクプロンプト縮小用CNN
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, embed_dim, kernel_size=2, stride=2),
            nn.GELU(),
        )

    def forward(self, points=None, labels=None, masks=None):
        sparse_embeddings = torch.empty((1, 0, self.embed_dim))
        dense_embeddings = None

        # 1. ポイントプロンプトの埋め込み (位置エンコーディング + 点タイプ)
        if points is not None and labels is not None:
            # 簡略化のため、ダミーの位置埋め込み加算
            batch_size = points.shape[0]
            pt_embeds = []
            for i in range(points.shape[1]):
                label = labels[:, i]
                base_embed = self.fg_point_embed.weight if label == 1 else self.bg_point_embed.weight
                pt_embeds.append(base_embed.repeat(batch_size, 1, 1))
            sparse_embeddings = torch.cat(pt_embeds, dim=1)

        # 2. マスクプロンプトの埋め込み
        if masks is not None:
            dense_embeddings = self.mask_downscaling(masks)

        return sparse_embeddings, dense_embeddings

class SAM2MaskDecoder(nn.Module):
    """
    Two-Way Transformer構造を利用し、マスクとIoU信頼度スコアを生成
    """
    def __init__(self, transformer_dim=256, num_multimask_outputs=3):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = num_multimask_outputs + 1 # 多重度マスク出力対応
        
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        
        # マスク生成用 HyperNetworks
        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transformer_dim, transformer_dim),
                nn.ReLU(),
                nn.Linear(transformer_dim, transformer_dim // 8)
            ) for _ in range(self.num_mask_tokens)
        ])
        self.iou_prediction_head = nn.Linear(transformer_dim, self.num_mask_tokens)

    def forward(self, image_embeddings, prompt_embeddings, high_res_features=None):
        # image_embeddings: [B, C, H, W]
        # prompt_embeddings: [B, N_prompt, C]
        B, C, H, W = image_embeddings.shape
        
        # トークン結合 [IoUトークン, マスク描画トークン, プロンプト]
        tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0).unsqueeze(0).repeat(B, 1, 1)
        if prompt_embeddings.shape[1] > 0:
            tokens = torch.cat([tokens, prompt_embeddings], dim=1)
            
        # 簡易Two-Way Interaction (トークンと画像特徴量の更新)
        img_flat = image_embeddings.flatten(2).permute(0, 2, 1) # [B, HW, C]
        updated_tokens = tokens + torch.mean(img_flat, dim=1, keepdim=True)
        
        # IoUスコア予測
        iou_pred = self.iou_prediction_head(updated_tokens[:, 0, :])
        
        # マスク用カーネル計算と画像特徴マップとの行列積
        mask_tokens_out = updated_tokens[:, 1 : 1 + self.num_mask_tokens, :]
        
        masks = []
        for i in range(self.num_mask_tokens):
            hyper_mlp = self.output_hypernetworks_mlps[i]
            kernel = hyper_mlp(mask_tokens_out[:, i, :]) # [B, C/8]
            
            # 画像特徴量を低解像度から拡大・投影
            feat_resized = F.interpolate(image_embeddings, size=(H*4, W*4), mode='bilinear', align_corners=False)
            feat_reduced = feat_resized[:, :C//8, :, :]
            
            # アダマール積によるマスク表示用ロジット算出
            mask = torch.einsum("bc, bchw -> bhw", kernel, feat_reduced).unsqueeze(1)
            masks.append(mask)
            
        pred_masks = torch.cat(masks, dim=1) # [B, num_masks, H*4, W*4]
        return pred_masks, iou_pred

class SAM2MemoryEncoder(nn.Module):
    """
    予測マスクと画像特徴量を合成し、次フレーム参照用の記憶を生成
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # マスクを画像特徴量のサイズに圧縮するコンボリューション
        self.mask_downsampler = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 32),
            nn.GELU(),
            nn.Conv2d(32, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        # 画像特徴量とマスク特徴量の融合
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, image_features, predicted_mask):
        # image_features: [B, C, H, W]
        # predicted_mask: [B, 1, H_mask, W_mask]
        
        # マスクダウンサンプリング
        mask_feat = self.mask_downsampler(predicted_mask)
        mask_feat = F.interpolate(mask_feat, size=image_features.shape[2:], mode='bilinear', align_corners=False)
        
        # 特徴量の結合と融合
        fused = torch.cat([image_features, mask_feat], dim=1)
        memory_embed = self.fusion(fused) # [B, C, H, W]
        
        return memory_embed

class SAM2Model(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.image_encoder = SAM2ImageEncoder(embed_dim=embed_dim)
        self.memory_attention = MemoryAttentionBlock(dim=embed_dim)
        self.prompt_encoder = SAM2PromptEncoder(embed_dim=embed_dim)
        self.mask_decoder = SAM2MaskDecoder(transformer_dim=embed_dim)
        self.memory_encoder = SAM2MemoryEncoder(embed_dim=embed_dim)
        
        # 過去フレームの記憶を保持するキュー (Memory Bank)
        self.memory_bank = []
        self.max_memory_size = 5 # 保持する過去フレームの上限数

    def reset_memory(self):
        """動画処理の開始時に記憶をリセット"""
        self.memory_bank = []

    def forward_frame(self, frame_img, points=None, labels=None, masks=None):
        # 1. 画像特徴の抽出
        feat_low_res, high_res_feats = self.image_encoder(frame_img) # [B, C, H', W']
        
        # 2. 記憶領域（Memory Bank）との連携
        B, C, H_f, W_f = feat_low_res.shape
        feat_flat = feat_low_res.flatten(2).permute(0, 2, 1) # [B, H'W', C]
        
        if len(self.memory_bank) > 0:
            memories = torch.cat(self.memory_bank, dim=1) # [B, M*H'W', C]
        else:
            memories = None
            
        # 記憶情報をアテンションで注入
        conditioned_feat_flat = self.memory_attention(feat_flat, memory_bank=memories)
        conditioned_feat = conditioned_feat_flat.permute(0, 2, 1).reshape(B, C, H_f, W_f)

        # 3. プロンプトのエンコード
        sparse_embeds, dense_embeds = self.prompt_encoder(points, labels, masks)

        # 4. マスク解読（Decoder）
        pred_masks, iou_scores = self.mask_decoder(conditioned_feat, sparse_embeds, high_res_feats)
        
        # 5. 今回のフレームから新しい記憶を生成してMemory Bankに蓄積
        best_mask = pred_masks[:, 0:1, :, :] # 最高スコアのマスクを選択
        new_memory = self.memory_encoder(feat_low_res, best_mask)
        new_memory_flat = new_memory.flatten(2).permute(0, 2, 1) # [B, H'W', C]
        
        # Memory Bankに追加（FIFO形式で古い記憶を削除）
        self.memory_bank.append(new_memory_flat)
        if len(self.memory_bank) > self.max_memory_size:
            self.memory_bank.pop(0)

        return pred_masks, iou_scores


# --- 動作確認用サンプルコード ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAM2Model().to(device)
    model.eval()

    # ダミー動画フレーム 3コマ (Batch=1, C=3, H=256, W=256)
    video_frames = [torch.randn(1, 3, 256, 256).to(device) for _ in range(3)]
    
    # 第1フレームにのみポイントプロンプトを与える
    prompt_points = torch.tensor([[[128, 128]]]).to(device) # (x, y)
    prompt_labels = torch.tensor([[1]]).to(device)         # 前景=1

    model.reset_memory()
    print("--- SAM2 動画推論シミュレーション ---")
    
    for frame_idx, frame in enumerate(video_frames):
        pts = prompt_points if frame_idx == 0 else None
        lbs = prompt_labels if frame_idx == 0 else None
        
        with torch.no_grad():
            masks, iou = model.forward_frame(frame, points=pts, labels=lbs)
            
        print(f"Frame {frame_idx + 1}: マスク出力形状 {masks.shape} | IoU予測: {iou[0, 0].item():.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """ピクセルのクラス不均衡（背景が多く対象領域が小さい）に対処するFocal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: [B, 1, H, W] (Logits)
        # targets: [B, 1, H, W] (Binary Mask: 0 or 1)
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()


class DiceLoss(nn.Module):
    """予測領域と正解領域の重なり度合いを評価するDice Loss"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        p_flat = p.flatten(1)
        t_flat = targets.flatten(1)

        intersection = (p_flat * t_flat).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (p_flat.sum(dim=1) + t_flat.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


def calculate_ground_truth_iou(pred_logits, gt_masks):
    """予測マスクとGTマスクの実際のIoU（Jaccard Index）を計算"""
    pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred_binary * gt_masks).sum(dim=(-2, -1))
    union = pred_binary.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


class SAM2CompositeLoss(nn.Module):
    """SAM2 用 複合損失関数 (Focal + Dice + IoU Head Loss)"""
    def __init__(self, focal_weight=20.0, dice_weight=1.0, iou_weight=1.0):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.iou_mse = nn.MSELoss()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

    def forward(self, pred_masks, pred_iou, gt_masks):
        # pred_masks: [B, num_masks, H, W]
        # pred_iou:   [B, num_masks]
        # gt_masks:   [B, 1, H, W]

        # 最もスコアが高い（またはインデックス0の）予測マスクを使用
        top_mask = pred_masks[:, 0:1, :, :]
        top_iou_pred = pred_iou[:, 0]

        # 1. Mask Loss 計算
        l_focal = self.focal_loss(top_mask, gt_masks)
        l_dice = self.dice_loss(top_mask, gt_masks)

        # 2. IoU Prediction Loss 計算
        actual_iou = calculate_ground_truth_iou(top_mask, gt_masks).detach()
        l_iou = self.iou_mse(top_iou_pred, actual_iou.squeeze(-1))

        # 3. 加重和
        total_loss = (
            self.focal_weight * l_focal +
            self.dice_weight * l_dice +
            self.iou_weight * l_iou
        )

        return total_loss, {
            "focal": l_focal.item(),
            "dice": l_dice.item(),
            "iou": l_iou.item()
        }

import torch
from torch.utils.data import DataLoader

def train_sam2_video_epoch(model, dataloader, optimizer, criterion, device, detach_interval=4):
    model.train()
    total_epoch_loss = 0.0

    for batch_idx, batch_data in enumerate(dataloader):
        # batch_data['video_frames']: [B, T, 3, H, W] (動画シーケンス)
        # batch_data['gt_masks']:     [B, T, 1, H, W] (各フレームの正解マスク)
        # batch_data['init_point']:   [B, 1, 2]       (第1フレーム用のプロンプト点)
        
        frames = batch_data['video_frames'].to(device)
        gt_masks = batch_data['gt_masks'].to(device)
        init_point = batch_data['init_point'].to(device)
        init_label = torch.ones((frames.shape[0], 1), device=device) # 前景ラベル

        B, T, C, H, W = frames.shape
        optimizer.zero_grad()
        
        # 記憶のリセット
        model.reset_memory()
        sequence_loss = 0.0

        for t in range(T):
            current_frame = frames[:, t, :, :, :]
            current_gt = gt_masks[:, t, :, :, :]

            # 第1フレームのみプロンプトを与え、以降は記憶のみで推論
            if t == 0:
                pts, lbs = init_point, init_label
            else:
                pts, lbs = None, None

            # フレーム単位の順伝播
            pred_masks, iou_scores = model.forward_frame(
                current_frame, points=pts, labels=lbs
            )

            # 損失計算
            loss, loss_dict = criterion(pred_masks, iou_scores, current_gt)
            sequence_loss += loss

            # メモリ膨張を防ぐため、一定ステップ毎にMemory Bankの計算グラフを切断
            if (t + 1) % detach_interval == 0:
                model.memory_bank = [m.detach() for m in model.memory_bank]

        # 動画1シーケンス全体（Tフレーム分）の平均損失で逆伝播
        sequence_loss = sequence_loss / T
        sequence_loss.backward()
        
        # 勾配クリッピング (安定化のため)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_epoch_loss += sequence_loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch [{batch_idx}/{len(dataloader)}] - Sequence Loss: {sequence_loss.item():.4f}")

    return total_epoch_loss / len(dataloader)