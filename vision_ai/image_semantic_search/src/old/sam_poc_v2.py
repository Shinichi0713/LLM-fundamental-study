import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# 1. Image Encoder (簡易版 ViT Patch Embedding + Conv Block)
# -------------------------------------------------------------------
class ImageEncoderViT(nn.Module):
    """
    画像をダウンサンプリングし、高次元特徴マップを抽出するエンコーダ
    (本家では Window Attention を含む ViT-H/L/B を使用)
    """
    def __init__(self, in_channels: int = 3, out_chans: int = 256):
        super().__init__()
        # 16x16 パッチ埋め込みを模した設定
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_chans, kernel_size=3, stride=4, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, 256, H/16, W/16)
        return self.stem(x)


# -------------------------------------------------------------------
# 2. Prompt Encoder (Point / Box プロンプト埋め込み)
# -------------------------------------------------------------------
class PromptEncoder(nn.Module):
    """
    ポイント（点）やバウンディングボックス座標をベクトル空間へ埋め込む
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        # ポイントの種別（前景点 / 背景点）に対する学習可能トークン
        self.point_embeddings = nn.Embedding(2, embed_dim)
        # 座標用の位置エンコーディング用プロジェクション
        self.pe_layer = nn.Linear(2, embed_dim)

    def forward(
        self, 
        points: torch.Tensor | None = None, 
        labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        - points: (B, N, 2) [0.0 ~ 1.0 に正規化された (x, y) 座標]
        - labels: (B, N) [1: 前景, 0: 背景]
        - returns: (B, N, embed_dim)
        """
        if points is None:
            # プロンプトが存在しない場合のデフォルト埋め込み
            return torch.zeros(1, 1, self.embed_dim, device=self.pe_layer.weight.device)
        
        # 座標の位置エンコーディング
        pe = self.pe_layer(points)  # (B, N, embed_dim)
        # ラベル埋め込みの加算
        pt_embed = self.point_embeddings(labels)  # (B, N, embed_dim)
        
        return pe + pt_embed


# -------------------------------------------------------------------
# 3. Mask Decoder (Two-Way Transformer + Mask Prediction)
# -------------------------------------------------------------------
class TwoWayAttentionBlock(nn.Module):
    """
    画像特徴量トークンとプロンプトトークン間の相互作用（Self & Cross Attention）
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_prompt_to_img = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_img_to_prompt = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # queries: プロンプトトークン (B, N, C)
        # keys: 画像トークン (B, HW, C)
        
        # 1. プロンプト側の Self-Attention
        q_res, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q_res)

        # 2. Cross-Attention: Prompt -> Image
        q_res, _ = self.cross_attn_prompt_to_img(queries, keys, keys)
        queries = self.norm2(queries + q_res)

        # 3. Cross-Attention: Image -> Prompt
        k_res, _ = self.cross_attn_img_to_prompt(keys, queries, queries)
        keys = self.norm3(keys + k_res)

        return queries, keys


class MaskDecoder(nn.Module):
    """
    更新された画像特徴量とプロンプトからマスクを出力するデコーダ
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.two_way_transformer = TwoWayAttentionBlock(embed_dim=embed_dim)
        
        # アイオユー（IoU）予測用およびマスク予測用の学習トークン
        self.iou_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 特徴マップ転送用の転置置換コンボリューション（Up-scaling）
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm([embed_dim // 4, 1, 1]), # 各チャネル正規化の代用
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.output_hypernetworks = nn.Linear(embed_dim, embed_dim // 8)

    def forward(
        self, 
        image_embeddings: torch.Tensor, 
        prompt_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        - image_embeddings: (B, C, H', W')
        - prompt_embeddings: (B, N, C)
        - returns: (B, 1, H, W) マスクのロジット
        """
        B, C, H, W = image_embeddings.shape
        # 画像特徴量をシーケンス化 (B, H'*W', C)
        image_pe = image_embeddings.flatten(2).permute(0, 2, 1)
        
        # 特殊トークン（IoU, Mask）とプロンプトトークンを統合
        tokens = torch.cat([self.iou_token.expand(B, -1, -1), self.mask_token.expand(B, -1, -1), prompt_embeddings], dim=1)
        
        # Two-Way Attention 処理
        queries, keys = self.two_way_transformer(tokens, image_pe)
        
        # マスク用トークンの出力を抽出
        mask_embed = queries[:, 1, :]  # (B, C)
        
        # 画像特徴マップの拡大 (B, C/8, H'*4, W'*4)
        upsampled_embedding = self.output_upscaling(image_embeddings)
        
        # ハイパーネットワークによるマスク生成行列の要素積演算
        hyper_weights = self.output_hypernetworks(mask_embed)  # (B, C/8)
        
        # (B, C/8, H'*4, W'*4) * (B, C/8, 1, 1) -> (B, 1, H'*4, W'*4)
        masks = torch.sum(upsampled_embedding * hyper_weights.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=True)
        return masks


# -------------------------------------------------------------------
# 4. SAM (Segment Anything Model) 全体モジュール
# -------------------------------------------------------------------
class SegmentAnythingModel(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.image_encoder = ImageEncoderViT(out_chans=embed_dim)
        self.prompt_encoder = PromptEncoder(embed_dim=embed_dim)
        self.mask_decoder = MaskDecoder(embed_dim=embed_dim)

    def forward(
        self, 
        images: torch.Tensor, 
        points: torch.Tensor | None = None, 
        labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        - images: (B, 3, H, W)
        - points: (B, N, 2)
        - labels: (B, N)
        - returns: (B, 1, H, W) アップサンプルされたセグメンテーションマスク
        """
        # 1. 高精細な画像埋め込みの取得 (重い処理)
        image_embeds = self.image_encoder(images)
        
        # 2. インタラクティブなプロンプト埋め込みの取得
        prompt_embeds = self.prompt_encoder(points, labels)
        
        # 3. マスクの予測・生成 (軽量な処理)
        low_res_masks = self.mask_decoder(image_embeds, prompt_embeds)
        
        # 4. 元の画像解像度に合わせてインターポレーション
        masks = F.interpolate(low_res_masks, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return masks


# -------------------------------------------------------------------
# 5. 動作確認用のメイン処理
# -------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SAMモデルの構築
    sam = SegmentAnythingModel(embed_dim=256).to(device)
    
    # ダミーデータ: 画像 (1, 3, 256, 256)
    dummy_image = torch.randn(1, 3, 256, 256, device=device)
    
    # ダミープロンプト: 画像の中央あたりを指す1点 (前景点=1)
    dummy_points = torch.tensor([[[0.5, 0.5]]], device=device)  # (1, 1, 2)
    dummy_labels = torch.tensor([[1]], device=device)          # (1, 1)
    
    # 順伝播の計算
    predicted_mask = sam(dummy_image, dummy_points, dummy_labels)
    
    print(f"入力画像サイズ: {dummy_image.shape}")
    print(f"出力マスクサイズ: {predicted_mask.shape}")  # (1, 1, 256, 256) となることを確認