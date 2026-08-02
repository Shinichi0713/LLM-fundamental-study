import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

class RegionCLIP(nn.Module):
    """
    RegionCLIP の簡略化概念実装
    - 画像バックボーン（CNN / ViT）から特徴マップを取得
    - RoIAlignにより領域（Region）特徴量を抽出
    - テキストエンコーダと領域特徴量を同一次元の空間へ射影し、アライメントを計算
    """
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        feature_dim: int = 2048,
        embed_dim: int = 512,
        output_size: int = 7,
        spatial_scale: float = 1.0 / 16.0,
        temperature: float = 0.07
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # RoIAlign による領域特徴の抽出
        self.roi_align = RoIAlign(
            output_size=(output_size, output_size),
            spatial_scale=spatial_scale,
            sampling_ratio=-1
        )
        
        # RoI特徴のプーリングと埋め込み次元へのプロジェクション
        self.roi_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.region_projector = nn.Linear(feature_dim, embed_dim)
        
        # 対照学習用の温度パラメータ
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature)))

    def extract_region_features(self, images: torch.Tensor, boxes_list: list[torch.Tensor]) -> torch.Tensor:
        """
        画像バッチとバウンディングボックスから領域埋め込みを抽出
        - images: (B, C, H, W)
        - boxes_list: 各画像ごとのバウンディングボックス [ (N_i, 4), ... ] (x1, y1, x2, y2)
        - returns: 全領域の特徴量 (Total_N, embed_dim)
        """
        # 1. バックボーン画像特徴マップの取得: (B, C, H', W')
        feature_maps = self.image_encoder(images)
        
        # 2. RoIAlign による各領域特徴の抽出: (Total_N, C, output_size, output_size)
        roi_features = self.roi_align(feature_maps, boxes_list)
        
        # 3. プーリング & 埋め込み空間への射影
        pooled_features = self.roi_pool(roi_features).flatten(1)  # (Total_N, C)
        region_embeds = self.region_projector(pooled_features)     # (Total_N, embed_dim)
        
        # 4. L2正規化
        region_embeds = F.normalize(region_embeds, dim=-1)
        return region_embeds

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        テキストのトークン系列からL2正規化されたテキスト埋め込みを抽出
        - text_tokens: (K, seq_len)
        - returns: (K, embed_dim)
        """
        text_embeds = self.text_encoder(text_tokens)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds

    def forward(
        self, 
        images: torch.Tensor, 
        boxes_list: list[torch.Tensor], 
        text_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        順伝播計算と領域-テキスト間のロジット計算
        """
        # 領域表現とテキスト表現の抽出
        region_embeds = self.extract_region_features(images, boxes_list)  # (N, D)
        text_embeds = self.encode_text(text_tokens)                        # (K, D)
        
        # コサイン類似度とロジットの計算
        logit_scale = self.logit_scale.exp()
        logits_per_region = logit_scale * (region_embeds @ text_embeds.T)  # (N, K)
        logits_per_text = logits_per_region.T                               # (K, N)
        
        return {
            "logits_per_region": logits_per_region,
            "logits_per_text": logits_per_text,
            "region_embeds": region_embeds,
            "text_embeds": text_embeds
        }

class RegionCLIPLoss(nn.Module):
    """
    領域埋め込みとテキスト埋め込み間の対照損失（InfoNCE Loss）
    """
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits_per_region: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        - logits_per_region: (N, K) 各領域に対する各テキスト概念の類似度スコア
        - targets: (N,) 各領域に対応する正しいテキスト概念のインデックス
        """
        loss_region = self.cross_entropy(logits_per_region, targets)
        return loss_region