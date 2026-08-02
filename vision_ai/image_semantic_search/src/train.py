import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import RoIAlign, batched_nms


# ----------------------------------------------------
# 1. RegionCLIP モデル定義
# ----------------------------------------------------
class RegionCLIP(nn.Module):
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
        
        self.roi_align = RoIAlign(
            output_size=(output_size, output_size),
            spatial_scale=spatial_scale,
            sampling_ratio=-1
        )
        self.roi_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.region_projector = nn.Linear(feature_dim, embed_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature)))

    def extract_region_features(self, images: torch.Tensor, boxes_list: list[torch.Tensor]) -> torch.Tensor:
        feature_maps = self.image_encoder(images)
        roi_features = self.roi_align(feature_maps, boxes_list)
        pooled_features = self.roi_pool(roi_features).flatten(1)
        region_embeds = self.region_projector(pooled_features)
        return F.normalize(region_embeds, dim=-1)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_encoder(text_tokens)
        return F.normalize(text_embeds, dim=-1)

    def forward(
        self, 
        images: torch.Tensor, 
        boxes_list: list[torch.Tensor], 
        text_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        region_embeds = self.extract_region_features(images, boxes_list)  # (Total_N, D)
        text_embeds = self.encode_text(text_tokens)                        # (K, D)
        
        logit_scale = self.logit_scale.exp()
        logits_per_region = logit_scale * (region_embeds @ text_embeds.T)  # (Total_N, K)
        
        return {
            "logits_per_region": logits_per_region,
            "region_embeds": region_embeds,
            "text_embeds": text_embeds
        }


# ----------------------------------------------------
# 2. ダミーデータセットの定義
# ----------------------------------------------------
class RegionDataset(Dataset):
    """
    画像、事前抽出されたバウンディングボックス、および候補概念テキストのダミーデータセット
    """
    def __init__(self, num_samples: int = 100, num_concepts: int = 20):
        super().__init__()
        self.num_samples = num_samples
        # 候補となるテキスト概念（例: "a photo of a cat", "a photo of a car" 等のトークン）
        self.concept_tokens = torch.randint(0, 1000, (num_concepts, 77))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 画像 tensor: (3, 224, 224)
        image = torch.randn(3, 224, 224)
        
        # 擬似領域（Bounding Box: x1, y1, x2, y2）をランダム生成 (5〜10個/画像)
        num_boxes = torch.randint(5, 11, (1,)).item()
        x1y1 = torch.rand(num_boxes, 2) * 150.0
        wh = torch.rand(num_boxes, 2) * 50.0 + 20.0
        boxes = torch.cat([x1y1, x1y1 + wh], dim=1)
        
        return {
            "image": image,
            "boxes": boxes
        }

def custom_collate_fn(batch: list[dict]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    可変長のバウンディングボックスに対応するカスタム collate 関数
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    boxes_list = [item["boxes"] for item in batch]
    return images, boxes_list


# ----------------------------------------------------
# 3. ダミーエンコーダ（動作用）
# ----------------------------------------------------
class DummyImageEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, 224, 224) -> (B, 2048, 14, 14) の特徴マップを返す
        return torch.randn(x.size(0), 2048, 14, 14, device=x.device)

class DummyTextEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (K, seq_len) -> (K, 512) の埋め込みを返す
        return torch.randn(x.size(0), 512, device=x.device)


# ----------------------------------------------------
# 4. 学習ループ処理
# ----------------------------------------------------
def train_regionclip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ハイパーパラメータ
    batch_size = 4
    num_epochs = 3
    learning_rate = 1e-4
    num_concepts = 20  # 概念プール（Concept Pool）のサイズ

    # データセット & データローダーの構築
    dataset = RegionDataset(num_samples=40, num_concepts=num_concepts)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn
    )

    # モデルとオプティマイザの初期化
    image_encoder = DummyImageEncoder()
    text_encoder = DummyTextEncoder()
    model = RegionCLIP(image_encoder, text_encoder).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # テキスト概念プール（固定的または学習対象のテキスト表現）
    concept_tokens = dataset.concept_tokens.to(device)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for step, (images, boxes_list) in enumerate(dataloader):
            images = images.to(device)
            boxes_list = [b.to(device) for b in boxes_list]

            optimizer.zero_grad()

            # 1. 順伝播
            outputs = model(images, boxes_list, concept_tokens)
            logits_per_region = outputs["logits_per_region"]  # (Total_N, K)

            # 2. 擬似ターゲットの生成 (Pseudo-Labeling Step)
            # 実際には教師なし事前学習において、事前学習済みTeacher CLIPのスコアを最大とする概念インデックスを割り当てます
            with torch.no_grad():
                pseudo_targets = torch.argmax(logits_per_region, dim=-1)  # (Total_N,)

            # 3. 領域-テキスト対照損失の計算
            loss = criterion(logits_per_region, pseudo_targets)

            # 4. 逆伝播 & パラメータ更新
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--> Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}\n")

if __name__ == "__main__":
    train_regionclip()