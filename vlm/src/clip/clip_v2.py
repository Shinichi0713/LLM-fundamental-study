import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # 画像とテキストの出力を共通次元に変換するプロジェクション層
        self.image_projection = nn.Linear(image_encoder.config.hidden_size, embed_dim)
        self.text_projection = nn.Linear(text_encoder.config.hidden_size, embed_dim)
        
        # スケーリングのための学習可能なパラメータ（温度パラメータ）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, input_ids, attention_mask):
        # 1. 各エンコーダーで特徴抽出
        image_features = self.image_encoder(images).last_hidden_state[:, 0, :] # [CLS]トークン
        text_features = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        # 2. 共通空間へ投影して正規化（L2ノルム）
        image_embeds = F.normalize(self.image_projection(image_features), p=2, dim=-1)
        text_embeds = F.normalize(self.text_projection(text_features), p=2, dim=-1)

        # 3. コサイン類似度の計算
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text