import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class JapaneseCLIP(nn.Module):
    def __init__(self, image_encoder, text_model_name="cl-tohoku/bert-base-japanese-v3", embed_dim=512):
        super().__init__()
        # 画像側：既存のCLIPのImage Encoder（例: ViT）
        self.image_encoder = image_encoder
        
        # テキスト側：日本語BERT
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # 特徴量を共通の埋め込み空間（512次元など）に投影
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        
        # 学習可能な温度パラメータ（類似度のスケーリング用）
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, images, input_ids, attention_mask):
        # 画像とテキストの特徴抽出
        image_features = self.image_encoder(images).last_hidden_state[:, 0, :]
        text_features = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        
        # 共通空間への投影と正規化
        image_embeds = F.normalize(image_features, p=2, dim=-1) # 画像エンコーダーが既に投影済みの場合
        text_embeds = F.normalize(self.text_projection(text_features), p=2, dim=-1)
        
        # 類似度行列の計算
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text