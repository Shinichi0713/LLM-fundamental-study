import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

class SimpleVLM(nn.Module):
    def __init__(self, vision_model_name, llm_model_name, vision_dim, llm_dim):
        super().__init__()
        
        # 1. Vision Encoder (例: CLIP-ViT)
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
        
        # 2. MLP Adapter
        self.adapter = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
        # 3. LLM (Frozen)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        
        # モデルの凍結（Warm-up用）
        for param in self.vision_encoder.parameters(): param.requires_grad = False
        for param in self.llm.parameters(): param.requires_grad = False

    def forward(self, pixel_values, input_ids):
        # A. 画像を特徴量に変換
        # vision_outputs.last_hidden_state: [batch, num_patches, vision_dim]
        vision_outputs = self.vision_encoder.vision_model(pixel_values)
        image_features = vision_outputs.last_hidden_state
        
        # B. アダプターで次元変換
        # image_features: [batch, num_patches, llm_dim]
        image_embeddings = self.adapter(image_features)
        
        # C. LLMの入力埋め込みを取得
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        # D. 画像とテキストを結合 (画像トークンを前方に配置)
        inputs_embeds = torch.cat([image_embeddings, text_embeddings], dim=1)
        
        # E. LLMによる推論
        return self.llm(inputs_embeds=inputs_embeds)