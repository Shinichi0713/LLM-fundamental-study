import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from transformers import VisualBertModel
from transformers import LayoutLMv3Model, LayoutLMv3Processor
from transformers import BlipProcessor, BlipForConditionalGeneration

class CrossModalAttention(nn.Module):
    def __init__(self, dim_text, dim_image, dim_hidden):
        super(CrossModalAttention, self).__init__()
        self.text_proj = nn.Linear(dim_text, dim_hidden)
        self.image_proj = nn.Linear(dim_image, dim_hidden)
        self.attention = nn.Linear(dim_hidden, 1)

    def forward(self, text_features, image_features):
        text_proj = self.text_proj(text_features)  # (batch_size, seq_len, dim_hidden)
        image_proj = self.image_proj(image_features)  # (batch_size, num_regions, dim_hidden)
        
        # クロスアテンション
        attention_cross = self.attention(torch.tanh(text_proj.unsqueeze(2) + image_proj.unsqueeze(1)))
        attn_scores = F.softmax(attention_cross, dim=-1)
        
        # アテンションをかけた画像特徴量
        attended_image_features = torch.matmul(attn_scores.squeeze(-1), image_features)
        
        return attended_image_features

# Example usage
batch_size = 32
seq_len = 10
num_regions = 20
dim_text = 768
dim_image = 2048
dim_hidden = 512

text_features = torch.randn(batch_size, seq_len, dim_text)
image_features = torch.randn(batch_size, num_regions, dim_image)

cross_modal_attn = CrossModalAttention(dim_text, dim_image, dim_hidden)
attended_image_features = cross_modal_attn(text_features, image_features)

print(attended_image_features.shape)  # Should be (batch_size, seq_len, dim_image)