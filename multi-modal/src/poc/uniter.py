import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class UNITER(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(UNITER, self).__init__()
        self.bert_config = BertConfig.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.image_linear = nn.Linear(2048, self.bert_config.hidden_size)  # Assuming image features are 2048-dim
        self.cross_attention = nn.MultiheadAttention(self.bert_config.hidden_size, num_heads=8)
    
    def forward(self, input_ids, attention_mask, token_type_ids, image_features):
        # Process text input through BERT
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        text_features = text_outputs.last_hidden_state
        
        # Process image features through a linear layer
        image_features = self.image_linear(image_features)
        
        # Concatenate text and image features
        combined_features = torch.cat((text_features, image_features), dim=1)
        
        # Create attention mask for combined features
        combined_attention_mask = torch.cat((attention_mask, torch.ones(image_features.size(0), image_features.size(1)).to(attention_mask.device)), dim=1)
        
        # Apply cross attention
        attn_output, _ = self.cross_attention(combined_features, combined_features, combined_features, attn_mask=combined_attention_mask)
        
        return attn_output

# Example usage
input_ids = torch.tensor([[101, 2054, 2003, 2023, 102]])  # Example token IDs
attention_mask = torch.tensor([[1, 1, 1, 1, 1]])  # Example attention mask
token_type_ids = torch.tensor([[0, 0, 0, 0, 0]])  # Example token type IDs
image_features = torch.randn(1, 10, 2048)  # Example image features (batch_size, num_regions, feature_dim)

model = UNITER()
output = model(input_ids, attention_mask, token_type_ids, image_features)
print(output.shape)