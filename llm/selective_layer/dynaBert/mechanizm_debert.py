import torch
import torch.nn as nn

class DeeBERT(nn.Module):
    def __init__(self, bert, num_labels, num_layers=12):
        super().__init__()
        self.bert = bert
        self.num_layers = num_layers
        # 各層の後に設置する Exit Head（分類器）
        self.exit_heads = nn.ModuleList([
            nn.Linear(bert.config.hidden_size, num_labels) 
            for _ in range(num_layers)
        ])
        self.threshold = 0.5  # エントロピー閾値

    def forward(self, input_ids, attention_mask, return_early=True):
        hidden = self.bert.embeddings(input_ids)
        
        for i in range(self.num_layers):
            # i番目のTransformer層を通す
            layer_output = self.bert.encoder.layer[i](hidden)
            
            # Exit Head で予測
            logits = self.exit_heads[i](layer_output[:, 0])  # [CLS]トークン
            probs = torch.softmax(logits, dim=-1)
            
            # 推論時のみ、閾値判定で途中return
            if return_early and not self.training:
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                if entropy.item() < self.threshold:
                    return logits  # ← ここで打ち切り！後続層は実行されない
            
            hidden = layer_output
        
        # 最終層まで来た場合
        return logits