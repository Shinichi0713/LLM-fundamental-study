import torch
import torch.nn as nn

class MultimodalNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, img_feat_dim, num_classes):
        super().__init__()
        # テキスト側
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_lstm = nn.LSTM(embed_dim, 128, batch_first=True)
        
        # 画像側（簡易CNN）
        self.img_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 統合層
        self.fc = nn.Linear(128 + 32, num_classes)

    def forward(self, text_input, img_input):
        # テキスト処理
        text_emb = self.text_embed(text_input)
        _, (text_feat, _) = self.text_lstm(text_emb)
        text_feat = text_feat.squeeze(0)
        
        # 画像処理
        img_feat = self.img_cnn(img_input)
        
        # 結合して分類
        combined = torch.cat([text_feat, img_feat], dim=1)
        out = self.fc(combined)
        return out