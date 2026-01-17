import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertLayer, BertConfig

class QFormer(nn.Module):
    def __init__(self, num_queries=32, embed_dim=768, num_layers=6):
        super().__init__()
        
        # 1. 学習可能なクエリ (Learnable Queries)
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        # 2. 共有されるTransformerレイヤー
        # Cross-Attentionを含むカスタムBertLayerを使用
        self.config = BertConfig(hidden_size=embed_dim, num_hidden_layers=num_layers)
        self.layers = nn.ModuleList([
            BertLayer(self.config) for _ in range(num_layers)
        ])
        
        # 3. 画像エンコーダとの接続用（Cross-Attention用）
        # 実際のBertLayerにはCross-Attentionが備わっています
        for layer in self.layers:
            layer.has_cross_attention = True

    def forward(self, image_embeds, text_tokens=None, text_mask=None):
        """
        image_embeds: [batch, num_patches, embed_dim] (Visual Features)
        text_tokens: [batch, seq_len, embed_dim] (Optional Text Input)
        """
        batch_size = image_embeds.shape[0]
        
        # クエリをバッチサイズ分コピー
        queries = self.queries.expand(batch_size, -1, -1)
        
        # テキストが入力された場合は、クエリと結合（Shared Self-Attentionのため）
        if text_tokens is not None:
            # クエリとテキストを連結
            # [batch, num_queries + seq_len, embed_dim]
            hidden_states = torch.cat([queries, text_tokens], dim=1)
        else:
            hidden_states = queries

        # Transformerレイヤーを順次適用
        for layer in self.layers:
            # 1. Self-Attention (クエリ間、およびクエリ-テキスト間の相互作用)
            # 2. Cross-Attention (image_embedsから情報を抽出)
            layer_outputs = layer(
                hidden_states,
                encoder_hidden_states=image_embeds, # 画像特徴を参照
            )
            hidden_states = layer_outputs[0]

        # LLMに渡すのはクエリに対応する部分のみ
        query_output = hidden_states[:, :self.queries.size(1), :]
        return query_output

# 動作確認
batch_size = 2
img_dim = 768
patches = 257 # 例: ViTの出力
visual_features = torch.randn(batch_size, patches, img_dim)

model = QFormer(num_queries=32, embed_dim=img_dim)
output = model(visual_features)

print(f"Input shape (Visual): {visual_features.shape}")
print(f"Output shape (Query): {output.shape}") # [2, 32, 768] に圧縮されている