import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """画像特徴から情報を吸い上げるためのクロスアテンション層"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, queries, image_embeds):
        # queries: [batch, 32, 768] (検索者)
        # image_embeds: [batch, 257, 768] (データベース)
        attn_output, _ = self.multihead_attn(query=queries, key=image_embeds, value=image_embeds)
        return self.norm(queries + attn_output)

class QFormerBlock(nn.Module):
    """Q-Formerの1レイヤー（Self-Attention + Cross-Attention + MLP）"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = CrossAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, image_embeds):
        # 1. Self-Attention (クエリ同士、またはクエリとテキストの相互作用)
        s_attn, _ = self.self_attn(x, x, x)
        x = self.norm1(x + s_attn)
        
        # 2. Cross-Attention (画像から情報を抽出)
        x = self.cross_attn(x, image_embeds)
        
        # 3. Feed Forward
        x = self.norm2(x + self.mlp(x))
        return x

class QFormer(nn.Module):
    def __init__(self, num_queries=32, embed_dim=768, num_layers=6, num_heads=12):
        super().__init__()
        # 学習可能なクエリ（ここがQ-Formerの「脳」の種）
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.layers = nn.ModuleList([
            QFormerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        # LLMの次元に合わせるための射影層 (例: 768 -> 4096)
        self.llm_proj = nn.Linear(embed_dim, 4096) 

    def forward(self, image_embeds):
        batch_size = image_embeds.shape[0]
        
        # クエリをバッチサイズに拡張
        x = self.query_tokens.expand(batch_size, -1, -1)
        
        # 各レイヤーで画像特徴と相互作用
        for layer in self.layers:
            x = layer(x, image_embeds)
            
        # LLM用のトークン形式に変換
        llm_tokens = self.llm_proj(x)
        return llm_tokens

# --- 動作確認 ---
batch_size = 1
image_features = torch.randn(batch_size, 257, 768) # ViTの出力を想定

qformer = QFormer()
output_tokens = qformer(image_features)

print(f"画像パッチ数: {image_features.shape[1]}")
print(f"LLMへ渡されるトークン数: {output_tokens.shape[1]}") # 32に圧縮される
print(f"各トークンの次元数: {output_tokens.shape[2]}") # 4096 (LLM用)