import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) の実装
    - Queryは複数ヘッド、Key/Valueは1ヘッド（全Queryで共有）
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Query投影: 出力次元 = num_heads * head_dim
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        # Key投影: 出力次元 = head_dim（1ヘッド分）
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=False)
        # Value投影: 出力次元 = head_dim（1ヘッド分）
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=False)
        # 出力投影
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,                      # [batch_size, seq_len, embed_dim]
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        フォワード計算（推論想定）
        - x: 現在ステップの入力（新規トークンのみを想定）
        - kv_cache: 過去の (K, V) テンソル（なければNone）
        - use_cache: Trueなら更新されたKVを返す
        """
        B, S, E = x.shape
        H, D = self.num_heads, self.head_dim

        # Query投影: [B, S, E] -> [B, S, H*D] -> [B, H, S, D]
        q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]

        # Key投影: [B, S, E] -> [B, S, D] -> [B, 1, S, D]
        k = self.k_proj(x).unsqueeze(1)  # [B, 1, S, D]

        # Value投影: [B, S, E] -> [B, S, D] -> [B, 1, S, D]
        v = self.v_proj(x).unsqueeze(1)  # [B, 1, S, D]

        if kv_cache is not None:
            # 過去のKVがあれば取得
            cached_k, cached_v = kv_cache  # [B, 1, prev_S, D]
            # 現在ステップのK,Vと連結
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # 現在の総シーケンス長
        total_S = k.size(2)

        # 注意スコア計算: Q @ K^T / sqrt(D)
        # q: [B, H, S, D], k: [B, 1, total_S, D]
        # -> k.transpose: [B, 1, D, total_S]
        # -> matmul: [B, H, S, total_S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)

        # 因果マスク（未来のトークンを見ない）
        mask = torch.triu(torch.ones(S, total_S), diagonal=1).bool()
        mask = mask.to(scores.device)
        scores = scores.masked_fill(mask, float("-inf"))

        # ソフトマックス
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, S, total_S]

        # 加重和: attn_weights @ v
        # [B, H, S, total_S] @ [B, 1, total_S, D]
        # -> vをヘッド数分ブロードキャストして計算
        # 実際には v を [B, H, total_S, D] に拡張して matmul する方がシンプル
        v_expanded = v.expand(B, H, total_S, D)  # [B, H, total_S, D]
        attn_output = torch.matmul(attn_weights, v_expanded)  # [B, H, S, D]

        # ヘッドを結合して出力次元に戻す
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, H * D)
        output = self.out_proj(attn_output)  # [B, S, E]

        # KVキャッシュ更新
        updated_cache = None
        if use_cache:
            # 現在ステップのK,Vのみをキャッシュとして返す
            updated_cache = (k, v)

        return output, updated_cache

class SimpleTransformerLayerWithMQA(nn.Module):
    """
    MQAを使った簡単なTransformerレイヤ（デモ用）
    """
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = MultiQueryAttention(embed_dim, num_heads, head_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 自己注意（MQA）
        attn_out, kv_cache = self.attention(
            x, kv_cache=kv_cache, use_cache=use_cache
        )
        x = x + attn_out
        x = self.norm1(x)

        # FFN
        ffn_out = self.mlp(x)
        x = x + ffn_out
        x = self.norm2(x)

        return x, kv_cache
    
def demo_mqa():
    """
    MQAの動作確認（逐次トークン生成を模倣）
    """
    B, E, H, D = 2, 128, 4, 32  # バッチ2, 埋め込み128, ヘッド4, ヘッド次元32
    mqa = MultiQueryAttention(embed_dim=E, num_heads=H, head_dim=D)

    # 1ステップ目: トークン0のみ
    x0 = torch.randn(B, 1, E)  # [2, 1, 128]
    output0, kv_cache = mqa(x0, kv_cache=None, use_cache=True)
    print("Step0 output shape:", output0.shape)  # [2, 1, 128]
    if kv_cache is not None:
        k, v = kv_cache
        print("Step0 K shape:", k.shape)  # [2, 1, 1, 32]
        print("Step0 V shape:", v.shape)  # [2, 1, 1, 32]

    # 2ステップ目: トークン1のみ（過去のKVを再利用）
    x1 = torch.randn(B, 1, E)
    output1, kv_cache = mqa(x1, kv_cache=kv_cache, use_cache=True)
    print("Step1 output shape:", output1.shape)
    if kv_cache is not None:
        k, v = kv_cache
        print("Step1 K shape:", k.shape)  # [2, 1, 2, 32]
        print("Step1 V shape:", v.shape)  # [2, 1, 2, 32]

    # 3ステップ目: トークン2のみ
    x2 = torch.randn(B, 1, E)
    output2, kv_cache = mqa(x2, kv_cache=kv_cache, use_cache=True)
    print("Step2 output shape:", output2.shape)
    if kv_cache is not None:
        k, v = kv_cache
        print("Step2 K shape:", k.shape)  # [2, 1, 3, 32]
        print("Step2 V shape:", v.shape)  # [2, 1, 3, 32]


if __name__ == "__main__":
    demo_mqa()