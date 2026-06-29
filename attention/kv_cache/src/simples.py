import torch
from typing import Optional, Tuple


class SimpleKVCache:
    """
    簡単なKVキャッシュ（PyTorchテンソル版）
    - バッチ1, ヘッド1, 可変シーケンス長, 固定ヘッド次元
    - 過去のKey/Valueを蓄積し、自己注意で再利用
    """
    def __init__(self) -> None:
        self.k_cache: Optional[torch.Tensor] = None  # [1, 1, seq_len, head_dim]
        self.v_cache: Optional[torch.Tensor] = None  # [1, 1, seq_len, head_dim]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        新しいKey/Valueをキャッシュに追加（シーケンス方向に連結）
        - k, v: [1, 1, new_seq_len, head_dim]
        """
        if self.k_cache is None:
            # 初回はそのまま保存
            self.k_cache = k.clone()
            self.v_cache = v.clone()
        else:
            # 既存のKVに連結
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)

    def get(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """現在のKVを取得（なければNone）"""
        if self.k_cache is None:
            return None
        return self.k_cache, self.v_cache

    def clear(self) -> None:
        """キャッシュをクリア"""
        self.k_cache = None
        self.v_cache = None

    def seq_len(self) -> int:
        """現在の累積シーケンス長を返す"""
        if self.k_cache is None:
            return 0
        return self.k_cache.size(2)  # seq_len
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSelfAttention(nn.Module):
    """
    簡単な自己注意機構（単一ヘッド、KVキャッシュ対応）
    """
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        # Q, K, V の線形投影（単一ヘッド）
        self.q_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.out_proj = nn.Linear(head_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,                    # [batch=1, seq_len, embed_dim]
        kv_cache: Optional[SimpleKVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[SimpleKVCache]]:
        """
        推論時のフォワード
        - x: 現在ステップの入力（新規トークンのみを想定）
        - kv_cache: 過去のKVを保持するキャッシュ
        - use_cache: TrueならKVキャッシュを更新して返す
        """
        B, S, E = x.shape
        assert B == 1, "この簡単モデルはバッチ1のみ対応"

        # Q, K, V を計算
        q = self.q_proj(x)  # [1, S, head_dim]
        k = self.k_proj(x)  # [1, S, head_dim]
        v = self.v_proj(x)  # [1, S, head_dim]

        # ヘッド次元を明示（[1, 1, S, head_dim]）
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        if kv_cache is not None:
            # 過去のKVがあれば取得
            cached = kv_cache.get()
            if cached is not None:
                cached_k, cached_v = cached  # [1, 1, prev_S, head_dim]
                # 現在ステップのK,Vと連結
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)

        # 現在の総シーケンス長
        total_S = k.size(2)

        # 注意スコア計算: Q @ K^T / sqrt(head_dim)
        # q: [1, 1, S, D], k: [1, 1, total_S, D]
        # -> scores: [1, 1, S, total_S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 因果マスク（未来のトークンを見ないようにする）
        # 上三角行列（対角より上を -inf でマスク）
        mask = torch.triu(torch.ones(S, total_S), diagonal=1).bool()
        mask = mask.to(scores.device)
        scores = scores.masked_fill(mask, float("-inf"))

        # ソフトマックス
        attn_weights = F.softmax(scores, dim=-1)  # [1, 1, S, total_S]

        # 加重和: attn_weights @ v
        # [1, 1, S, total_S] @ [1, 1, total_S, D] -> [1, 1, S, D]
        attn_output = torch.matmul(attn_weights, v)

        # ヘッド次元を戻す [1, S, D]
        attn_output = attn_output.squeeze(1)

        # 出力投影
        output = self.out_proj(attn_output)  # [1, S, E]

        # KVキャッシュ更新
        updated_cache = None
        if use_cache:
            if kv_cache is None:
                kv_cache = SimpleKVCache()
            # 現在ステップのK,Vのみをキャッシュに追加
            kv_cache.update(k, v)
            updated_cache = kv_cache

        return output, updated_cache


class SimpleTransformerModel(nn.Module):
    """
    簡単なTransformer風モデル（単一レイヤ、単一ヘッド）
    """
    def __init__(self, vocab_size: int, embed_dim: int, head_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = SimpleSelfAttention(embed_dim, head_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,              # [1, seq_len]
        kv_cache: Optional[SimpleKVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[SimpleKVCache]]:
        """
        推論時のフォワード
        - input_ids: 現在ステップのトークンID（新規トークンのみを想定）
        - kv_cache: 過去のKVキャッシュ
        - use_cache: Trueなら更新されたKVキャッシュを返す
        """
        # 埋め込み
        x = self.token_embedding(input_ids)  # [1, S, E]

        # 自己注意（KVキャッシュ利用）
        attn_out, kv_cache = self.attention(x, kv_cache=kv_cache, use_cache=use_cache)

        # 出力投影（語彙サイズ次元）
        logits = self.output_proj(attn_out)  # [1, S, vocab_size]

        return logits, kv_cache
    
def demo_simple_model():
    """
    簡単なモデル＋KVキャッシュの動作確認
    """
    vocab_size = 10
    embed_dim = 16
    head_dim = 8

    model = SimpleTransformerModel(vocab_size, embed_dim, head_dim)

    # 1ステップ目: トークン0のみ
    input_ids0 = torch.tensor([[0]])  # [1, 1]
    logits0, kv_cache = model(input_ids0, kv_cache=None, use_cache=True)
    print("Step0 logits shape:", logits0.shape)  # [1, 1, 10]
    print("Step0 cache seq_len:", kv_cache.seq_len())

    # 2ステップ目: トークン1のみ（過去のKVを再利用）
    input_ids1 = torch.tensor([[1]])  # [1, 1]
    logits1, kv_cache = model(input_ids1, kv_cache=kv_cache, use_cache=True)
    print("Step1 logits shape:", logits1.shape)
    print("Step1 cache seq_len:", kv_cache.seq_len())

    # 3ステップ目: トークン2のみ
    input_ids2 = torch.tensor([[2]])
    logits2, kv_cache = model(input_ids2, kv_cache=kv_cache, use_cache=True)
    print("Step2 logits shape:", logits2.shape)
    print("Step2 cache seq_len:", kv_cache.seq_len())

    # キャッシュクリア
    kv_cache.clear()
    print("After clear, seq_len:", kv_cache.seq_len())


if __name__ == "__main__":
    demo_simple_model()