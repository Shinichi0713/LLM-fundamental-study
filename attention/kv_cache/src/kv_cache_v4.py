import torch
from typing import Dict, List, Optional, Tuple


class KVCache:
    """
    Transformer推論用のKVキャッシュ（PyTorchテンソル版）
    - バッチ・シーケンス長・ヘッド数・ヘッド次元に対応
    - 各レイヤ・各ヘッドごとにKey/Valueテンソルを保持
    - 逐次推論時に過去のKVを蓄積し、自己注意で再利用
    """
    def __init__(self) -> None:
        # 構造: {layer_idx: {"k": Tensor, "v": Tensor}}
        # テンソル形状: [batch_size, num_heads, seq_len, head_dim]
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def init_layer_cache(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        指定レイヤのKVキャッシュを初期化（初回ステップ用）
        - k, v: [batch_size, num_heads, seq_len, head_dim]
        """
        self._cache[layer_idx] = {"k": k.clone(), "v": v.clone()}

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        既存のKVキャッシュに新しいKey/Valueを連結して更新
        - k, v: 現在ステップのKV [batch_size, num_heads, new_seq_len, head_dim]
        """
        if layer_idx not in self._cache:
            # 初回はそのまま保存
            self.init_layer_cache(layer_idx, k, v)
            return

        # 既存のKVを取得
        prev_k = self._cache[layer_idx]["k"]  # [B, H, prev_seq, D]
        prev_v = self._cache[layer_idx]["v"]  # [B, H, prev_seq, D]

        # シーケンス方向に連結
        # [B, H, prev_seq + new_seq, D]
        updated_k = torch.cat([prev_k, k], dim=2)
        updated_v = torch.cat([prev_v, v], dim=2)

        self._cache[layer_idx]["k"] = updated_k
        self._cache[layer_idx]["v"] = updated_v

    def get(
        self,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        指定レイヤのKVを取得（なければNone）
        戻り値: (k, v) テンソル [B, H, seq_len, D]
        """
        if layer_idx not in self._cache:
            return None
        cache_entry = self._cache[layer_idx]
        return cache_entry["k"], cache_entry["v"]

    def clear_layer(self, layer_idx: int) -> None:
        """指定レイヤのキャッシュをクリア"""
        if layer_idx in self._cache:
            del self._cache[layer_idx]

    def clear_all(self) -> None:
        """全レイヤのキャッシュをクリア"""
        self._cache.clear()

    def total_seq_len(self, layer_idx: int) -> int:
        """指定レイヤの累積シーケンス長を返す（バッチ0, ヘッド0を基準）"""
        if layer_idx not in self._cache:
            return 0
        k = self._cache[layer_idx]["k"]  # [B, H, S, D]
        return k.size(2)  # seq_len

    def __repr__(self) -> str:
        info = []
        for layer_idx, cache in self._cache.items():
            k = cache["k"]
            B, H, S, D = k.shape
            info.append(f"layer{layer_idx}: shape=[{B},{H},{S},{D}]")
        return f"KVCache({', '.join(info)})"
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttentionWithKVCache(nn.Module):
    """
    KVキャッシュを使った因果的自己注意機構（推論用）
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Q, K, V の線形投影
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

        # 因果マスク（上三角行列）を事前計算
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

    def forward(
        self,
        x: torch.Tensor,              # [batch_size, seq_len, embed_dim]
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        推論時のフォワード
        - x: 現在ステップの入力（過去トークン＋新規トークン、または新規トークンのみ）
        - kv_cache: 過去のKVを保持するキャッシュ（Noneなら初回）
        - layer_idx: どのレイヤのキャッシュを使うか
        - use_cache: TrueならKVキャッシュを更新して返す
        """
        B, S, E = x.shape
        H, D = self.num_heads, self.head_dim

        # Q, K, V を計算
        q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
        k = self.k_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
        v = self.v_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]

        if kv_cache is not None:
            # 過去のKVがあれば取得
            cached_kv = kv_cache.get(layer_idx)
            if cached_kv is not None:
                cached_k, cached_v = cached_kv  # [B, H, prev_S, D]
                # 現在ステップのK,Vと連結
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)

        # 現在の総シーケンス長
        total_S = k.size(2)

        # 注意スコア計算: Q @ K^T / sqrt(D)
        # q: [B, H, S, D], k: [B, H, total_S, D] -> scores: [B, H, S, total_S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)

        # 因果マスク適用（未来のトークンを見ないようにする）
        if total_S <= self.max_seq_len:
            mask = self.causal_mask[:S, :total_S]  # [S, total_S]
        else:
            # 事前計算マスクを超える長さの場合は動的に生成（簡易）
            mask = torch.triu(torch.ones(S, total_S), diagonal=1).bool()
            mask = mask.to(scores.device)
        scores = scores.masked_fill(mask, float("-inf"))

        # ソフトマックス
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, S, total_S]

        # 加重和: attn_weights @ v
        # [B, H, S, total_S] @ [B, H, total_S, D] -> [B, H, S, D]
        attn_output = torch.matmul(attn_weights, v)

        # ヘッドを結合して出力次元に戻す
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, H * D)
        output = self.out_proj(attn_output)  # [B, S, E]

        # KVキャッシュ更新
        updated_cache = None
        if use_cache:
            if kv_cache is None:
                kv_cache = KVCache()
            # 現在ステップのK,Vをキャッシュに追加
            # 注意: ここでは「現在ステップのK,Vのみ」を追加する想定
            # （実際の実装では、xが「新規トークンのみ」か「全トークン」かで設計が変わる）
            kv_cache.update(layer_idx, k, v)
            updated_cache = kv_cache

        return output, updated_cache



def demo_kv_cache_usage():
    """
    KVキャッシュを使った逐次トークン生成のデモ（概念）
    """
    B, E, H, D = 1, 256, 8, 32  # バッチ1, 埋め込み256, ヘッド8, ヘッド次元32
    attn = CausalSelfAttentionWithKVCache(embed_dim=E, num_heads=H, head_dim=D)

    # 初回ステップ: トークン0のみ
    x0 = torch.randn(B, 1, E)  # [1, 1, 256]
    output0, kv_cache = attn(x0, kv_cache=None, layer_idx=0, use_cache=True)
    print("Step0 cache:", kv_cache)
    print("Step0 total_seq_len:", kv_cache.total_seq_len(0))

    # 2ステップ目: トークン1のみ（過去のKVを再利用）
    x1 = torch.randn(B, 1, E)  # [1, 1, 256]
    output1, kv_cache = attn(x1, kv_cache=kv_cache, layer_idx=0, use_cache=True)
    print("Step1 cache:", kv_cache)
    print("Step1 total_seq_len:", kv_cache.total_seq_len(0))

    # 3ステップ目: トークン2のみ
    x2 = torch.randn(B, 1, E)
    output2, kv_cache = attn(x2, kv_cache=kv_cache, layer_idx=0, use_cache=True)
    print("Step2 cache:", kv_cache)
    print("Step2 total_seq_len:", kv_cache.total_seq_len(0))

    # キャッシュクリア
    kv_cache.clear_all()
    print("After clear_all:", kv_cache)


class MultiLayerKVCache:
    """
    複数レイヤをまとめて管理するKVキャッシュラッパー
    """
    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.caches = [KVCache() for _ in range(num_layers)]

    def update_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.caches[layer_idx].update(layer_idx, k, v)

    def get_layer(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.caches[layer_idx].get(layer_idx)

    def clear_all(self) -> None:
        for cache in self.caches:
            cache.clear_all()
