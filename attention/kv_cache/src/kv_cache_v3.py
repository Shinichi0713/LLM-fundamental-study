from typing import Any, Dict, Optional


class SimpleKVCache:
    """
    単純なキー・バリューキャッシュ（辞書ベース）
    """
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        """キーと値を保存"""
        self._store[key] = value

    def get(self, key: str) -> Optional[Any]:
        """キーに対応する値を取得（存在しなければNone）"""
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        """キーを削除（存在すればTrue、なければFalse）"""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """全キャッシュをクリア"""
        self._store.clear()

    def size(self) -> int:
        """キャッシュエントリ数"""
        return len(self._store)

from typing import Any, Dict, Optional
from collections import OrderedDict


class LRUKVCache:
    """
    LRU（Least Recently Used）による容量制御付きKVキャッシュ
    """
    def __init__(self, max_size: int = 1000) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._max_size = max_size
        self._store: OrderedDict[str, Any] = OrderedDict()

    def put(self, key: str, value: Any) -> None:
        """キーと値を保存（LRUで容量制御）"""
        if key in self._store:
            # 既存キーはアクセス順を更新
            self._store.move_to_end(key)
        else:
            # 新規キーは容量チェック
            if len(self._store) >= self._max_size:
                # 最も古い（LRU）エントリを削除
                self._store.popitem(last=False)
        self._store[key] = value

    def get(self, key: str) -> Optional[Any]:
        """キーに対応する値を取得（存在しなければNone、LRU順序更新あり）"""
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def delete(self, key: str) -> bool:
        """キーを削除（存在すればTrue、なければFalse）"""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """全キャッシュをクリア"""
        self._store.clear()

    def size(self) -> int:
        """キャッシュエントリ数"""
        return len(self._store)

    def max_size(self) -> int:
        """最大容量"""
        return self._max_size


from typing import List, Optional, Tuple


class TransformerKVCache:
    """
    Transformerの自己注意機構向けの簡易KVキャッシュ
    （実際にはテンソルライブラリを使いますが、ここでは概念を示すためリストで表現）
    """
    def __init__(self) -> None:
        # 各レイヤ・ヘッドごとのKVを保持する辞書
        # 実際には {layer_idx: {head_idx: (keys, values)}} のような構造
        self._cache: dict = {}

    def put(
        self,
        layer_idx: int,
        head_idx: int,
        keys: List[float],
        values: List[float],
    ) -> None:
        """指定レイヤ・ヘッドのKVを保存（上書き）"""
        if layer_idx not in self._cache:
            self._cache[layer_idx] = {}
        self._cache[layer_idx][head_idx] = (keys.copy(), values.copy())

    def get(
        self,
        layer_idx: int,
        head_idx: int,
    ) -> Optional[Tuple[List[float], List[float]]]:
        """指定レイヤ・ヘッドのKVを取得（なければNone）"""
        layer_cache = self._cache.get(layer_idx)
        if layer_cache is None:
            return None
        kv = layer_cache.get(head_idx)
        if kv is None:
            return None
        # コピーを返す（安全のため）
        return (kv[0].copy(), kv[1].copy())

    def update(
        self,
        layer_idx: int,
        head_idx: int,
        new_keys: List[float],
        new_values: List[float],
    ) -> None:
        """既存のKVに新しいKVを連結して保存（推論時の典型的な使い方）"""
        existing = self.get(layer_idx, head_idx)
        if existing is None:
            # 初回はそのまま保存
            self.put(layer_idx, head_idx, new_keys, new_values)
        else:
            # 既存のKVに新しいKVを連結
            prev_keys, prev_values = existing
            updated_keys = prev_keys + new_keys
            updated_values = prev_values + new_values
            self.put(layer_idx, head_idx, updated_keys, updated_values)

    def clear_layer(self, layer_idx: int) -> None:
        """指定レイヤのキャッシュをクリア"""
        if layer_idx in self._cache:
            del self._cache[layer_idx]

    def clear_all(self) -> None:
        """全レイヤのキャッシュをクリア"""
        self._cache.clear()


# 使用例（概念的なもの）
if __name__ == "__main__":
    cache = TransformerKVCache()

    # 1ステップ目: トークン0に対するKV
    cache.update(layer_idx=0, head_idx=0, new_keys=[0.1, 0.2], new_values=[0.3, 0.4])

    # 2ステップ目: トークン1に対するKVを追加
    cache.update(layer_idx=0, head_idx=0, new_keys=[0.5, 0.6], new_values=[0.7, 0.8])

    # 取得すると、トークン0と1のKVが連結されている
    k, v = cache.get(layer_idx=0, head_idx=0)
    print("Keys:", k)   # [0.1, 0.2, 0.5, 0.6]
    print("Values:", v) # [0.3, 0.4, 0.7, 0.8]

