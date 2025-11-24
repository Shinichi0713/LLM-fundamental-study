import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        
        # 1. 周波数 (theta) の計算
        # dim は head_dim (d_model / num_heads)
        # 2つずつペアにするため、計算するのは dim/2 個の周波数
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # モデルの状態として保存しない（勾配計算不要）バッファとして登録
        self.register_buffer("inv_freq", inv_freq)
        
        # cos/sinのキャッシュを初期化
        self.max_seq_len_cached = max_seq_len
        t = torch.arange(max_seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        
        # 外積を使って (seq_len, dim/2) の角度グリッドを作成
        freqs = torch.outer(t, self.inv_freq)
        
        # LLaMAスタイル: ベクトルの前半と後半をペアにするため、同じ角度を2回繰り返して結合
        # emb: (seq_len, dim) -> [theta_0, theta_1, ..., theta_0, theta_1, ...]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # cos, sin を計算してキャッシュ (batch, head次元のためにunsqueezeしておく)
        # shape: (1, 1, seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        # x: (batch, n_heads, seq_len, head_dim)
        if seq_len > self.max_seq_len_cached:
            # キャッシュサイズを超えた場合の再計算ロジック（省略可だが実用的には必要）
            self._update_cache(seq_len, x.device)
            
        # 入力シーケンス長に合わせてスライスして返す
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    def _update_cache(self, seq_len, device):
        # キャッシュ更新用メソッド（__init__と同じ処理）
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])


def rotate_half(x):
    """
    ベクトルを半分に分割し、符号を反転させて入れ替える関数。
    これが (-x2, x1) の部分に相当します。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    クエリとキーにRoPEを適用する関数。
    数式: (x * cos) + (rotate_half(x) * sin)
    """
    # q, k の形状: (batch, n_heads, seq_len, head_dim)
    # cos, sin の形状: (1, 1, seq_len, head_dim) -> ブロードキャストされる
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


if __name__ == "__main__":
    # パラメータ設定
    batch_size = 2
    n_heads = 8
    seq_len = 10
    head_dim = 64  # d_model / n_heads

    # 1. インスタンス化
    rope = RotaryEmbedding(dim=head_dim, max_seq_len=2048)

    # 2. ダミーのQueryとKeyを作成 (Transformer内部での計算を想定)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)

    # 3. 現在のシーケンス長に対応する cos, sin を取得
    cos, sin = rope(q, seq_len=seq_len)

    # 4. 回転を適用
    q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)

    print(f"Original Q shape: {q.shape}")
    print(f"Rotated Q shape:  {q_rotated.shape}")
    # -> 形状は変わらず、値だけが回転されている
