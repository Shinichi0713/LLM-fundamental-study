import torch
import torch.nn as nn

class AbsolutePositionalEncoding2D(nn.Module):
    def __init__(self, height: int, width: int, embed_dim: int):
        """
        2D Absolute Positional Encoding (APE)

        Args:
            height (int): パッチグリッドの高さ
            width (int): パッチグリッドの幅
            embed_dim (int): 特徴量の次元数 (2の倍数である必要があります)
        """
        super().__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
        
        # X軸用とY軸用に次元を半分ずつ割り当てる
        half_dim = embed_dim // 2
        
        # 学習可能な1D位置埋め込みテーブル
        self.y_embed = nn.Parameter(torch.randn(height, 1, half_dim))
        self.x_embed = nn.Parameter(torch.randn(1, width, half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル [Batch, Height * Width, Embed_Dim] 
               または [Batch, Height, Width, Embed_Dim]
        Returns:
            位置情報が加算されたテンソル (xと同じ形状)
        """
        # Y方向とX方向の位置埋め込みをブロードキャストして結合 [Height, Width, Embed_Dim]
        y_pos = self.y_embed.expand(self.height, self.width, -1)
        x_pos = self.x_embed.expand(self.height, self.width, -1)
        
        pos_embed = torch.cat([y_pos, x_pos], dim=-1) # [Height, Width, Embed_Dim]
        pos_embed = pos_embed.view(-1, self.embed_dim) # [Height * Width, Embed_Dim]
        
        # 入力特徴量に加算 (Broadcasting)
        if x.dim() == 3: # [Batch, Patches, Embed_Dim]
            return x + pos_embed.unsqueeze(0)
        else: # [Batch, Height, Width, Embed_Dim]
            return x + pos_embed.view(1, self.height, self.width, self.embed_dim)

import torch
import torch.nn as nn

class AbsolutePositionalEncoding2D(nn.Module):
    def __init__(self, height: int, width: int, embed_dim: int):
        """
        2D Absolute Positional Encoding (APE)

        Args:
            height (int): パッチグリッドの高さ
            width (int): パッチグリッドの幅
            embed_dim (int): 特徴量の次元数 (2の倍数である必要があります)
        """
        super().__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
        
        # X軸用とY軸用に次元を半分ずつ割り当てる
        half_dim = embed_dim // 2
        
        # 学習可能な1D位置埋め込みテーブル
        self.y_embed = nn.Parameter(torch.randn(height, 1, half_dim))
        self.x_embed = nn.Parameter(torch.randn(1, width, half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル [Batch, Height * Width, Embed_Dim] 
               または [Batch, Height, Width, Embed_Dim]
        Returns:
            位置情報が加算されたテンソル (xと同じ形状)
        """
        # Y方向とX方向の位置埋め込みをブロードキャストして結合 [Height, Width, Embed_Dim]
        y_pos = self.y_embed.expand(self.height, self.width, -1)
        x_pos = self.x_embed.expand(self.height, self.width, -1)
        
        pos_embed = torch.cat([y_pos, x_pos], dim=-1) # [Height, Width, Embed_Dim]
        pos_embed = pos_embed.view(-1, self.embed_dim) # [Height * Width, Embed_Dim]
        
        # 入力特徴量に加算 (Broadcasting)
        if x.dim() == 3: # [Batch, Patches, Embed_Dim]
            return x + pos_embed.unsqueeze(0)
        else: # [Batch, Height, Width, Embed_Dim]
            return x + pos_embed.view(1, self.height, self.width, self.embed_dim)


import torch
import torch.nn as nn

class AbsolutePositionalEncoding2D(nn.Module):
    def __init__(self, height: int, width: int, embed_dim: int):
        """
        2D Absolute Positional Encoding (APE)

        Args:
            height (int): パッチグリッドの高さ
            width (int): パッチグリッドの幅
            embed_dim (int): 特徴量の次元数 (2の倍数である必要があります)
        """
        super().__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
        
        # X軸用とY軸用に次元を半分ずつ割り当てる
        half_dim = embed_dim // 2
        
        # 学習可能な1D位置埋め込みテーブル
        self.y_embed = nn.Parameter(torch.randn(height, 1, half_dim))
        self.x_embed = nn.Parameter(torch.randn(1, width, half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル [Batch, Height * Width, Embed_Dim] 
               または [Batch, Height, Width, Embed_Dim]
        Returns:
            位置情報が加算されたテンソル (xと同じ形状)
        """
        # Y方向とX方向の位置埋め込みをブロードキャストして結合 [Height, Width, Embed_Dim]
        y_pos = self.y_embed.expand(self.height, self.width, -1)
        x_pos = self.x_embed.expand(self.height, self.width, -1)
        
        pos_embed = torch.cat([y_pos, x_pos], dim=-1) # [Height, Width, Embed_Dim]
        pos_embed = pos_embed.view(-1, self.embed_dim) # [Height * Width, Embed_Dim]
        
        # 入力特徴量に加算 (Broadcasting)
        if x.dim() == 3: # [Batch, Patches, Embed_Dim]
            return x + pos_embed.unsqueeze(0)
        else: # [Batch, Height, Width, Embed_Dim]
            return x + pos_embed.view(1, self.height, self.width, self.embed_dim)


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- 1. 定義した2D APEと2D RoPEモジュール (前述のコードを簡略的に保持) ---

class AbsolutePositionalEncoding2D(nn.Module):
    def __init__(self, height: int, width: int, embed_dim: int):
        super().__init__()
        self.height, self.width, self.embed_dim = height, width, embed_dim
        half_dim = embed_dim // 2
        self.y_embed = nn.Parameter(torch.randn(height, 1, half_dim))
        self.x_embed = nn.Parameter(torch.randn(1, width, half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_pos = self.y_embed.expand(self.height, self.width, -1)
        x_pos = self.x_embed.expand(self.height, self.width, -1)
        pos_embed = torch.cat([y_pos, x_pos], dim=-1).view(-1, self.embed_dim)
        return x + pos_embed.unsqueeze(0)

class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.dim_per_axis = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_1d_rotary_embed(self, pos: torch.Tensor):
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def _rotate_half(self, x: torch.Tensor):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, height: int, width: int) -> torch.Tensor:
        device = q.device
        grid_y = torch.arange(height, device=device).float()
        grid_x = torch.arange(width, device=device).float()

        cos_y, sin_y = self._get_1d_rotary_embed(grid_y)
        cos_x, sin_x = self._get_1d_rotary_embed(grid_x)

        cos_y = cos_y.unsqueeze(1).repeat(1, width, 1)
        sin_y = sin_y.unsqueeze(1).repeat(1, width, 1)
        cos_x = cos_x.unsqueeze(0).repeat(height, 1, 1)
        sin_x = sin_x.unsqueeze(0).repeat(height, 1, 1)

        cos_2d = torch.cat([cos_y, cos_x], dim=-1).view(-1, self.head_dim).unsqueeze(0).unsqueeze(0)
        sin_2d = torch.cat([sin_y, sin_x], dim=-1).view(-1, self.head_dim).unsqueeze(0).unsqueeze(0)

        return (q * cos_2d) + (self._rotate_half(q) * sin_2d)

# --- 2. 特定の「点」とのアテンション挙動を可視化するメイン実験 ---

def run_point_correspondence_experiment():
    torch.manual_seed(42) # 再現性の確保
    height, width = 8, 8
    num_patches = height * width
    embed_dim = 64
    
    # 全てのパッチ（点）に「全く同じ基本特徴量」を与える
    # -> アテンションの変化は「位置エンコーディングの影響のみ」になる
    base_feature = torch.ones(1, num_patches, embed_dim)

    # ----------------------------------------------------
    # A) 2D APE の挙動計算
    # ----------------------------------------------------
    ape = AbsolutePositionalEncoding2D(height, width, embed_dim)
    feat_ape = ape(base_feature) # 位置ベクトルを加算 [1, 64, 64]
    
    # クエリ (Q) と キー (K) の内積（アテンション行列）を計算
    # Q * K^T -> [64, 64] (すべての点同士の関連度)
    attn_matrix_ape = torch.bmm(feat_ape, feat_ape.transpose(1, 2)).squeeze(0)

    # ----------------------------------------------------
    # B) 2D RoPE の挙動計算
    # ----------------------------------------------------
    rope = RotaryPositionEmbedding2D(head_dim=embed_dim)
    # Q, K を作成（バッチ=1, ヘッド=1 に整形）
    q = base_feature.unsqueeze(1) # [1, 1, 64, 64]
    k = base_feature.unsqueeze(1)
    
    q_rope = rope(q, height, width)
    k_rope = rope(k, height, width)
    
    # 内積によるアテンション行列計算
    attn_matrix_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1)).squeeze()

    # ----------------------------------------------------
    # C) 特定の基準点 A (Y=1, X=1) から各点へのスコア抽出
    # ----------------------------------------------------
    target_y, target_x = 1, 1
    target_idx = target_y * width + target_x # パッチのフラット化インデックス (9番目)

    # 基準点 (1, 1) から見た各マスへのアテンションマップ [8, 8]
    map_ape = attn_matrix_ape[target_idx].view(height, width).detach().numpy()
    map_rope = attn_matrix_rope[target_idx].view(height, width).detach().numpy()

    # ----------------------------------------------------
    # D) 比較用ヒートマップの描画
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 2D APE の結果
    im0 = axes[0].imshow(map_ape, cmap="viridis")
    axes[0].set_title("2D APE: Attention from Point (1,1)", fontsize=12)
    axes[0].plot(target_x, target_y, "r*", markersize=15, label="Target Point (1,1)")
    axes[0].legend()
    fig.colorbar(im0, ax=axes[0])

    # 2D RoPE の結果
    im1 = axes[1].imshow(map_rope, cmap="viridis")
    axes[1].set_title("2D RoPE: Attention from Point (1,1)", fontsize=12)
    axes[1].plot(target_x, target_y, "r*", markersize=15, label="Target Point (1,1)")
    axes[1].legend()
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

    # 特定の点（1マス離れた場所 vs 遠くの場所）のスコア比較を出力
    p_near = (2, 2) # 右下に1マス
    p_far = (5, 4)  # 右下に遠い場所
    
    idx_near = p_near[0] * width + p_near[1]
    idx_far = p_far[0] * width + p_far[1]

    print(f"=== 基準点 (1, 1) から見たアテンションスコア比較 ===")
    print(f"[2D APE]")
    print(f"  - 点 (2, 2) [近距離] へのスコア: {map_ape[p_near[0], p_near[1]]:.4f}")
    print(f"  - 点 (5, 4) [遠距離] へのスコア: {map_ape[p_far[0], p_far[1]]:.4f}")
    print(f"[2D RoPE]")
    print(f"  - 点 (2, 2) [近距離] へのスコア: {map_rope[p_near[0], p_near[1]]:.4f}")
    print(f"  - 点 (5, 4) [遠距離] へのスコア: {map_rope[p_far[0], p_far[1]]:.4f}")

if __name__ == "__main__":
    run_point_correspondence_experiment()