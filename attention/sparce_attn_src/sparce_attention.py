import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalSparseAttention(nn.Module):
    """
    ローカルアテンションの概念に基づくSparse Attentionのシンプルな実装。
    各トークンは、その周囲の定義された窓（window_size）内のトークンにのみ注目します。
    """
    def __init__(self, d_model, window_size):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        # Q, K, V の線形変換層（簡略化のため、Multi-Headではないシングルヘッドとして定義）
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x の形状: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # 1. Q, K, V の計算
        Q = self.query(x)  # (batch_size, seq_len, d_model)
        K = self.key(x)    # (batch_size, seq_len, d_model)
        V = self.value(x)  # (batch_size, seq_len, d_model)

        # 2. アテンションスコア（Q * K^T）の計算
        # Q と K^T のバッチ行列積: (batch_size, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)

        # 3. ローカルアテンションマスクの作成
        # マスクを適用し、窓の外にあるスコアを非常に小さな値（Softmax後に0になるように）に設定します。
        
        # マスクの初期化: 全てTrue (注目可能)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        
        # 注目窓（window_size）の定義
        # 例：window_size=3 の場合、各トークンは自分自身と前後1トークンに注目。
        # (window_size - 1) // 2 が片側の半窓のサイズになります。
        half_window = (self.window_size - 1) // 2
        
        # 窓の外をFalse（注目不可）に設定
        for i in range(seq_len):
            # 左端
            start = max(0, i - half_window)
            # 右端
            end = min(seq_len, i + half_window + 1)
            
            # 窓内の領域をTrueに保つ（デフォルトでTrueなので、ここでは処理をスキップ）
            # 窓外の領域を明示的にFalseにする必要があるが、ここではより単純な方法で。
            
            # 現在のインデックス i から見て、窓の外側にあるインデックスをFalseに設定
            # i から start より左、または end より右のインデックス
            
            # マスク処理の簡略化のため、窓の外を負の無限大 (-inf) にするためのマスクを作成
            
            # --- マスク生成のよりシンプルなロジック ---
            # 注目可能なインデックスの相対距離の絶対値が half_window 以下であること
            # distance_matrix: (seq_len, seq_len)
            distance_matrix = torch.abs(torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0))
            
            # True: 注目可能 (距離が half_window 以下)、False: 注目不可
            local_mask = (distance_matrix <= half_window).to(x.device)
            
        # 4. マスクの適用
        # 注目不可な部分のスコアを負の無限大に設定（Softmax適用後に0になるように）
        # `local_mask` は (seq_len, seq_len) なので、ブロードキャストされます。
        scores = scores.masked_fill(~local_mask, -torch.inf)

        # 5. Softmaxの適用
        attention_weights = F.softmax(scores, dim=-1) # (batch_size, seq_len, seq_len)

        # 6. 重みとVの積（Valueの加重平均）
        output = torch.matmul(attention_weights, V) # (batch_size, seq_len, d_model)

        return output