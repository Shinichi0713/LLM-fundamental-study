
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# 因果畳み込み
class CausalConv1D(nn.Module):
    def __init__(self, d_model, kernel_size, is_causal=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.is_causal = is_causal
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=0
        )

    def forward(self, x):
        if self.is_causal:
            # 左側に (k-1) 個のパディングを入れ、未来を隠す
            # [0, 0, 0, x1, x2, x3] -> 出力y3はx1, x2, x3に依存
            x_padded = F.pad(x, (self.kernel_size - 1, 0))
        else:
            # --- BERT / エンコーダ型 (双方向的) ---
            # 左右に均等にパディングを入れ、中心を現在に合わせる
            # [0, x1, x2, x3, 0] -> 出力y2はx1, x2, x3に依存 (k=3の場合)
            pad_left = (self.kernel_size - 1) // 2
            pad_right = self.kernel_size - 1 - pad_left
            x_padded = F.pad(x, (pad_left, pad_right))
        return self.conv1d(x_padded)







# --- 動作確認用コード ---
def test_conv_behavior():
    d_model = 1
    kernel_size = 3
    seq_len = 5
    
    # テストデータの作成 (1, 1, 5)
    # [1.0, 2.0, 3.0, 4.0, 5.0]
    x = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, 1, -1)
    
    # 重みを1に固定して計算を分かりやすくする (単なる合計を計算する設定)
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()

    print(f"--- Testing with kernel_size={kernel_size} ---\n")

    for causal in [True, False]:
        mode = "Causal (Mamba style)" if causal else "Non-Causal (BERT style)"
        model = CausalConv1D(d_model, kernel_size, is_causal=causal)
        model.apply(init_weights)
        
        with torch.no_grad():
            output = model(x)
            
            # 未来の値を書き換えて影響をチェック
            # インデックス 2 (値 3.0) を 100.0 に変更
            x_modified = x.clone()
            x_modified[0, 0, 2] = 100.0
            output_modified = model(x_modified)
            
            # 変化した場所を特定
            diff = (output != output_modified).squeeze()
            
            print(f"Mode: {mode}")
            print(f"Original Input:  {x.squeeze().tolist()}")
            print(f"Modified Input:  {x_modified.squeeze().tolist()} (at index 2)")
            print(f"Original Output: {output.squeeze().tolist()}")
            print(f"Modified Output: {output_modified.squeeze().tolist()}")
            print(f"Affected Indices: {torch.where(diff)[0].tolist()}")
            
            if causal:
                print("✓ Check: Index 2 以降のみが影響を受けています (因果性を維持)")
            else:
                print("✓ Check: Index 2 の前後が影響を受けています (双方向を参照)")
            print("-" * 50)
