import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def causal_conv1d_visualization():
    """
    Mambaで用いられる因果1次元畳み込みの処理イメージを可視化する
    """
    # 設定
    seq_len = 10
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride = 1

    # ダミー入力（1バッチ, 1チャネル, シーケンス長）
    # 値はわかりやすくするため単調増加
    x = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, 1, -1)  # (1, 1, L)

    # カーネル（重み）を固定値で設定（例: [0.5, 1.0, 0.5]）
    conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,  # あとで左パディングを手動で追加して因果性を確保
        bias=False
    )
    # カーネル重みを固定
    with torch.no_grad():
        conv.weight.data = torch.tensor([[[0.5, 1.0, 0.5]]], dtype=torch.float32)

    # 因果畳み込みのための左パディング（past-only）
    # padding = kernel_size - 1 で左側だけ埋める
    x_padded = torch.nn.functional.pad(x, (kernel_size - 1, 0))  # (1, 1, L + kernel_size - 1)

    # 畳み込み実行
    y = conv(x_padded)  # (1, 1, L)

    # テンソルをnumpyに変換
    x_np = x.squeeze().numpy()
    kernel_np = conv.weight.data.squeeze().numpy()
    y_np = y.squeeze().detach().numpy()

    # 可視化
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 入力シーケンス
    ax1.stem(range(seq_len), x_np, basefmt=" ", linefmt="C0-", markerfmt="C0o")
    ax1.set_ylabel("Input $x_t$")
    ax1.set_title("Causal 1D Convolution Visualization")
    ax1.grid(True, alpha=0.3)

    # カーネル（フィルタ）
    ax2.stem(range(kernel_size), kernel_np, basefmt=" ", linefmt="C1-", markerfmt="C1s")
    ax2.set_ylabel("Kernel $w$")
    ax2.grid(True, alpha=0.3)

    # 出力シーケンス（畳み込み結果）
    ax3.stem(range(seq_len), y_np, basefmt=" ", linefmt="C2-", markerfmt="C2^")
    ax3.set_ylabel("Output $y_t$")
    ax3.set_xlabel("Time step $t$")
    ax3.grid(True, alpha=0.3)

    # 因果性の説明テキスト
    ax3.text(
        0.02, 0.98,
        "Causal: each $y_t$ depends only on $x_t, x_{t-1}, x_{t-2}$ (past only)",
        transform=ax3.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()

    # 数値例の表示（最初の数ステップ）
    print("入力 x:", x_np)
    print("カーネル w:", kernel_np)
    print("出力 y:", y_np)
    print("\n例: y[2] = w[0]*x[0] + w[1]*x[1] + w[2]*x[2]")
    print(f"     = {kernel_np[0]:.1f}*{x_np[0]:.0f} + {kernel_np[1]:.1f}*{x_np[1]:.0f} + {kernel_np[2]:.1f}*{x_np[2]:.0f}")
    print(f"     = {y_np[2]:.1f}")

if __name__ == "__main__":
    causal_conv1d_visualization()