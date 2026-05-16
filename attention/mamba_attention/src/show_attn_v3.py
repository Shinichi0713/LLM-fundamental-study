import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from mamba_ssm import Mamba

def mamba_internal_attention_visualization():
    """
    Mambaの内部アテンション（selective SSMの選択パターン）を可視化する
    """
    # 設定
    batch_size = 1
    seq_len = 16
    d_model = 16
    d_state = 16
    d_conv = 4
    expand = 2

    # 小さなMambaモデル
    model = Mamba(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    ).cuda()

    # ダミー入力（簡単のため単調増加）
    x = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1).repeat(1, 1, d_model).cuda()
    # 正規化して適度な範囲に
    x = x / x.max()

    # 順伝播（内部パラメータをフックで取得）
    # ここでは簡易的に、SSMの内部状態変化を「アテンション」として解釈
    with torch.no_grad():
        y = model(x)  # (1, L, D)

    # 内部パラメータの経路を可視化するための簡易アテンション行列を生成
    # 実際のMamba実装ではB, C, Δが入力依存で変化するが、
    # ここでは「出力の変化量」からどの入力が影響したかを近似的に見る
    attention_matrix = compute_simple_attention(x, y, seq_len)

    # 可視化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 入力シーケンス
    ax1.imshow(x.squeeze().cpu().numpy().T, aspect="auto", cmap="viridis")
    ax1.set_title("Input Sequence (channels)")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Channel")

    # 出力シーケンス
    ax2.imshow(y.squeeze().cpu().numpy().T, aspect="auto", cmap="viridis")
    ax2.set_title("Output Sequence (channels)")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Channel")

    # 簡易アテンションヒートマップ
    im = ax3.imshow(attention_matrix, aspect="auto", cmap="hot", interpolation="nearest")
    ax3.set_title("Internal 'Attention' (approx.)")
    ax3.set_xlabel("Output time step")
    ax3.set_ylabel("Input time step")
    plt.colorbar(im, ax=ax3, label="Influence (arb. units)")

    plt.tight_layout()
    plt.show()

    # 数値例（最初の数ステップ）
    print("入力シーケンス (先頭5ステップ, チャネル0):")
    print(x[0, :5, 0].cpu().numpy())
    print("\n出力シーケンス (先頭5ステップ, チャネル0):")
    print(y[0, :5, 0].cpu().numpy())
    print("\n簡易アテンション行列 (5x5):")
    print(attention_matrix[:5, :5].round(3))

def compute_simple_attention(x, y, seq_len):
    """
    簡易的に「どの入力がどの出力に影響したか」を近似するアテンション行列を計算
    （実際のMambaの内部パラメータ経路を完全に再現するものではありませんが、
     選択的SSMの「どの過去情報を保持したか」のイメージを可視化します）
    """
    # ここでは、各出力位置での変化量がどの入力位置の変化と相関が高いかを
    # 単純な相関で近似（実際の実装ではB, C, Δの入力依存性を追跡する必要があります）
    x_flat = x.squeeze().cpu().numpy()  # (L, D)
    y_flat = y.squeeze().cpu().numpy()  # (L, D)

    attention = np.zeros((seq_len, seq_len))
    for t_out in range(seq_len):
        for t_in in range(t_out + 1):  # 因果性: 未来は見ない
            # 単純な内積で「影響度」を近似
            influence = np.dot(x_flat[t_in], y_flat[t_out])
            attention[t_out, t_in] = influence

    # 正規化（可視化用）
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    return attention

if __name__ == "__main__":
    mamba_internal_attention_visualization()