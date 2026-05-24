import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


def demo_svd_effect(X, max_rank=None, figsize=(12, 6)):
    """
    SVDの効果（低ランク近似による復元精度の変化）を自動で可視化する関数
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        元のデータ行列
    max_rank : int or None
        表示する最大ランク（Noneの場合は min(n_samples, n_features) まで）
    figsize : tuple
        図のサイズ
    """
    n_samples, n_features = X.shape
    if max_rank is None:
        max_rank = min(n_samples, n_features)
    else:
        max_rank = min(max_rank, min(n_samples, n_features))
    
    # 元の行列のFrobeniusノルム（正規化用）
    norm_X = np.linalg.norm(X, 'fro')
    
    # 各ランクでの近似誤差を保存
    ranks = list(range(1, max_rank + 1))
    errors = []
    
    # 近似行列を保存（一部だけ描画に使う）
    reconstructions = {}
    plot_ranks = [1, max_rank // 3, 2 * max_rank // 3, max_rank]
    
    for r in ranks:
        # 縮小SVDで低ランク近似
        svd = TruncatedSVD(n_components=r)
        X_r_reconstructed = svd.inverse_transform(svd.fit_transform(X))
        
        # 近似誤差（相対Frobeniusノルム）
        rel_error = np.linalg.norm(X - X_r_reconstructed, 'fro') / norm_X
        errors.append(rel_error)
        
        if r in plot_ranks:
            reconstructions[r] = X_r_reconstructed
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()
    
    # (0) 元データ
    if X.ndim == 2 and X.shape[1] == 2:
        # 2次元散布図
        axes[0].scatter(X[:, 0], X[:, 1], alpha=0.6, s=20)
        axes[0].set_title("Original Data")
        axes[0].set_xlabel("Feature 1")
        axes[0].set_ylabel("Feature 2")
        axes[0].grid(True)
        axes[0].set_aspect('equal')
    else:
        # それ以外はヒートマップ的に表示（簡易）
        im = axes[0].imshow(X, aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=axes[0])
        axes[0].set_title("Original Matrix")
    
    # (1)-(3) 代表的なランクでの近似結果
    for idx, r in enumerate(plot_ranks[:3], start=1):
        X_rec = reconstructions[r]
        if X.ndim == 2 and X.shape[1] == 2:
            axes[idx].scatter(X_rec[:, 0], X_rec[:, 1], alpha=0.6, s=20)
            axes[idx].set_title(f"Rank-{r} Approximation")
            axes[idx].set_xlabel("Feature 1")
            axes[idx].set_ylabel("Feature 2")
            axes[idx].grid(True)
            axes[idx].set_aspect('equal')
        else:
            im = axes[idx].imshow(X_rec, aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=axes[idx])
            axes[idx].set_title(f"Rank-{r} Approximation")
    
    # (4) 誤差曲線
    axes[4].plot(ranks, errors, marker='o', linestyle='-')
    axes[4].set_xlabel("Rank (k)")
    axes[4].set_ylabel("Relative Reconstruction Error")
    axes[4].set_title("SVD: Error vs Rank")
    axes[4].grid(True)
    
    # (5) 誤差曲線（対数スケール）
    axes[5].semilogy(ranks, errors, marker='o', linestyle='-')
    axes[5].set_xlabel("Rank (k)")
    axes[5].set_ylabel("Relative Error (log scale)")
    axes[5].set_title("SVD: Error vs Rank (log)")
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.show()


# 画像風の行列を生成（簡易）
np.random.seed(123)
H, W = 30, 40
# 低ランク成分 + ノイズ
true_rank = 5
U_true = np.random.randn(H, true_rank)
V_true = np.random.randn(W, true_rank)
signal = U_true @ V_true.T
noise = 0.1 * np.random.randn(H, W)
X_img_like = signal + noise

# SVD効果の自動可視化
demo_svd_effect(X_img_like, max_rank=10, figsize=(12, 6))