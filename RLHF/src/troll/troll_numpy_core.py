"""
TROLL (Trust Regions improve Reinforcement Learning for Large Language Models)
のコアアルゴリズムを NumPy で実装した検証用コード。

論文: Becker, Freymuth, Thilges, Otto, Neumann (ICLR 2026, Oral)
      https://arxiv.org/abs/2510.03817
公式実装: https://github.com/niklasfreymuth/TROLL
          (verl フレームワーク + 依存ライブラリ pbecker93/discrete_trpl 上に構築)

このファイルは自動微分ライブラリなしで、射影の数式が正しく動くことを
検証するためのリファレンス実装です。実際の学習には troll_torch.py の
PyTorch 版 (autograd 対応) を使用してください。
"""

import numpy as np


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(logits, axis=axis, keepdims=True)
    z = logits - m
    lse = np.log(np.sum(np.exp(z), axis=axis, keepdims=True)) + m
    return logits - lse


def sparsify(
    old_logits: np.ndarray,
    new_logits: np.ndarray,
    top_k: int = 64,
    mass_threshold: float = 1e-5,
):
    """
    論文 3.3節 "Scaling TROLL with Sparsity" の実装。

    語彙全体 (V次元) の分布を、旧方策の確率が高い上位トークンだけの
    部分空間 (K+1次元。+1 は残り確率質量をまとめた "rest" バケット) に
    縮約する。これにより射影のコストを O(V) から O(K) に落とす。

    Returns
    -------
    old_lp_sparse: [..., K+1] 旧方策の対数確率 (最後の次元が rest バケット)
    new_lp_sparse: [..., K+1] 新方策の対数確率
    top_idx:       [..., K]   元の語彙インデックス (rest バケットは含まない)
    """
    old_lp_full = log_softmax(old_logits, axis=-1)
    new_lp_full = log_softmax(new_logits, axis=-1)
    old_p_full = np.exp(old_lp_full)

    # 旧方策の確率が高い順にソート (トラストリージョンの中心は old policy なので
    # old policy 側の質量分布を基準にトークンを選ぶ)
    sorted_idx = np.argsort(-old_p_full, axis=-1)
    sorted_p = np.take_along_axis(old_p_full, sorted_idx, axis=-1)
    cum_p = np.cumsum(sorted_p, axis=-1)

    # 累積質量が 1-delta に達するまで、ただし最大 top_k 個まで
    keep_mask = cum_p <= (1.0 - mass_threshold)
    keep_mask[..., 0] = True  # 最低1トークンは必ず保持
    keep_mask[..., top_k:] = False

    k = max(int(keep_mask.sum(axis=-1).max()), 1)
    top_idx = sorted_idx[..., :k]

    old_lp_top = np.take_along_axis(old_lp_full, top_idx, axis=-1)
    new_lp_top = np.take_along_axis(new_lp_full, top_idx, axis=-1)

    # 残り確率質量を "rest" バケットとして1次元追加する
    old_rest_mass = np.clip(1.0 - np.exp(old_lp_top).sum(axis=-1, keepdims=True), 1e-12, None)
    new_rest_mass = np.clip(1.0 - np.exp(new_lp_top).sum(axis=-1, keepdims=True), 1e-12, None)

    old_lp_sparse = np.concatenate([old_lp_top, np.log(old_rest_mass)], axis=-1)
    new_lp_sparse = np.concatenate([new_lp_top, np.log(new_rest_mass)], axis=-1)

    return old_lp_sparse, new_lp_sparse, top_idx


def _kl_at_eta(eta: np.ndarray, old_lp: np.ndarray, new_lp: np.ndarray) -> np.ndarray:
    """
    与えられた eta での KL(pi_eta || old) を計算する。
    combined = (eta * log old + log new) / (eta + 1)
    pi_eta = softmax(combined)
    """
    eta = eta[..., None]
    combined = (eta * old_lp + new_lp) / (eta + 1.0)
    log_z = np.log(np.sum(np.exp(combined - combined.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)) \
        + combined.max(axis=-1, keepdims=True)
    log_pi = combined - log_z
    pi = np.exp(log_pi)
    kl = np.sum(pi * (log_pi - old_lp), axis=-1)
    return kl, log_pi


def project(
    old_lp: np.ndarray,
    new_lp: np.ndarray,
    epsilon: float = 0.05,
    bisection_iters: int = 60,
    max_expand: int = 40,
):
    """
    TROLL の閉形式射影 (プロジェクトページの式) を解く:

        argmin_pi  KL(pi || new)
        s.t.       KL(pi || old) <= epsilon

    解は pi ∝ exp( (eta* log(old) + log(new)) / (eta*+1) )
    (対数確率で扱っているので式中の log(new) は元の post の "new" ロジット項に対応)

    eta* は 1次元凸双対問題で、KL(pi_eta || old) = epsilon を満たす
    eta を二分探索 (bracketing) で求める。

    KL(new||old) <= epsilon ならば射影不要 (eta*=0, pi=new)。
    """
    batch_shape = old_lp.shape[:-1]

    # eta=0 (射影なし) での KL をまず計算
    kl0, log_pi0 = _kl_at_eta(np.zeros(batch_shape), old_lp, new_lp)
    needs_projection = kl0 > epsilon

    eta_star = np.zeros(batch_shape)
    log_pi_star = log_pi0.copy()

    if np.any(needs_projection):
        # 上限 hi を KL(pi_hi||old) <= epsilon になるまで倍々に拡張
        hi = np.ones(batch_shape)
        for _ in range(max_expand):
            kl_hi, _ = _kl_at_eta(hi, old_lp, new_lp)
            not_enough = np.logical_and(needs_projection, kl_hi > epsilon)
            if not np.any(not_enough):
                break
            hi = np.where(not_enough, hi * 2.0, hi)

        lo = np.zeros(batch_shape)
        for _ in range(bisection_iters):
            mid = (lo + hi) / 2.0
            kl_mid, _ = _kl_at_eta(mid, old_lp, new_lp)
            # KL は eta に関して単調減少。KL(mid) > eps なら mid はまだ小さすぎる -> lo=mid
            too_small = kl_mid > epsilon
            lo = np.where(np.logical_and(needs_projection, too_small), mid, lo)
            hi = np.where(np.logical_and(needs_projection, ~too_small), mid, hi)

        eta_final = (lo + hi) / 2.0
        eta_star = np.where(needs_projection, eta_final, eta_star)
        _, log_pi_proj = _kl_at_eta(eta_star, old_lp, new_lp)
        log_pi_star = np.where(needs_projection[..., None], log_pi_proj, log_pi0)

    kl_final, _ = _kl_at_eta(eta_star, old_lp, new_lp)
    return log_pi_star, eta_star, kl_final


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    V = 32000  # Qwen 系の語彙サイズ相当
    batch = 8

    old_logits = rng.normal(0, 2.0, size=(batch, V)).astype(np.float64)
    # new policy はold からわずかに(大きく)ずれた分布をシミュレート
    new_logits = old_logits + rng.normal(0, 3.0, size=(batch, V)).astype(np.float64)

    old_lp_sparse, new_lp_sparse, top_idx = sparsify(old_logits, new_logits, top_k=64, mass_threshold=1e-5)
    print(f"スパース化後の次元 (rest含む): {old_lp_sparse.shape[-1]} / 元の語彙数: {V}")

    log_pi, eta_star, kl_final = project(old_lp_sparse, new_lp_sparse, epsilon=0.05)

    print("\n=== 射影結果の検証 ===")
    for i in range(batch):
        print(f"row {i}: eta*={eta_star[i]:.4f}, KL(pi||old)={kl_final[i]:.6f} (制約: <=0.05)")

    assert np.all(kl_final <= 0.05 + 1e-4), "KL制約が破られています"
    print("\nOK: 全サンプルでKL(pi||old) <= epsilon が満たされています。")
