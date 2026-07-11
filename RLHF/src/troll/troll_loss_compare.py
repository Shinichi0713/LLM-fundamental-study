"""
TROLL目的関数 vs PPOクリップ目的関数 の比較。

論文の主張:
  「PPOクリップはクリップ範囲外で勾配を完全に打ち切るが、
   TROLLは射影を通して常に微分可能で勾配が流れ続ける」

これを有限差分 (finite difference) で数値的に検証する。
(このマシンにはPyTorchをインストールできない [ネットワーク制限] ため、
 NumPyの数値微分でアルゴリズムの性質そのものを検証する。
 実運用版の自動微分コードは troll_torch.py を参照。)
"""

import numpy as np
from troll_numpy_core import sparsify, project, log_softmax


def troll_objective(old_logits, new_logits, action_idx, advantage, epsilon=0.05, alpha=1.0,
                     top_k=64, mass_threshold=1e-5):
    """
    J_TROLL = (pi(a)/old(a)) * A  -  alpha * KL(new || stopgrad(pi))

    1トークン・1サンプル分のスカラーを返す (batch対応)。
    action_idx: 実際にサンプリングされたトークンの語彙インデックス
    """
    old_lp_sparse, new_lp_sparse, top_idx = sparsify(old_logits, new_logits, top_k, mass_threshold)
    log_pi, eta_star, kl_final = project(old_lp_sparse, new_lp_sparse, epsilon)

    batch = old_logits.shape[0]
    ratios = np.zeros(batch)
    for b in range(batch):
        # サンプルされたトークンが top_k に含まれるか探す
        pos = np.where(top_idx[b] == action_idx[b])[0]
        old_lp_full = log_softmax(old_logits[b:b+1], axis=-1)[0, action_idx[b]]
        if len(pos) > 0:
            pi_lp_action = log_pi[b, pos[0]]
        else:
            # rest バケットに含まれる場合: rest内で比率が保存されると近似し、
            # rest バケットレベルの射影後/射影前 比率を、元のold確率に掛けて近似する。
            old_rest_lp = old_lp_sparse[b, -1]
            pi_rest_lp = log_pi[b, -1]
            pi_lp_action = old_lp_full + (pi_rest_lp - old_rest_lp)
        ratios[b] = np.exp(pi_lp_action - old_lp_full)

    # 重要度重み付き目的 (advantage側は最大化なので loss は符号反転して使う想定。
    # ここでは "目的関数" 自体、つまり最大化したい値を返す)
    is_objective = ratios * advantage

    # KL回帰項: KL(new || stopgrad(pi))。 stopgrad(pi) は定数として扱うので
    # NumPyでは単に log_pi をそのまま定数として使えばよい (バックプロップしないから)。
    new_p_sparse = np.exp(new_lp_sparse)
    kl_regression = np.sum(new_p_sparse * (new_lp_sparse - log_pi), axis=-1)

    objective = is_objective - alpha * kl_regression
    return objective, ratios, kl_final


def ppo_clip_objective(old_logits, new_logits, action_idx, advantage, clip_eps=0.2):
    """
    標準的なPPOクリップ目的関数 (1トークン分)。
    """
    batch = old_logits.shape[0]
    old_lp_full = log_softmax(old_logits, axis=-1)
    new_lp_full = log_softmax(new_logits, axis=-1)

    old_lp_action = old_lp_full[np.arange(batch), action_idx]
    new_lp_action = new_lp_full[np.arange(batch), action_idx]
    ratio = np.exp(new_lp_action - old_lp_action)

    unclipped = ratio * advantage
    clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    objective = np.minimum(unclipped, clipped)
    return objective, ratio


def finite_diff_grad(f, x_row, idx, h=1e-3):
    """
    f: バッチ次元込みの2次元 logits [1, V] を受け取り、スカラーの目的関数値を返す関数
    x_row: 1次元の logits [V] (batch=1 の1行分)
    idx: 勾配を見たいトークンのインデックス
    """
    x_plus = x_row.copy()
    x_plus[idx] += h
    x_minus = x_row.copy()
    x_minus[idx] -= h
    return (f(x_plus[None, :]) - f(x_minus[None, :])) / (2 * h)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    V = 1000  # 有限差分を高速に回すため小さめの語彙で検証
    batch = 1
    action_idx = np.array([5])
    advantage = np.array([1.0])

    old_logits = rng.normal(0, 1.0, size=(batch, V))

    # ケース1: 新方策が旧方策から「大きく」ずれている
    #          (PPOクリップなら比率がクリップ範囲外に出て勾配ゼロになりやすい状況)
    large_shift = rng.normal(0, 4.0, size=(batch, V))
    new_logits_large = old_logits + large_shift
    # 実際に比率がクリップ外か確認
    _, ratio_large = ppo_clip_objective(old_logits, new_logits_large, action_idx, advantage)
    print(f"[大きな更新] PPO重要度比率 r = {ratio_large[0]:.4f} (クリップ範囲 [0.8, 1.2])")

    def ppo_f_large(logits):
        obj, _ = ppo_clip_objective(old_logits, logits, action_idx, advantage)
        return obj[0]

    def troll_f_large(logits):
        obj, _, _ = troll_objective(old_logits, logits, action_idx, advantage, epsilon=0.05, top_k=64)
        return obj[0]

    # クリップされている「その」トークン(action_idx)自身のロジットに対する勾配を見る
    grad_ppo = finite_diff_grad(ppo_f_large, new_logits_large[0], action_idx[0])
    grad_troll = finite_diff_grad(troll_f_large, new_logits_large[0], action_idx[0])

    print(f"  PPOクリップ目的関数の勾配 (該当トークンのlogitに対して): {grad_ppo:.6f}")
    print(f"  TROLL目的関数の勾配     (該当トークンのlogitに対して): {grad_troll:.6f}")
    print(f"  -> {'PPOは勾配ゼロ(クリップにより打ち切り)。TROLLは非ゼロの勾配を保持。' if abs(grad_ppo) < 1e-8 and abs(grad_troll) > 1e-8 else '両者とも勾配あり(この設定では未クリップの可能性)'}")

    print()

    # ケース2: 新方策がほぼ旧方策と同じ (クリップ範囲内)
    small_shift = rng.normal(0, 0.01, size=(batch, V))
    new_logits_small = old_logits + small_shift
    _, ratio_small = ppo_clip_objective(old_logits, new_logits_small, action_idx, advantage)
    print(f"[小さな更新] PPO重要度比率 r = {ratio_small[0]:.4f} (クリップ範囲内)")

    def ppo_f_small(logits):
        obj, _ = ppo_clip_objective(old_logits, logits, action_idx, advantage)
        return obj[0]

    def troll_f_small(logits):
        obj, _, _ = troll_objective(old_logits, logits, action_idx, advantage, epsilon=0.05, top_k=64)
        return obj[0]

    grad_ppo_small = finite_diff_grad(ppo_f_small, new_logits_small[0], action_idx[0])
    grad_troll_small = finite_diff_grad(troll_f_small, new_logits_small[0], action_idx[0])
    print(f"  PPOクリップ目的関数の勾配: {grad_ppo_small:.6f}")
    print(f"  TROLL目的関数の勾配:     {grad_troll_small:.6f}")
    print("  -> クリップ範囲内では両者とも勾配が流れる(同様の振る舞い)。")
