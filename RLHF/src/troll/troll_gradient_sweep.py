"""
更新幅 t を 0 -> 大 とスイープしながら、
  - 重要度比 r(t)
  - PPOクリップ目的関数の勾配
  - TROLL目的関数の勾配
を観察する。

狙い: PPOは r が [1-eps_ppo, 1+eps_ppo] の外に出た瞬間、勾配が
「厳密にゼロ」になる(不連続なカットオフ)。
TROLLは射影によって勾配が滑らかに減衰していくが、ゼロにはならない
(射影が微分可能であるため)。
"""

import numpy as np
from troll_numpy_core import sparsify, project, log_softmax
from troll_loss_compare import ppo_clip_objective, troll_objective, finite_diff_grad


def main():
    rng = np.random.default_rng(7)
    V = 1000
    batch = 1
    action_idx = np.array([5])
    advantage = np.array([1.0])  # 正のアドバンテージ: 確率を上げたい方向

    old_logits = rng.normal(0, 1.0, size=(1, V))

    # 「実際にサンプルされたトークン(action_idx)の logit だけ」を単調に増やしていく
    # -> 重要度比 r が単調に増加していく状況を作る
    ts = np.linspace(0.0, 6.0, 25)

    print(f"{'t':>6} | {'ratio r':>10} | {'PPO grad':>12} | {'TROLL grad':>12} | 備考")
    print("-" * 70)

    for t in ts:
        new_logits = old_logits.copy()
        new_logits[0, action_idx[0]] += t  # 対象トークンのlogitだけシフト

        _, ratio = ppo_clip_objective(old_logits, new_logits, action_idx, advantage, clip_eps=0.2)

        def ppo_f(logits_row_2d):
            obj, _ = ppo_clip_objective(old_logits, logits_row_2d, action_idx, advantage, clip_eps=0.2)
            return obj[0]

        def troll_f(logits_row_2d):
            obj, _, _ = troll_objective(old_logits, logits_row_2d, action_idx, advantage,
                                         epsilon=0.05, top_k=64)
            return obj[0]

        grad_ppo = finite_diff_grad(ppo_f, new_logits[0], action_idx[0], h=1e-3)
        grad_troll = finite_diff_grad(troll_f, new_logits[0], action_idx[0], h=1e-3)

        note = ""
        if ratio[0] > 1.2:
            note = "<- PPOクリップ範囲外(r>1.2)"
        print(f"{t:6.2f} | {ratio[0]:10.4f} | {grad_ppo:12.6f} | {grad_troll:12.6f} | {note}")


if __name__ == "__main__":
    main()
