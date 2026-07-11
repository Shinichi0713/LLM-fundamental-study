"""
TROLL (Trust Regions improve Reinforcement Learning for Large Language Models)
の PyTorch (autograd 対応) 実装。

論文: https://arxiv.org/abs/2510.03817  (ICLR 2026, Oral)
公式実装: https://github.com/niklasfreymuth/TROLL

NOTE: このコードは troll_numpy_core.py / troll_loss_compare.py / troll_gradient_sweep.py
で検証したアルゴリズムをそのまま PyTorch に移植したものです。
本環境はネットワークが遮断されておりPyTorchをインストール・実行検証できないため、
このファイル自体は「実行未検証」です。ロジックはNumPy版と1対1で対応させてあり、
NumPy版で全ての性質(KL制約の充足、射影不要時のeta=0への収束、
クリップ境界での勾配消失 vs TROLLでの勾配保持)を検証済みです。
ご自身の環境で `pytest troll_torch_test.py` 等により最終確認してください。

使い方の想定 (verl 等の学習ループ内):

    old_logits: [B, T, V]  (rollout 収集時点の方策のロジット。stop-gradient対象)
    new_logits: [B, T, V]  (現在最適化中の方策のロジット。勾配が流れる)
    actions:    [B, T]     (実際にサンプルされたトークンID)
    advantages: [B, T]     (トークンレベルのアドバンテージ)
    loss_mask:  [B, T]     (パディング等を除外するマスク)

    loss, info = troll_loss(old_logits, new_logits, actions, advantages, loss_mask)
    loss.backward()
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sparsify(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
    top_k: int = 64,
    mass_threshold: float = 1e-5,
):
    """
    語彙全体 [..., V] を、旧方策の確率質量が高い上位 K トークン + "rest" バケット
    の [..., K+1] 次元に縮約する。

    old_logits, new_logits には勾配が必要な情報 (new_logits) と不要な情報 (old_logits) が
    混在するため、呼び出し側で old_logits は事前に detach しておくこと。

    Returns
    -------
    old_lp_sparse, new_lp_sparse: [..., K+1]
    top_idx: [..., K]  (rest バケットの分は含まない、元の語彙インデックス)
    """
    old_lp_full = F.log_softmax(old_logits, dim=-1)
    new_lp_full = F.log_softmax(new_logits, dim=-1)
    old_p_full = old_lp_full.exp()

    sorted_p, sorted_idx = torch.sort(old_p_full, dim=-1, descending=True)
    cum_p = torch.cumsum(sorted_p, dim=-1)

    keep_mask = cum_p <= (1.0 - mass_threshold)
    keep_mask[..., 0] = True
    if top_k < keep_mask.shape[-1]:
        keep_mask[..., top_k:] = False

    k = max(int(keep_mask.sum(dim=-1).max().item()), 1)
    top_idx = sorted_idx[..., :k]

    old_lp_top = torch.gather(old_lp_full, -1, top_idx)
    new_lp_top = torch.gather(new_lp_full, -1, top_idx)

    old_rest_mass = (1.0 - old_lp_top.exp().sum(dim=-1, keepdim=True)).clamp_min(1e-12)
    new_rest_mass = (1.0 - new_lp_top.exp().sum(dim=-1, keepdim=True)).clamp_min(1e-12)

    old_lp_sparse = torch.cat([old_lp_top, old_rest_mass.log()], dim=-1)
    new_lp_sparse = torch.cat([new_lp_top, new_rest_mass.log()], dim=-1)

    return old_lp_sparse, new_lp_sparse, top_idx


def _kl_at_eta(eta: torch.Tensor, old_lp: torch.Tensor, new_lp: torch.Tensor):
    """KL(pi_eta || old) と log(pi_eta) を返す。 eta: [...,] (最後の次元なし)"""
    eta_ = eta.unsqueeze(-1)
    combined = (eta_ * old_lp + new_lp) / (eta_ + 1.0)
    log_pi = combined - torch.logsumexp(combined, dim=-1, keepdim=True)
    pi = log_pi.exp()
    kl = (pi * (log_pi - old_lp)).sum(dim=-1)
    return kl, log_pi


@torch.no_grad()
def _find_eta_star(old_lp: torch.Tensor, new_lp: torch.Tensor, epsilon: float,
                    bisection_iters: int = 30, max_expand: int = 30):
    """
    KL(pi_eta || old) = epsilon を満たす eta* を二分探索で求める。
    勾配は不要 (eta* は定数として扱い、射影後の log_pi を計算する際に
    再度 differentiable な形で _kl_at_eta を呼び出す)。
    """
    batch_shape = old_lp.shape[:-1]
    device = old_lp.device

    kl0, _ = _kl_at_eta(torch.zeros(batch_shape, device=device), old_lp, new_lp)
    needs_projection = kl0 > epsilon

    hi = torch.ones(batch_shape, device=device)
    for _ in range(max_expand):
        kl_hi, _ = _kl_at_eta(hi, old_lp, new_lp)
        not_enough = needs_projection & (kl_hi > epsilon)
        if not not_enough.any():
            break
        hi = torch.where(not_enough, hi * 2.0, hi)

    lo = torch.zeros(batch_shape, device=device)
    for _ in range(bisection_iters):
        mid = (lo + hi) / 2.0
        kl_mid, _ = _kl_at_eta(mid, old_lp, new_lp)
        too_small = kl_mid > epsilon  # KLはetaに関して単調減少
        lo = torch.where(needs_projection & too_small, mid, lo)
        hi = torch.where(needs_projection & ~too_small, mid, hi)

    eta_star = torch.where(needs_projection, (lo + hi) / 2.0, torch.zeros_like(lo))
    return eta_star


def project(old_lp: torch.Tensor, new_lp: torch.Tensor, epsilon: float = 0.05):
    """
    TROLL のトラストリージョン射影。
    eta* の探索自体は no_grad (2分探索は微分不可能な離散手続きのため)、
    ただし射影後の分布 log_pi は eta* を「定数」として new_lp から
    再計算するので、new_lp -> log_pi への経路には勾配が流れる
    (closed-form gradient)。
    """
    eta_star = _find_eta_star(old_lp.detach(), new_lp.detach(), epsilon)
    # eta_star は定数として扱い、log_pi は differentiable な new_lp から計算しなおす
    kl_final, log_pi = _kl_at_eta(eta_star, old_lp, new_lp)
    return log_pi, eta_star, kl_final


def gather_action_logprob(log_pi_sparse: torch.Tensor, top_idx: torch.Tensor,
                           old_lp_sparse: torch.Tensor, actions: torch.Tensor):
    """
    実際にサンプルされたトークン (actions) に対する、射影後分布での対数確率を取り出す。
    top_idx に含まれていれば直接、含まれていなければ rest バケットの
    射影前後の比率で近似する。
    """
    B_shape = actions.shape
    K = top_idx.shape[-1]

    match = (top_idx == actions.unsqueeze(-1))  # [..., K]
    found = match.any(dim=-1)  # [...]
    match_idx = match.float().argmax(dim=-1)  # [...] (found=Falseの場合は無視される)

    log_pi_at_action_if_found = torch.gather(log_pi_sparse[..., :K], -1, match_idx.unsqueeze(-1)).squeeze(-1)

    # rest バケットのフォールバック: 元のoldの対数確率 + (rest部分の射影前後の比率)
    # ここでは呼び出し側から元の old_log_prob (射影前, フル語彙) を渡してもらう設計が
    # 望ましいが、簡略化のため rest バケットの比率のみを使う近似を採用する。
    old_rest_lp = old_lp_sparse[..., -1]
    pi_rest_lp = log_pi_sparse[..., -1]
    rest_ratio_log = pi_rest_lp - old_rest_lp  # log( rest後 / rest前 )

    return log_pi_at_action_if_found, found, rest_ratio_log


def troll_loss(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    epsilon: float = 0.05,
    alpha: float = 1.0,
    top_k: int = 64,
    mass_threshold: float = 1e-5,
):
    """
    TROLL の目的関数 (最大化したい量) を計算し、その符号反転 (最小化損失) を返す。

    J_TROLL = E[ (pi(a)/old(a)) * A ]  -  alpha * KL( new || stopgrad(pi) )

    Parameters
    ----------
    old_logits: [B, T, V]  rollout時点のロジット (内部でdetachされる)
    new_logits: [B, T, V]  現在最適化中の方策のロジット (勾配が流れる)
    actions:    [B, T]     long tensor. サンプルされたトークンID
    advantages: [B, T]     float tensor
    loss_mask:  [B, T]     1=有効トークン, 0=無視 (パディング等)
    """
    old_logits = old_logits.detach()

    old_lp_sparse, new_lp_sparse, top_idx = sparsify(old_logits, new_logits, top_k, mass_threshold)
    log_pi_sparse, eta_star, kl_final = project(old_lp_sparse, new_lp_sparse, epsilon)

    old_lp_full_action = torch.gather(
        F.log_softmax(old_logits, dim=-1), -1, actions.unsqueeze(-1)
    ).squeeze(-1)

    log_pi_at_action, found, rest_ratio_log = gather_action_logprob(
        log_pi_sparse, top_idx, old_lp_sparse, actions
    )

    # found=True: 射影後分布での直接の対数確率を使用
    # found=False (top_kに含まれない稀なケース): rest バケットの比率で近似
    log_pi_action_full = torch.where(
        found, log_pi_at_action, old_lp_full_action + rest_ratio_log
    )

    ratio = (log_pi_action_full - old_lp_full_action).exp()
    is_objective = ratio * advantages

    new_p_sparse = new_lp_sparse.exp()
    kl_regression = (new_p_sparse * (new_lp_sparse - log_pi_sparse.detach())).sum(dim=-1)

    objective = is_objective - alpha * kl_regression

    if loss_mask is not None:
        objective = objective * loss_mask
        denom = loss_mask.sum().clamp_min(1.0)
        loss = -(objective.sum() / denom)
    else:
        loss = -objective.mean()

    info = {
        "ratio_mean": ratio.detach().mean(),
        "kl_to_old_mean": kl_final.detach().mean(),
        "eta_star_mean": eta_star.detach().mean(),
        "kl_regression_mean": kl_regression.detach().mean(),
    }
    return loss, info


def ppo_clip_loss(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    clip_eps: float = 0.2,
):
    """比較用の標準PPOクリップ損失。"""
    old_logits = old_logits.detach()
    old_lp = torch.gather(F.log_softmax(old_logits, dim=-1), -1, actions.unsqueeze(-1)).squeeze(-1)
    new_lp = torch.gather(F.log_softmax(new_logits, dim=-1), -1, actions.unsqueeze(-1)).squeeze(-1)

    ratio = (new_lp - old_lp).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    objective = torch.minimum(unclipped, clipped)

    if loss_mask is not None:
        objective = objective * loss_mask
        denom = loss_mask.sum().clamp_min(1.0)
        loss = -(objective.sum() / denom)
    else:
        loss = -objective.mean()

    return loss, {"ratio_mean": ratio.detach().mean()}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, V = 4, 16, 32000
    old_logits = torch.randn(B, T, V) * 2.0
    new_logits = (old_logits + torch.randn(B, T, V) * 0.5).requires_grad_(True)
    actions = torch.randint(0, V, (B, T))
    advantages = torch.randn(B, T)
    loss_mask = torch.ones(B, T)

    loss, info = troll_loss(old_logits, new_logits, actions, advantages, loss_mask)
    loss.backward()
    print("TROLL loss:", loss.item())
    print("info:", {k: v.item() for k, v in info.items()})
    print("grad norm (new_logits):", new_logits.grad.norm().item())
