# ACT Parity: 改善案1+2 実装版（第3版）
# 改善案1: 入力再提示 + 処理済みソフトマスク
# 改善案2: 不確実性(予測変動)をhalt入力に追加

import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# ---------------- 設定 ----------------
D, H, M, EPS = 16, 256, 32, 0.01
BS, LR, CLIP = 128, 1e-3, 5.0
MAX_ITERS = 30000
TAU = 1e-3
TAU_WARMUP = 2000
TAU_START_K_RATIO = 0.5
HALT_BIAS = -1.5
CURRICULUM_THRESH = 0.93
FORCE_ADVANCE_AFTER = 8
RUN_SANITY = True
SEED = 0
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)


# ---------------- データ ----------------
def make_batch(B, k_max=D, k_exact=False):
    if k_exact:
        k = torch.full((B, 1), k_max)
    else:
        k = torch.randint(1, k_max + 1, (B, 1))
    ranks = torch.rand(B, D).argsort(1).argsort(1)
    mask = (ranks < k).float()
    sign = torch.randint(0, 2, (B, D)).float() * 2 - 1
    x = mask * sign
    y = ((x == 1).sum(1) % 2).float()
    return x.to(dev), y.to(dev), mask.sum(1).to(dev)


# ---------------- モデル ----------------
class Model(nn.Module):
    def __init__(self, fixed_steps=None):
        super().__init__()
        self.fixed = fixed_steps
        # 入力: x(D) + processed(D) + flag(1) + step_feat(1) = 2D+2
        self.cell = nn.GRUCell(D * 2 + 2, H)
        # 次に処理する次元の重みを生成（ソフトマスク用）
        self.attention = nn.Linear(H, D)
        # 出力層
        self.out = nn.Linear(H, 1)
        # halt: 隠れ状態 + 不確実性(1) + 未処理割合(1) = H+2
        self.halt = nn.Linear(H + 2, 1)
        nn.init.constant_(self.halt.bias, HALT_BIAS)

    def step(self, x, s, n, processed):
        """1ステップのGRU更新と処理済みマスクの更新を返す"""
        step_feat = x.new_full((x.size(0), 1), n / M)
        flag = x.new_full((x.size(0), 1), 1.0 if n == 0 else 0.0)
        # 改善案1: xを毎ステップ再提示 + processedも入力
        inp = torch.cat([x, processed, flag, step_feat], 1)
        new_s = self.cell(inp, s)
        # 改善案1: ソフト的に「次に見る次元」を選び、processedを更新
        attn = torch.sigmoid(self.attention(new_s))
        new_processed = torch.clamp(processed + attn * (1 - processed), 0, 1)
        return new_s, new_processed

    def forward(self, x):
        B = x.size(0)
        s = x.new_zeros(B, H)
        # 改善案1: 処理済みマスクの初期化
        processed = x.new_zeros(B, D)
        if self.fixed is not None:
            for n in range(self.fixed):
                s, processed = self.step(x, s, n, processed)
            return torch.sigmoid(self.out(s)).squeeze(-1), x.new_full((B,), float(self.fixed))

        halt_sum, R, N, y_acc = (x.new_zeros(B) for _ in range(4))
        y_prev = None  # 改善案2: 前ステップの予測を保持

        for n in range(M):
            s, processed = self.step(x, s, n, processed)
            y = torch.sigmoid(self.out(s)).squeeze(-1)

            # 改善案2: 不確実性 = 前ステップとの予測差の絶対値
            if y_prev is None:
                uncertainty = x.new_zeros(B)
            else:
                uncertainty = (y - y_prev).abs()
            y_prev = y.clone()

            # 改善案1: 未処理次元の割合（残りが多いほど止まりにくい）
            remaining_ratio = 1 - processed.mean(dim=1)

            # halt機構への入力に不確実性と未処理割合を追加
            halt_inp = torch.cat([s, uncertainty.unsqueeze(1), remaining_ratio.unsqueeze(1)], dim=1)
            h = torch.sigmoid(self.halt(halt_inp)).squeeze(-1)

            still = halt_sum < 1 - EPS
            if not still.any():
                break
            h_eff = torch.where(still, h, torch.zeros_like(h))
            will_halt = still & ((halt_sum + h_eff >= 1 - EPS) | (n == M - 1))
            p = torch.where(will_halt, 1 - halt_sum, h_eff)
            R = R + torch.where(will_halt, 1 - halt_sum, torch.zeros_like(R))
            N = N + still.float()
            y_acc = y_acc + p * y
            halt_sum = halt_sum + p

        return y_acc, N + R


# ---------------- 評価 ----------------
@torch.no_grad()
def validate(model, k_max, k_exact, B=1024):
    model.eval()
    x, t, _ = make_batch(B, k_max, k_exact)
    y, rho = model(x)
    model.train()
    acc = ((y > 0.5).float() == t).float().mean().item()
    return acc, rho.mean().item(), (rho >= M - 0.5).float().mean().item()


@torch.no_grad()
def report_by_k(model, B=2048):
    model.eval()
    x, t, k = make_batch(B, D)
    y, rho = model(x)
    model.train()
    c = ((y > 0.5).float() == t).float()
    print(f"\n[最終評価] acc={c.mean():.3f}  ponder={rho.mean():.2f}  "
          f"hit-M={(rho >= M - 0.5).float().mean():.3f}")
    edges = np.linspace(0, D, 5)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (k > lo) & (k <= hi)
        if m.any():
            print(f"  k in ({lo:4.1f},{hi:4.1f}]  acc={c[m].mean():.3f}  ponder={rho[m].mean():.2f}")


# ---------------- 学習 ----------------
def train(model, tau_target, name, curriculum=True):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    k_max = 2 if curriculum else D
    tau_start = None
    stuck_counter = 0
    last_k_max = k_max
    t0 = time.time()

    for it in range(MAX_ITERS):
        x, t, _ = make_batch(BS, k_max)
        y, rho = model(x)
        y = y.clamp(1e-6, 1 - 1e-6)

        if k_max >= D * TAU_START_K_RATIO:
            if tau_start is None:
                tau_start = it
            tau = tau_target * min(1.0, (it - tau_start) / TAU_WARMUP)
        else:
            tau = 0.0

        loss = F.binary_cross_entropy(y, t) + tau * rho.mean()

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()

        if it % 200 == 0:
            va, vr, vh = validate(model, k_max, k_exact=True)

            if k_max == last_k_max:
                stuck_counter += 1
            else:
                stuck_counter = 0
                last_k_max = k_max

            print(f"[{name}] it={it:5d} k_max={k_max:2d} tau={tau:.1e} "
                  f"loss={loss.item():.3f} val_acc(k={k_max})={va:.3f} "
                  f"ponder={vr:.1f} hit-M={vh:.2f} stuck={stuck_counter} ({time.time()-t0:.0f}s)")

            if curriculum and k_max < D:
                if va > CURRICULUM_THRESH or stuck_counter >= FORCE_ADVANCE_AFTER:
                    k_max += 1
                    stuck_counter = 0
                    print(f"[{name}] >>> k_max advanced to {k_max} (va={va:.3f})")

            if k_max >= D and tau_start is not None and it - tau_start > TAU_WARMUP + 2000:
                full_acc, _, _ = validate(model, D, k_exact=False)
                if full_acc > 0.95:
                    print(f"[{name}] 収束と判断して終了 (acc={full_acc:.3f})")
                    break

    return model


# ---------------- 実行 ----------------
if RUN_SANITY:
    print("=== Sanity: 固定ステップ(N=8)で学習できるか ===")
    sanity = Model(fixed_steps=8).to(dev)
    train(sanity, 0.0, "Fixed N=8")
    report_by_k(sanity)
    print("→ これが学習できない場合、ACT ではなくタスク設定/最適化の問題です。\n")

print("=== ACT (改善案1+2) ===")
act = Model().to(dev)
train(act, TAU, f"ACT tau={TAU:g}")
report_by_k(act)