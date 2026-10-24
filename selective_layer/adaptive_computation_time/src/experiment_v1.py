# ACT (Graves 2016) Parity 実験 + 可視化ダッシュボード
# Colab では 1 セルに貼り付けて実行できます（要: torch, matplotlib, scipy。Colab 標準で入っています）

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ---------------- 設定（まず小さく、動いたら拡大） ----------------
D, H, M, EPS = 24, 128, 40, 0.01      # 入力次元, 隠れ次元, ポンダー上限, ε
ITERS, BS, LR = 4000, 128, 1e-3       # 学習回数, バッチ, 学習率
TAUS = [0, 1e-4, 1e-3, 1e-2, 1e-1]    # ACT の τ スイープ
FIXED_NS = [1, 2, 4, 8, 16, 32]       # 固定ステップのベースライン
MAIN_TAU = 1e-3                       # 詳細表示に使う ACT
SEED = 0
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)


# ---------------- データ ----------------
def make_batch(B):
    k = torch.randint(1, D + 1, (B, 1))
    ranks = torch.rand(B, D).argsort(1).argsort(1)
    mask = (ranks < k).float()
    sign = torch.randint(0, 2, (B, D)).float() * 2 - 1
    x = mask * sign
    y = ((x == 1).sum(1) % 2).float()
    return x.to(dev), y.to(dev), mask.sum(1).to(dev)


# ---------------- モデル（fixed_steps を指定すると固定ステップ版） ----------------
class Model(nn.Module):
    def __init__(self, fixed_steps=None):
        super().__init__()
        self.fixed = fixed_steps
        self.wx = nn.Linear(D + 1, H)
        self.wh = nn.Linear(H, H, bias=False)
        self.halt = nn.Linear(H, 1)
        self.out = nn.Linear(H, 1)
        nn.init.constant_(self.halt.bias, 1.0)

    def step(self, x, s, n):
        flag = x.new_full((x.size(0), 1), 1.0 if n == 0 else 0.0)
        inp = torch.cat([x if n == 0 else torch.zeros_like(x), flag], 1)
        return torch.tanh(self.wx(inp) + self.wh(s))

    def forward(self, x):
        B = x.size(0)
        s = x.new_zeros(B, H)
        if self.fixed is not None:                       # 固定ステップ
            for n in range(self.fixed):
                s = self.step(x, s, n)
            y = torch.sigmoid(self.out(s)).squeeze(-1)
            return y, x.new_full((B,), float(self.fixed))
        halt_sum, R, N, y_acc = (x.new_zeros(B) for _ in range(4))
        for n in range(M):                               # ACT
            s = self.step(x, s, n)
            h = torch.sigmoid(self.halt(s)).squeeze(-1)
            y = torch.sigmoid(self.out(s)).squeeze(-1)
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


# ---------------- 学習・評価 ----------------
def train(model, tau, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    log = {"it": [], "loss": [], "acc": [], "rho": []}
    for it in range(ITERS):
        x, t, _ = make_batch(BS)
        y, rho = model(x)
        y = y.clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(y, t) + tau * rho.mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 50 == 0:
            log["it"].append(it); log["loss"].append(loss.item())
            log["acc"].append(((y > 0.5).float() == t).float().mean().item())
            log["rho"].append(rho.mean().item())
        if it % 1000 == 0:
            print(f"[{name}] it={it} loss={loss.item():.3f} "
                  f"acc={log['acc'][-1]:.3f} ponder={log['rho'][-1]:.1f}")
    return log


@torch.no_grad()
def evaluate(model, n_batches=40, B=256):
    model.eval()
    ks, rhos, cs = [], [], []
    for _ in range(n_batches):
        x, t, k = make_batch(B)
        y, rho = model(x)
        ks.append(k.cpu()); rhos.append(rho.cpu())
        cs.append(((y > 0.5).float() == t).float().cpu())
    model.train()
    return (torch.cat(ks).numpy(), torch.cat(rhos).numpy(), torch.cat(cs).numpy())


# ---------------- 実験の実行 ----------------
act, fixed = {}, {}
for tau in TAUS:
    m = Model().to(dev)
    log = train(m, tau, f"ACT tau={tau:g}")
    act[tau] = dict(log=log, ev=evaluate(m))
for n in FIXED_NS:
    m = Model(fixed_steps=n).to(dev)
    log = train(m, 0.0, f"Fixed N={n}")
    fixed[n] = dict(log=log, ev=evaluate(m))


# ---------------- 可視化 ----------------
edges = np.linspace(0, D, 5)                      # 難易度バケット
centers = (edges[:-1] + edges[1:]) / 2

def bucket_acc(k, c):
    return [c[(k > lo) & (k <= hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]

def smooth(v, w=10):
    v = np.asarray(v, float)
    return np.convolve(v, np.ones(w) / w, mode="valid")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
cmap_t = plt.cm.viridis(np.linspace(0, 0.9, len(TAUS)))
cmap_f = plt.cm.Greys(np.linspace(0.35, 0.9, len(FIXED_NS)))

# (1) 難しさ(k) vs ポンダー時間 ヒートマップ
ax = axes[0, 0]
k, rho, c = act[MAIN_TAU]["ev"]
h = ax.hist2d(k, rho, bins=[D, 30], cmap="magma_r")
plt.colorbar(h[3], ax=ax, label="サンプル数")
mean_rho = [rho[k == v].mean() for v in range(1, D + 1)]
ax.plot(range(1, D + 1), mean_rho, "c-o", ms=3, label="平均")
r, _ = spearmanr(k, rho)
ax.set(title=f"(A) 難しさ vs ポンダー時間 (τ={MAIN_TAU:g}, Spearman={r:.2f})",
       xlabel="非ゼロ要素数 k", ylabel="ポンダー時間 ρ")
ax.legend()

# (2) 難易度バケット別 精度
ax = axes[0, 1]
for col, n in zip(cmap_f, FIXED_NS):
    kk, _, cc = fixed[n]["ev"]
    ax.plot(centers, bucket_acc(kk, cc), "--", color=col, lw=1.2, label=f"固定 N={n}")
kk, _, cc = act[MAIN_TAU]["ev"]
ax.plot(centers, bucket_acc(kk, cc), "r-o", lw=3, label=f"ACT τ={MAIN_TAU:g}")
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set(title="(D) 難易度バケット別の精度", xlabel="k（バケット中心）",
       ylabel="精度", ylim=(0.4, 1.02))
ax.legend(fontsize=8, ncol=2)

# (3) パレート図: 精度 vs 平均ポンダー時間
ax = axes[0, 2]
fx = [fixed[n]["ev"][1].mean() for n in FIXED_NS]
fy = [fixed[n]["ev"][2].mean() for n in FIXED_NS]
ax.plot(fx, fy, "ks--", label="固定ステップ")
for n, x_, y_ in zip(FIXED_NS, fx, fy):
    ax.annotate(f"N={n}", (x_, y_), textcoords="offset points", xytext=(4, -10), fontsize=8)
for col, tau in zip(cmap_t, TAUS):
    _, rr, cc = act[tau]["ev"]
    ax.scatter(rr.mean(), cc.mean(), s=140, color=col, edgecolor="r", zorder=3)
    ax.annotate(f"τ={tau:g}", (rr.mean(), cc.mean()), textcoords="offset points",
                xytext=(6, 6), fontsize=8, color="darkred")
ax.scatter([], [], s=140, color="w", edgecolor="r", label="ACT")
ax.set(title="(B) 精度 vs 計算量（左上ほど良い）", xlabel="平均ポンダー時間（対数軸）", ylabel="精度")
ax.set_xscale("log"); ax.legend(); ax.grid(alpha=0.3)

# (4) τ スイープ
ax = axes[1, 0]
xs = np.arange(len(TAUS))
accs = [act[t]["ev"][2].mean() for t in TAUS]
rhos = [act[t]["ev"][1].mean() for t in TAUS]
ax.plot(xs, accs, "b-o", label="精度")
ax.set_ylabel("精度", color="b"); ax.set_ylim(0.4, 1.02)
ax2 = ax.twinx()
ax2.plot(xs, rhos, "r-s", label="平均ポンダー時間")
ax2.set_ylabel("平均ポンダー時間", color="r")
ax.set_xticks(xs); ax.set_xticklabels([f"{t:g}" for t in TAUS])
ax.set(title="(C) τ スイープ：トレードオフ", xlabel="τ")

# (5) ポンダー時間の分布
ax = axes[1, 1]
for col, tau in zip(cmap_t, TAUS):
    _, rr, _ = act[tau]["ev"]
    ax.hist(rr, bins=np.linspace(0, M + 1, 41), histtype="step", lw=2, color=col,
            label=f"τ={tau:g} (上限到達 {np.mean(rr >= M - 0.5):.1%})")
ax.axvline(M, color="r", ls=":", label=f"上限 M={M}")
ax.set(title="ポンダー時間の分布（上限到達率に注意）", xlabel="ρ", ylabel="サンプル数")
ax.set_yscale("log"); ax.legend(fontsize=8)

# (6) 学習曲線（メイン ACT）
ax = axes[1, 2]
lg = act[MAIN_TAU]["log"]
w = 10
its = np.array(lg["it"])[w - 1:]
ax.plot(its, smooth(lg["acc"], w), "b-", label="精度")
ax.set_ylabel("精度", color="b"); ax.set_ylim(0.4, 1.02)
ax3 = ax.twinx()
ax3.plot(its, smooth(lg["rho"], w), "r-", label="ポンダー時間")
ax3.set_ylabel("ポンダー時間", color="r")
ax.set(title=f"学習曲線 (ACT τ={MAIN_TAU:g})", xlabel="イテレーション")

fig.suptitle(f"ACT Parity 実験ダッシュボード（D={D}, M={M}, seed={SEED}）", fontsize=15)
plt.tight_layout()
plt.savefig("act_dashboard.png", dpi=150)
plt.show()

# 数値サマリー
print(f"\n{'model':<14}{'acc':>7}{'ponder':>9}{'hit-M':>8}")
for n in FIXED_NS:
    _, rr, cc = fixed[n]["ev"]
    print(f"{'Fixed N='+str(n):<14}{cc.mean():>7.3f}{rr.mean():>9.2f}{'-':>8}")
for tau in TAUS:
    _, rr, cc = act[tau]["ev"]
    print(f"{'ACT τ='+format(tau, 'g'):<14}{cc.mean():>7.3f}{rr.mean():>9.2f}{np.mean(rr >= M - 0.5):>8.1%}")