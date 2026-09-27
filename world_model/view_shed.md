すでに `stealth_route_planner.py` として配信済みですが、コードの全文をこちらにも記載いたします。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stealth_route_planner.py

DSMデータと観測者（人）の位置から、観測者に見つからない経路をA*探索で計算する。

機能:
  - DSMからviewshed（可視領域）を計算（レイ投射法による高速版）
  - 複数観測者の統合可視性スコアを生成
  - A*探索で「見えないセルのみ」を通るステルス経路を計画
  - 可視性リスクを考慮した経路最適化（距離 vs リスクのトレードオフ）

使い方:
  1. DSMデータを NumPy 2D array として読み込む
  2. 観測者位置のリスト、スタート・ゴール位置を指定
  3. compute_combined_visibility() で可視性スコアを計算
  4. astar_stealth_risk() でステルス経路を探索
  5. 結果を可視化・保存

依存ライブラリ:
  - numpy
  - matplotlib
  - heapq (標準ライブラリ)
  - math (標準ライブラリ)
"""

import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import math
import json


# ============================================================
# 1. Viewshed計算（レイ投射法による高速版）
# ============================================================

def compute_viewshed_fast(dsm, observer, observer_height=1.6, cell_size=5.0, max_range=None):
    """
    DSM上のobserver位置からのviewshedを高速に計算する（レイ投射法）。

    Parameters
    ----------
    dsm : 2D numpy array
        標高データ（メートル）。shape = (rows, cols)
    observer : tuple (row, col)
        観測者のグリッド位置
    observer_height : float
        観測者の眼の高さ（地表面からの高さ、メートル）。デフォルト1.6m
    cell_size : float
        DSMのセル1辺の長さ（メートル）。デフォルト5.0m
    max_range : float or None
        最大視程（メートル）。Noneの場合は全域を計算。

    Returns
    -------
    viewshed : 2D numpy array (bool)
        True = 観測者から見えるセル、False = 見えないセル
    """
    rows, cols = dsm.shape
    obs_r, obs_c = observer
    obs_elev = dsm[obs_r, obs_c] + observer_height

    viewshed = np.zeros((rows, cols), dtype=bool)
    viewshed[obs_r, obs_c] = True

    # 角度方向のサンプリング数（精度と速度のトレードオフ）
    n_angles = max(rows, cols) * 4

    for angle_idx in range(n_angles):
        angle = 2 * math.pi * angle_idx / n_angles
        dc = math.cos(angle)
        ds = math.sin(angle)

        max_step = int(math.sqrt(rows**2 + cols**2)) + 1
        max_elev_angle = -float('inf')

        for step in range(1, max_step):
            r = obs_r + step * ds
            c = obs_c + step * dc

            ri = int(round(r))
            ci = int(round(c))

            if not (0 <= ri < rows and 0 <= ci < cols):
                break

            dist = math.sqrt(((ri - obs_r) * cell_size)**2 + ((ci - obs_c) * cell_size)**2)
            if max_range is not None and dist > max_range:
                break

            delta_h = dsm[ri, ci] - obs_elev
            elev_angle = math.degrees(math.atan2(delta_h, dist)) if dist > 0 else 90

            if elev_angle >= max_elev_angle:
                viewshed[ri, ci] = True
                max_elev_angle = elev_angle
            # 遮られたセルは見えないまま

    return viewshed


# ============================================================
# 2. 複数観測者の統合可視性スコア
# ============================================================

def compute_combined_visibility(dsm, observers, observer_height=1.6, cell_size=5.0, max_range=None):
    """
    複数観測者からの統合可視性を計算する。

    Parameters
    ----------
    dsm : 2D numpy array
        標高データ
    observers : list of tuple (row, col)
        観測者位置のリスト
    observer_height : float
        各観測者の眼の高さ（メートル）
    cell_size : float
        セルサイズ（メートル）
    max_range : float or None
        各観測者の最大視程（メートル）

    Returns
    -------
    visibility_score : 2D numpy array (int)
        各セルが見える観測者の数。
        0 = 誰からも見えない（完全ステルス）
        1 = 1人から見える、2 = 2人から見える、...
    """
    rows, cols = dsm.shape
    visibility_score = np.zeros((rows, cols), dtype=np.int32)

    for i, obs in enumerate(observers):
        print(f"  観測者 {i+1}/{len(observers)} のviewshedを計算中...")
        vs = compute_viewshed_fast(dsm, obs, observer_height, cell_size, max_range)
        visibility_score += vs.astype(np.int32)

    return visibility_score


# ============================================================
# 3. A*経路探索（可視性リスク最小化）
# ============================================================

def astar_stealth_risk(dsm, visibility_score, start, goal,
                       cell_size=5.0, risk_weight=100.0, max_visibility=0):
    """
    visibility_scoreを考慮したステルス経路をA*探索で計算する。

    Parameters
    ----------
    dsm : 2D numpy array
        標高データ（経路探索の境界チェックなどに使用）
    visibility_score : 2D numpy array (int)
        compute_combined_visibility() の出力。
        各セルが何人の観測者から見えるかを示すスコア。
    start : tuple (row, col)
        出発点のグリッド位置
    goal : tuple (row, col)
        目標点のグリッド位置
    cell_size : float
        セルサイズ（メートル）。デフォルト5.0m
    risk_weight : float
        可視性1あたりの追加移動コスト。
        大きいほどリスク回避を重視する。デフォルト100.0
    max_visibility : int
        許容する最大可視性。
        0 = 完全ステルスのみ（誰からも見えないセルのみ通過）
        1 = 1人から見えるセルまで許容
        2 = 2人から見えるセルまで許容

    Returns
    -------
    path : list of (row, col) or None
        見つかった経路。Noneの場合は経路なし。
    info : dict or None
        経路の統計情報:
          - steps: ステップ数
          - total_distance_m: 総移動距離（メートル）
          - max_visibility_on_path: 経路上の最大可視性スコア
          - risky_cells_count: 可視性>0のセル数
          - risky_cells_ratio: リスクセルの比率（0〜1）
    """
    rows, cols = dsm.shape

    # 8方向の移動（上下左右＋斜め）
    neighbors = [
        (-1, 0, cell_size), (1, 0, cell_size), (0, -1, cell_size), (0, 1, cell_size),
        (-1, -1, cell_size * math.sqrt(2)), (-1, 1, cell_size * math.sqrt(2)),
        (1, -1, cell_size * math.sqrt(2)), (1, 1, cell_size * math.sqrt(2))
    ]

    def heuristic(r1, c1, r2, c2):
        """ユークリッド距離による推定コスト（A*のヒューリスティック）"""
        return math.sqrt(((r1 - r2) * cell_size)**2 + ((c1 - c2) * cell_size)**2)

    counter = 0
    open_set = []
    h0 = heuristic(start[0], start[1], goal[0], goal[1])
    heappush(open_set, (h0, counter, start[0], start[1]))

    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: h0}
    visited = set()

    while open_set:
        _, _, r, c = heappop(open_set)

        if (r, c) in visited:
            continue
        visited.add((r, c))

        if (r, c) == goal:
            # 経路復元
            path = []
            node = (r, c)
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()

            # 統計情報の計算
            total_dist = 0.0
            max_vis_on_path = 0
            risky_cells = 0
            for i in range(1, len(path)):
                dr = (path[i][0] - path[i-1][0]) * cell_size
                dc = (path[i][1] - path[i-1][1]) * cell_size
                total_dist += math.sqrt(dr**2 + dc**2)
                vis = visibility_score[path[i]]
                max_vis_on_path = max(max_vis_on_path, vis)
                if vis > 0:
                    risky_cells += 1

            info = {
                'steps': len(path),
                'total_distance_m': total_dist,
                'max_visibility_on_path': max_vis_on_path,
                'risky_cells_count': risky_cells,
                'risky_cells_ratio': risky_cells / len(path) if path else 0
            }
            return path, info

        for dr, dc, move_cost in neighbors:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            # 可視性チェック: 許容値を超えるセルは通過不可
            vis = visibility_score[nr, nc]
            if vis > max_visibility:
                continue

            # 移動コスト = 距離 + 可視性リスク
            cell_risk = risk_weight * vis
            tentative_g = g_score[(r, c)] + move_cost + cell_risk

            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                came_from[(nr, nc)] = (r, c)
                g_score[(nr, nc)] = tentative_g
                f = tentative_g + heuristic(nr, nc, goal[0], goal[1])
                f_score[(nr, nc)] = f
                counter += 1
                heappush(open_set, (f, counter, nr, nc))

    return None, None


# ============================================================
# 4. サンプルDSM生成（デモ用）
# ============================================================

def generate_sample_dsm(size=200, cell_size=5.0, seed=456):
    """
    デモ用のサンプルDSMを生成する。

    中央に南北方向の谷（低地）があり、東側の高台に観測者を配置する想定の地形。
    谷の底は低く、両側の壁が高いため、東側からの視線を地形が遮る。

    Parameters
    ----------
    size : int
        DSMの縦横サイズ（正方形）。デフォルト200
    cell_size : float
        セルサイズ（メートル）。デフォルト5.0m
    seed : int
        ランダムシード。デフォルト456

    Returns
    -------
    dsm : 2D numpy array
        標高データ（メートル）
    cell_size : float
        セルサイズ（メートル）
    """
    dsm = np.ones((size, size), dtype=np.float32) * 40.0
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    # 中央の谷（南北方向に走る低地）
    valley_center = size // 2
    valley_width = 15

    for c in range(size):
        dist_from_valley = abs(c - valley_center)
        if dist_from_valley < valley_width:
            dsm[:, c] = 5.0
        elif dist_from_valley < valley_width + 20:
            ratio = (dist_from_valley - valley_width) / 20
            dsm[:, c] = 5.0 + 35.0 * ratio
        else:
            dsm[:, c] = 40.0

    # 小さな丘や凹凸を加えて自然な地形に
    np.random.seed(seed)
    for _ in range(20):
        cx, cy = np.random.randint(10, size-10, size=2)
        h = np.random.uniform(-5, 8)
        r = np.random.uniform(5, 12)
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        dsm += h * np.exp(-dist**2 / (2 * r**2))

    # 谷の底はクリップして低く保つ
    dsm[:, valley_center-valley_width:valley_center+valley_width] = np.clip(
        dsm[:, valley_center-valley_width:valley_center+valley_width], 0, 15
    )

    return dsm, cell_size


# ============================================================
# 5. 可視化
# ============================================================

def visualize_result(dsm, observers, start, goal, visibility_score,
                     path_stealth, path_risky,
                     save_path='stealth_route_result.png'):
    """
    結果を4パネルの図で可視化する。

    Parameters
    ----------
    dsm : 2D numpy array
        標高データ
    observers : list of tuple
        観測者位置のリスト
    start, goal : tuple
        スタート・ゴール位置
    visibility_score : 2D numpy array (int)
        統合可視性スコア
    path_stealth : list of tuple or None
        完全ステルス経路
    path_risky : list of tuple or None
        リスク許容経路
    save_path : str
        保存先のファイルパス
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # (a) DSM地形
    ax = axes[0, 0]
    im = ax.imshow(dsm, cmap='terrain', origin='upper', vmin=0, vmax=45)
    for i, obs in enumerate(observers):
        ax.scatter(obs[1], obs[0], c='red', s=150, marker='o',
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(f'観測者{i+1}', (obs[1], obs[0]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=9, color='red')
    ax.scatter(start[1], start[0], c='lime', s=200, marker='s',
               edgecolors='black', linewidths=2, label='スタート', zorder=5)
    ax.scatter(goal[1], goal[0], c='blue', s=300, marker='*',
               edgecolors='black', linewidths=2, label='ゴール', zorder=5)
    ax.set_title('(a) DSM 地形', fontsize=13)
    ax.set_xlabel('列（x方向）')
    ax.set_ylabel('行（y方向）')
    ax.legend(loc='upper left', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='標高 (m)')

    # (b) 統合可視性スコア
    ax = axes[0, 1]
    vis_colors = np.zeros((*visibility_score.shape, 3))
    vis_colors[visibility_score == 0] = [0.2, 0, 0.4]      # 濃紫: 完全ステルス
    vis_colors[visibility_score == 1] = [0.9, 0.5, 0.1]    # オレンジ: 1人から見える
    vis_colors[visibility_score == 2] = [0.9, 0.1, 0.1]    # 赤: 2人から見える
    vis_colors[visibility_score >= 3] = [1, 1, 1]          # 白: 3人以上
    ax.imshow(vis_colors, origin='upper')
    for i, obs in enumerate(observers):
        ax.scatter(obs[1], obs[0], c='red', s=150, marker='o',
                   edgecolors='white', linewidths=1.5, zorder=5)
    ax.scatter(start[1], start[0], c='lime', s=200, marker='s',
               edgecolors='white', linewidths=2, zorder=5)
    ax.scatter(goal[1], goal[0], c='blue', s=300, marker='*',
               edgecolors='white', linewidths=2, zorder=5)
    ax.set_title('(b) 統合可視性スコア（紫=見えない / 暖色=見える）', fontsize=13)
    ax.set_xlabel('列（x方向）')
    ax.set_ylabel('行（y方向）')

    # (c) 完全ステルス経路
    ax = axes[1, 0]
    ax.imshow(dsm, cmap='gray', origin='upper', alpha=0.35, vmin=0, vmax=45)
    vis_masked = np.ma.masked_where(visibility_score == 0, visibility_score)
    ax.imshow(vis_masked, cmap='autumn', alpha=0.35, origin='upper', vmin=0, vmax=3)

    if path_stealth:
        path_arr = np.array(path_stealth)
        ax.plot(path_arr[:, 1], path_arr[:, 0], 'c-', linewidth=4,
                label='完全ステルス経路', zorder=4)
        ax.scatter(path_arr[:, 1], path_arr[:, 0], c='cyan', s=12, zorder=4)

    for i, obs in enumerate(observers):
        ax.scatter(obs[1], obs[0], c='red', s=150, marker='o',
                   edgecolors='black', linewidths=1.5, zorder=5)
    ax.scatter(start[1], start[0], c='lime', s=200, marker='s',
               edgecolors='black', linewidths=2, label='スタート', zorder=5)
    ax.scatter(goal[1], goal[0], c='blue', s=300, marker='*',
               edgecolors='black', linewidths=2, label='ゴール', zorder=5)
    ax.set_title('(c) 完全ステルス経路（紫領域のみを通る）', fontsize=13)
    ax.set_xlabel('列（x方向）')
    ax.set_ylabel('行（y方向）')
    ax.legend(loc='upper left', fontsize=9)

    # (d) リスク許容経路との比較
    ax = axes[1, 1]
    ax.imshow(dsm, cmap='gray', origin='upper', alpha=0.35, vmin=0, vmax=45)
    ax.imshow(vis_masked, cmap='autumn', alpha=0.35, origin='upper', vmin=0, vmax=3)

    if path_risky:
        path_arr2 = np.array(path_risky)
        ax.plot(path_arr2[:, 1], path_arr2[:, 0], 'm-', linewidth=4,
                label='リスク許容経路', zorder=4)
        ax.scatter(path_arr2[:, 1], path_arr2[:, 0], c='magenta', s=12, zorder=4)

    if path_stealth:
        path_arr = np.array(path_stealth)
        ax.plot(path_arr[:, 1], path_arr[:, 0], 'c--', linewidth=2,
                alpha=0.5, label='完全ステルス経路（参考）', zorder=3)

    for i, obs in enumerate(observers):
        ax.scatter(obs[1], obs[0], c='red', s=150, marker='o',
                   edgecolors='black', linewidths=1.5, zorder=5)
    ax.scatter(start[1], start[0], c='lime', s=200, marker='s',
               edgecolors='black', linewidths=2, label='スタート', zorder=5)
    ax.scatter(goal[1], goal[0], c='blue', s=300, marker='*',
               edgecolors='black', linewidths=2, label='ゴール', zorder=5)
    ax.set_title('(d) リスク許容経路との比較', fontsize=13)
    ax.set_xlabel('列（x方向）')
    ax.set_ylabel('行（y方向）')
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n可視化画像を保存しました: {save_path}")


# ============================================================
# 6. メイン処理（デモ実行）
# ============================================================

def main():
    """デモ実行: サンプルDSMでステルス経路を計算・可視化する。"""

    # DSM生成
    print("=== サンプルDSMを生成中 ===")
    dsm, cell_size = generate_sample_dsm(size=200, cell_size=5.0, seed=456)
    rows, cols = dsm.shape
    print(f"DSMサイズ: {rows} x {cols}, セルサイズ: {cell_size}m")

    # 観測者（東側の高台に3人）
    observers = [
        (50, 160),   # 北東
        (100, 170),  # 東中央
        (150, 160),  # 南東
    ]

    # スタート: 谷の北端（西側）
    start = (30, 30)
    # ゴール: 谷の南端（西側）
    goal = (170, 30)

    print(f"\n観測者: {observers}")
    print(f"スタート: {start}, ゴール: {goal}")

    # 統合可視性スコアの計算
    print("\n=== 統合可視性スコアを計算中 ===")
    visibility_score = compute_combined_visibility(
        dsm, observers, observer_height=1.6, cell_size=cell_size, max_range=300
    )
    print(f"可視性スコア: min={visibility_score.min()}, max={visibility_score.max()}")
    print(f"完全ステルスセル（0）: {(visibility_score == 0).sum()} / {visibility_score.size}")

    # ケース1: 完全ステルス
    print("\n=== ケース1: 完全ステルス経路（max_visibility=0） ===")
    path_stealth, info_stealth = astar_stealth_risk(
        dsm, visibility_score, start, goal,
        cell_size=cell_size, risk_weight=100.0, max_visibility=0
    )

    if path_stealth:
        print(f"  経路発見: {info_stealth['steps']} ステップ, "
              f"{info_stealth['total_distance_m']:.1f} m")
        print(f"  経路上の最大可視性: {info_stealth['max_visibility_on_path']}")
        print(f"  リスクセル比率: {info_stealth['risky_cells_ratio']*100:.1f}%")
    else:
        print("  完全ステルス経路は見つかりませんでした。")

    # ケース2: リスク許容
    print("\n=== ケース2: リスク許容経路（max_visibility=1） ===")
    path_risky, info_risky = astar_stealth_risk(
        dsm, visibility_score, start, goal,
        cell_size=cell_size, risk_weight=50.0, max_visibility=1
    )

    if path_risky:
        print(f"  経路発見: {info_risky['steps']} ステップ, "
              f"{info_risky['total_distance_m']:.1f} m")
        print(f"  経路上の最大可視性: {info_risky['max_visibility_on_path']}")
        print(f"  リスクセル比率: {info_risky['risky_cells_ratio']*100:.1f}%")
    else:
        print("  リスク許容経路は見つかりませんでした。")

    # 可視化
    print("\n=== 結果を可視化中 ===")
    visualize_result(dsm, observers, start, goal, visibility_score,
                     path_stealth, path_risky,
                     save_path='stealth_route_result.png')

    # 経路座標をJSONで保存（オプション）
    if path_stealth:
        route_data = {
            'start': start,
            'goal': goal,
            'observers': observers,
            'path_stealth': path_stealth,
            'info_stealth': info_stealth,
            'path_risky': path_risky,
            'info_risky': info_risky,
        }
        with open('stealth_route_data.json', 'w', encoding='utf-8') as f:
            json.dump(route_data, f, ensure_ascii=False, indent=2)
        print("経路データを保存しました: stealth_route_data.json")


if __name__ == "__main__":
    main()
```

---

### コードの使い方

**1. 実行**

```bash
python stealth_route_planner.py
```

**2. 実際のDSMデータを使う場合**

基盤地図情報の5mメッシュDEMやGeoTIFFを読み込んで `dsm` 配列に代入するだけで動作します。

```python
from osgeo import gdal
import numpy as np

# GeoTIFF読み込み
ds = gdal.Open('your_dsm.tif')
dsm = ds.ReadAsArray().astype(np.float32)

# 観測者・スタート・ゴールをグリッド座標で指定
observers = [(row1, col1), (row2, col2)]
start = (row_start, col_start)
goal = (row_goal, col_goal)

# 可視性スコア計算
visibility_score = compute_combined_visibility(
    dsm, observers, observer_height=1.6, cell_size=5.0
)

# ステルス経路探索
path, info = astar_stealth_risk(
    dsm, visibility_score, start, goal,
    max_visibility=0, risk_weight=100.0
)
```

**3. 主要パラメータの調整**

| パラメータ | 意味 | 推奨値 |
|-----------|------|--------|
| `max_visibility` | 許容可視性 | 0=完全ステルス, 1=1人まで許容 |
| `risk_weight` | 可視性1あたりの追加コスト | 大きいほどリスク回避重視 |
| `observer_height` | 観測者の眼の高さ | 1.2〜1.7m（立位・車載など） |
| `max_range` | 最大視程 | 300m程度で十分な場合が多い |
| `cell_size` | DSMの解像度 | 5.0m（基盤地図情報準拠） |

ファイル `stealth_route_planner.py` も配信済みですので、そちらから直接ダウンロードしてお使いいただけます。


