#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViewShed解析スクリプト - LAS地形データから直接解析

入力:
  - 地形データ: LAS/LAZファイル（点群）
  - UAVウェイポイント: LAS/LAZファイル（点群）
  - 監視者位置: スクリプト内で定義

処理:
  1. LAS地形点群を読み込み
  2. 点群からDEM（標高グリッド）を生成
  3. 複数監視者からのViewShed解析
  4. 元の地形点群に可視性を反映したLASを出力
  5. UAVウェイポイントの可視性付きLASを出力

出力:
  - terrain_viewshed.las : 地形点群（緑=可視, 青=不可視）
  - waypoints_viewshed.las : UAVウェイポイント（赤=可, 青=不可視）
  - viewshed_3d_result.png : 可視化画像
"""

import numpy as np
import os

# ============================================================
# ライブラリ
# ============================================================
try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    print("[ERROR] laspyが必要です: pip install laspy[lazrs]")
    raise

try:
    from scipy.ndimage import generic_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# 設定（ここを実際のデータに変更してください）
# ============================================================
# 入力: 地形データのLAS/LAZファイル
TERRAIN_LAS = "terrain.las"       # 地形点群のLASファイルパス

# 入力: UAVウェイポイントのLAS/LAZファイル
WAYPOINT_LAS = "waypoints.las"    # UAVウェイポイントのLASフイルパス

# 監視者の位置リスト
OBSERVERS = [
    {'x': 20.0, 'y': 20.0, 'z': 12.0},   # 監視者1
    {'x': 80.0, 'y': 80.0, 'z': 11.0},   # 監視者2
    # 必要に応じて追加
]

# DEMの解像度（メートル）
RESOLUTION = 1.0

# 監視者の目の高さ（地上からの高さ、メートル）
OBSERVER_HEIGHT = 1.7

# 出力ファイル名
OUTPUT_TERRAIN = "terrain_viewshed.las"
OUTPUT_WAYPOINTS = "waypoints_viewshed.las"
OUTPUT_IMAGE = "viewshed_3d_result.png"


# ============================================================
# 1. LASファイル読み込み
# ============================================================
def read_las(filepath):
    """LAS/LAZファイルから点群座標を読み込む"""
    print(f"読み込み中: {filepath}")
    las = laspy.read(filepath)
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    print(f"  点数: {len(x)}")
    return x, y, z, las.header


# ============================================================
# 2. DEM生成（LAS点群 → 標高グリッ）
# ============================================================
def create_dem(px, py, pz, resolution):
    """
    LAS点群からDEM（標高グリッド）を生成する

    各グリッドセルに複数点が入る場合、最大標高を採用
    空のセルは近傍補間で埋める
    """
    x_min, x_max = px.min(), px.max()
    y_min, y_max = py.min(), py.max()

    n_cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    n_rows = int(np.ceil((y_max - y_min) / resolution)) + 1

    dem = np.full((n_rows, n_cols), np.nan)

    for xi, yi, zi in zip(px, py, pz):
        col = int((xi - x_min) / resolution)
        row = int((yi - y_min) / resolution)
        col = min(col, n_cols - 1)
        row = min(row, n_rows - 1)
        if np.isnan(dem[row, col]) or zi > dem[row, col]:
            dem[row, col] = zi

    # NaN補間
    if HAS_SCIPY:
        def nanmean(w):
            v = w[~np.isnan(w)]
            return v.mean() if len(v) > 0 else np.nan
        for _ in range(100):
            if not np.isnan(dem).any():
                break
            dem = generic_filter(dem, nanmean, size=3, mode='nearest')
    else:
        for _ in range(100):
            if not np.isnan(dem).any():
                break
            m = np.isnan(dem)
            dem[m] = np.roll(dem, 1, axis=0)[m]
            dem[m] = np.roll(dem, 1, axis=1)[m]

    return dem, x_min, y_min, n_cols, n_rows


# ============================================================
# 3. ViewShed解析
# ============================================================
def bresenham(x0, y0, x1, y1):
    """Bresenham線描画アルゴリズム"""
    cells = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def viewshed_raster(dem, x_min, y_min, resolution, observer, obs_height=1.7):
    """
    単一監視者からDEM上の全セルの可視性を判定

    Returns:
        2D bool array: True=可視, False=不可視
    """
    ox, oy, oz = observer['x'], observer['y'], observer['z']
    obs_elev = oz + obs_height
    n_rows, n_cols = dem.shape
    vis = np.zeros((n_rows, n_cols), dtype=bool)

    g0c = int((ox - x_min) / resolution)
    g0r = int((oy - y_min) / resolution)

    if 0 <= g0c < n_cols and 0 <= g0r < n_rows:
        vis[g0r, g0c] = True

    for row in range(n_rows):
        for col in range(n_cols):
            if row == g0r and col == g0c:
                continue

            wx = x_min + col * resolution + resolution / 2
            wy = y_min + row * resolution + resolution / 2
            wz = dem[row, col]

            cells = bresenham(g0c, g0r, col, row)
            if len(cells) < 2:
                vis[row, col] = True
                continue

            dx, dy = wx - ox, wy - oy
            total_dist = np.sqrt(dx**2 + dy**2)
            if total_dist == 0:
                vis[row, col] = True
                continue

            d_elev = wz - obs_elev
            blocked = False

            for cc, cr in cells[:-1]:
                if not (0 <= cc < n_cols and 0 <= cr < n_rows):
                    continue
                cx = x_min + cc * resolution + resolution / 2
                cy = y_min + cr * resolution + resolution / 2
                dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                t = dist / total_dist
                sight_elev = obs_elev + d_elev * t
                if dem[cr, cc] > sight_elev + 1e-6:
                    blocked = True
                    break

            vis[row, col] = not blocked

    return vis


def viewshed_raster_multi(dem, x_min, y_min, resolution, observers, obs_height=1.7):
    """複数監視者からDEM上の全セルの可視性を判定（OR条件）"""
    combined = np.zeros(dem.shape, dtype=bool)
    for obs in observers:
        vis = viewshed_raster(dem, x_min, y_min, resolution, obs, obs_height)
        combined |= vis
    return combined


# ============================================================
# 4. 点群への可視性マッピング
# ============================================================
def map_vis_to_points(px, py, vis_raster, x_min, y_min, resolution):
    """点群の各点をDEMセルに対応付け、可視性を取得"""
    cols = ((px - x_min) / resolution).astype(int)
    rows = ((py - y_min) / resolution).astype(int)
    n_rows, n_cols = vis_raster.shape
    cols = np.clip(cols, 0, n_cols - 1)
    rows = np.clip(rows, 0, n_rows - 1)
    return vis_raster[rows, cols]


def viewshed_waypoints(dem, x_min, y_min, resolution, observers, wp_x, wp_y, wp_z, obs_height=1.7):
    """複数監視者からUAVウェイポイントへの可視性を判定"""
    combined = np.zeros(len(wp_x), dtype=bool)
    n_rows, n_cols = dem.shape

    for obs in observers:
        ox, oy, oz = obs['x'], obs['y'], obs['z']
        obs_elev = oz + obs_height
        vis = np.ones(len(wp_x), dtype=bool)

        for i, (wx, wy, wz) in enumerate(zip(wp_x, wp_y, wp_z)):
            g0c = int((ox - x_min) / resolution)
            g0r = int((oy - y_min) / resolution)
            g1c = int((wx - x_min) / resolution)
            g1r = int((wy - y_min) / resolution)

            if not (0 <= g1c < n_cols and 0 <= g1r < n_rows):
                vis[i] = False
                continue

            cells = bresenham(g0c, g0r, g1c, g1r)
            if len(cells) < 2:
                continue

            dx, dy = wx - ox, wy - oy
            total_dist = np.sqrt(dx**2 + dy**2)
            if total_dist == 0:
                continue

            d_elev = wz - obs_elev
            blocked = False

            for cc, cr in cells[:-1]:
                if not (0 <= cc < n_cols and 0 <= cr < n_rows):
                    continue
                cx = x_min + cc * resolution + resolution / 2
                cy = y_min + cr * resolution + resolution / 2
                dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                t = dist / total_dist
                sight_elev = obs_elev + d_elev * t
                if dem[cr, cc] > sight_elev + 1e-6:
                    blocked = True
                    break

            vis[i] = not blocked

        combined |= vis

    return combined


# ============================================================
# 5. LAS出力
# ============================================================
def save_terrain_las(px, py, pz, visibility, header, filepath):
    """
    地形点群に可視性を反映した色を付与してLAS出力

    可視: 緑 (0, 65535, 0)
    不可視: 青 (0, 0, 32768)
    """
    new_header = laspy.LasHeader(point_format=2, version="1.2")
    new_header.scales = header.scales
    new_header.offsets = header.offsets

    las = laspy.LasData(new_header)
    las.x = px
    las.y = py
    las.z = pz

    n = len(px)
    las.red   = np.where(visibility, 0,      0).astype(np.uint16)
    las.green = np.where(visibility, 65535,  0).astype(np.uint16)
    las.blue  = np.where(visibility, 0,  32768).astype(np.uint16)

    las.write(filepath)
    print(f"  保存完了: {filepath} (可視={visibility.sum()}/{n})")


def save_waypoints_las(px, py, pz, visibility, header, filepath):
    """
    UAVウェイポイントを可視性に応じて赤/青で着色してLAS出力

    可視: 赤 (65535, 0, 0)
    不可視: 青 (0, 0, 65535)
    """
    new_header = laspy.LasHeader(point_format=2, version="1.2")
    new_header.scales = header.scales
    new_header.offsets = header.offsets

    las = laspy.LasData(new_header)
    las.x = px
    las.y = py
    las.z = pz

    n = len(px)
    las.red   = np.where(visibility, 65535,  0).astype(np.uint16)
    las.green = np.where(visibility, 0,      0).astype(np.uint16)
    las.blue  = np.where(visibility, 0,  65535).astype(np.uint16)

    las.write(filepath)
    print(f"  保存完了: {filepath} (可視={visibility.sum()}/{n})")


# ============================================================
# 6. 可視化
# ============================================================
def visualize(dem, x_min, y_min, resolution, observers,
              tx, ty, t_vis, wx, wy, w_vis, output_path):
    """解析結果を4パネルで可視化"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    extent = [x_min, x_min + dem.shape[1]*resolution,
              y_min, y_min + dem.shape[0]*resolution]

    # DEM + 監視者
    ax = axes[0, 0]
    im = ax.imshow(dem, extent=extent, origin='lower', cmap='terrain', alpha=0.9)
    for obs in observers:
        ax.scatter(obs['x'], obs['y'], c='lime', marker='^', s=200,
                   edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_title('地形DEMと監視者位置')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='標高 (m)')

    # ViewShed Raster
    ax = axes[0, 1]
    vis_raster = viewshed_raster_multi(dem, x_min, y_min, resolution, observers, OBSERVER_HEIGHT)
    colors = np.zeros((*vis_raster.shape, 3))
    colors[vis_raster] = [0, 1, 0]
    colors[~vis_raster] = [0, 0, 0.3]
    ax.imshow(colors, extent=extent, origin='lower', alpha=0.9)
    for obs in observers:
        ax.scatter(obs['x'], obs['y'], c='red', marker='^', s=200,
                   edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_title(f'ViewShed Raster (可視セル: {vis_raster.sum()}/{vis_raster.size})')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')

    # 地形点群の可視性
    ax = axes[1, 0]
    ax.scatter(tx[t_vis], ty[t_vis], c='lime', s=1, alpha=0.6, label='可視')
    ax.scatter(tx[~t_vis], ty[~t_vis], c='navy', s=1, alpha=0.6, label='不可視')
    for obs in observers:
        ax.scatter(obs['x'], obs['y'], c='red', marker='^', s=200,
                   edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(f'地形点群の可視性 (可視={t_vis.sum()}/{len(tx)})')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal'); ax.legend(markerscale=5)

    # UAVウェイポイントの可視性
    ax = axes[1, 1]
    ax.imshow(dem, extent=extent, origin='lower', cmap='terrain', alpha=0.4)
    ax.scatter(wx[w_vis], wy[w_vis], c='red', s=50, alpha=0.9, label='可視', zorder=4)
    ax.scatter(wx[~w_vis], wy[~w_vis], c='blue', s=50, alpha=0.9, label='不可視', zorder=4)
    for obs in observers:
        ax.scatter(obs['x'], obs['y'], c='lime', marker='^', s=200,
                   edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_title(f'UAVウェイポイントの可視性 (可視={w_vis.sum()}/{len(wx)})')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal'); ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  保存完了: {output_path}")


# ============================================================
# サンプルデータ生成（LASファイルがない場合の動作確認用）
# ============================================================
def generate_sample_terrain():
    """サンプル地形点群を生成"""
    np.random.seed(42)
    n = 5000
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = 10 + 2*np.sin(x/15)*np.cos(y/15) + np.random.normal(0, 0.3, n)
    for hx, hy, hh, hr in [(30, 40, 12, 8), (70, 60, 15, 10), (50, 20, 10, 6)]:
        nh = 500
        th = np.random.uniform(0, 2*np.pi, nh)
        r = np.random.uniform(0, hr, nh)
        x = np.concatenate([x, hx + r*np.cos(th)])
        y = np.concatenate([y, hy + r*np.sin(th)])
        z = np.concatenate([z, 10 + hh*np.exp(-(r**2)/(2*(hr/2)**2))])
    return x, y, z


def generate_sample_waypoints():
    """サンプルUAVウェイポイントを生成"""
    np.random.seed(123)
    n = 50
    return (np.random.uniform(10, 90, n),
            np.random.uniform(10, 90, n),
            18 + np.random.normal(0, 2, n))


# ============================================================
# メイン処理
# ============================================================
def main():
    print("=" * 60)
    print("ViewShed解析 - LAS地形データ版")
    print("=" * 60)

    # --- 1. データ読み込み ---
    print("\n[1/5] LASデータ読み込み")
    if os.path.exists(TERRAIN_LAS):
        tx, ty, tz, header = read_las(TERRAIN_LAS)
    else:
        print(f"  {TERRAIN_LAS} が見つかりません。サンプル地形を使用します。")
        tx, ty, tz = generate_sample_terrain()
        header = laspy.LasHeader(point_format=2, version="1.2")

    if os.path.exists(WAYPOINT_LAS):
        wx, wy, wz, wp_header = read_las(WAYPOINT_LAS)
    else:
        print(f"  {WAYPOINT_LAS} が見つかりません。サンプルウェイポイントを使用します。")
        wx, wy, wz = generate_sample_waypoints()
        wp_header = header

    # --- 2. DEM生成 ---
    print("\n[2/5] DEM生成（LAS点群 → 標高グリッド）")
    dem, x_min, y_min, n_cols, n_rows = create_dem(tx, ty, tz, RESOLUTION)
    print(f"  DEMサイズ: {n_cols} x {n_rows}")
    print(f"  標高範囲: {dem.min():.2f} ~ {dem.max():.2f} m")

    # --- 3. ViewShed解析 ---
    print("\n[3/5] ViewShed解析")
    vis_raster = viewshed_raster_multi(dem, x_min, y_min, RESOLUTION, OBSERVERS, OBSERVER_HEIGHT)
    print(f"  可視セル数: {vis_raster.sum()}/{vis_raster.size}")

    # 地形点群への可視性マッピング
    t_vis = map_vis_to_points(tx, ty, vis_raster, x_min, y_min, RESOLUTION)
    print(f"  可視な地形点: {t_vis.sum()}/{len(tx)}")

    # UAVウェイポイントへの可視性判定
    w_vis = viewshed_waypoints(dem, x_min, y_min, RESOLUTION, OBSERVERS, wx, wy, wz, OBSERVER_HEIGHT)
    print(f"  可視なウェイポイント: {w_vis.sum()}/{len(wx)}")

    # --- 4. LAS出力 ---
    print("\n[4/5] LASファイル出力")
    save_terrain_las(tx, ty, tz, t_vis, header, OUTPUT_TERRAIN)
    save_waypoints_las(wx, wy, wz, w_vis, wp_header, OUTPUT_WAYPOINTS)

    # --- 5. 可視化 ---
    print("\n[5/5] 可視化")
    visualize(dem, x_min, y_min, RESOLUTION, OBSERVERS,
              tx, ty, t_vis, wx, wy, w_vis, OUTPUT_IMAGE)

    print("\n" + "=" * 60)
    print("完了しました")
    print("=" * 60)
    print(f"出力ファイル:")
    print(f"  地形点群（可視性付き）: {OUTPUT_TERRAIN}")
    print(f"  UAVウェイポイント（可視性付き）: {OUTPUT_WAYPOINTS}")
    print(f"  可視化画像: {OUTPUT_IMAGE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
