#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViewShed解析スクリプト - 元の地形点群に可視性を反映

機能:
1. LAZ/LAS点群ファイル（地形）を読み込み
2. DEMを生成
3. 複数の監視者からのViewShed解析を実行
4. 元の地形点群の各点に可視性に応じた色を付与してLAS出力
5. UAVウェイポイントも可視性に応じて赤/青で出力
"""

import numpy as np
import sys
import os

# ============================================================
# ライブラリのインポート
# ============================================================
try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    print("[警告] laspyがインストールされていません。")

try:
    from scipy.ndimage import generic_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[警告] scipyがインストールされていません。")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# DEM生成
# ============================================================
def create_dem_from_points(x, y, z, resolution=1.0):
    """点群データからDEM（標高グリッド）を生成する"""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    n_cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    n_rows = int(np.ceil((y_max - y_min) / resolution)) + 1

    dem = np.full((n_rows, n_cols), np.nan)

    for xi, yi, zi in zip(x, y, z):
        col = int((xi - x_min) / resolution)
        row = int((yi - y_min) / resolution)
        col = min(col, n_cols - 1)
        row = min(row, n_rows - 1)
        if np.isnan(dem[row, col]) or zi > dem[row, col]:
            dem[row, col] = zi

    dem = fill_nan_dem(dem)
    return dem, x_min, y_min, n_cols, n_rows


def fill_nan_dem(dem):
    """DEMのNaNを近傍値で補間する"""
    if not HAS_SCIPY:
        filled = dem.copy()
        for _ in range(100):
            if not np.isnan(filled).any():
                break
            mask = np.isnan(filled)
            filled[mask] = np.roll(filled, 1, axis=0)[mask]
            filled[mask] = np.roll(filled, 1, axis=1)[mask]
        return filled

    def nanmean_window(window):
        valid = window[~np.isnan(window)]
        return valid.mean() if len(valid) > 0 else np.nan

    filled = dem.copy()
    for _ in range(100):
        if not np.isnan(filled).any():
            break
        filled = generic_filter(filled, nanmean_window, size=3, mode='nearest')
    return filled


# ============================================================
# ViewShed解析
# ============================================================
def bresenham_line(x0, y0, x1, y1):
    """Bresenhamの線描画アルゴリズム"""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
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


def viewshed_raster_single_observer(observer_x, observer_y, observer_z, dem,
                                     x_min, y_min, resolution, observer_height=1.7):
    """
    単一監視者からDEM上の各セルの可視性を判定する

    Returns:
    --------
    visible_raster : 2D bool array
        各セルの可視性（True=可視）
    """
    obs_elev = observer_z + observer_height
    n_rows, n_cols = dem.shape
    visible_raster = np.zeros((n_rows, n_cols), dtype=bool)

    g0_col = int((observer_x - x_min) / resolution)
    g0_row = int((observer_y - y_min) / resolution)

    if 0 <= g0_col < n_cols and 0 <= g0_row < n_rows:
        visible_raster[g0_row, g0_col] = True

    for row in range(n_rows):
        for col in range(n_cols):
            if row == g0_row and col == g0_col:
                continue

            wx = x_min + col * resolution + resolution / 2
            wy = y_min + row * resolution + resolution / 2
            wz = dem[row, col]

            line_cells = bresenham_line(g0_col, g0_row, col, row)
            if len(line_cells) < 2:
                visible_raster[row, col] = True
                continue

            dx = wx - observer_x
            dy = wy - observer_y
            total_dist = np.sqrt(dx**2 + dy**2)
            if total_dist == 0:
                visible_raster[row, col] = True
                continue

            d_elev = wz - obs_elev
            blocked = False

            for j, (cc, cr) in enumerate(line_cells[:-1]):
                if not (0 <= cc < n_cols and 0 <= cr < n_rows):
                    continue

                cell_x = x_min + cc * resolution + resolution / 2
                cell_y = y_min + cr * resolution + resolution / 2
                dist = np.sqrt((cell_x - observer_x)**2 + (cell_y - observer_y)**2)
                t = dist / total_dist
                sight_elev = obs_elev + d_elev * t
                terrain_elev = dem[cr, cc]

                if terrain_elev > sight_elev + 1e-6:
                    blocked = True
                    break

            visible_raster[row, col] = not blocked

    return visible_raster


def viewshed_raster_multi_observers(observers, dem, x_min, y_min, resolution, observer_height=1.7):
    """複数監視者からDEM上の各セルの可視性を判定（OR条件）"""
    combined = np.zeros(dem.shape, dtype=bool)
    for obs in observers:
        vis = viewshed_raster_single_observer(
            obs['x'], obs['y'], obs['z'], dem, x_min, y_min, resolution, observer_height
        )
        combined |= vis
    return combined


# ============================================================
# 点群への可視性マッピング
# ============================================================
def map_visibility_to_points(px, py, visibility_raster, x_min, y_min, resolution):
    """
    点群の各点をDEMセルにマッピングし、可視性を取得する

    Parameters:
    -----------
    px, py : ndarray
        点群のXY座標
    visibility_raster : 2D bool array
        DEMの各セルの可視性
    x_min, y_min : float
        DEMの原点
    resolution : float
        DEMの解像度

    Returns:
    --------
    point_visibility : bool ndarray
        各点の可視性（True=可視）
    """
    cols = ((px - x_min) / resolution).astype(int)
    rows = ((py - y_min) / resolution).astype(int)

    n_rows, n_cols = visibility_raster.shape
    cols = np.clip(cols, 0, n_cols - 1)
    rows = np.clip(rows, 0, n_rows - 1)

    return visibility_raster[rows, cols]


# ============================================================
# LASファイル入出力
# ============================================================
def read_laz_points(filepath):
    """LAZ/LASファイルから点群データを読み込む"""
    if not HAS_LASPY:
        raise ImportError("laspyがインストールされていません。\npip install laspy[lazrs]")

    las = laspy.read(filepath)
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    # 追加属性があれば取得
    extras = {}
    if hasattr(las, 'classification'):
        extras['classification'] = np.array(las.classification)
    if hasattr(las, 'intensity'):
        extras['intensity'] = np.array(las.intensity)
    if hasattr(las, 'return_number'):
        extras['return_number'] = np.array(las.return_number)
    if hasattr(las, 'number_of_returns'):
        extras['number_of_returns'] = np.array(las.number_of_returns)
    if hasattr(las, 'scan_direction_flag'):
        extras['scan_direction_flag'] = np.array(las.scan_direction_flag)
    if hasattr(las, 'edge_of_flight_line'):
        extras['edge_of_flight_line'] = np.array(las.edge_of_flight_line)

    return x, y, z, extras, las.header


def save_colored_terrain_points(px, py, pz, visibility, output_path, header=None):
    """
    元の地形点群に可視性を反映した色を付与してLASファイルを保存する

    可視な点: 緑色 (0, 65535, 0) - 監視者から見えている地形
    不可視な点: 濃い青 (0, 0, 32768) - 遮られている地形

    Parameters:
    -----------
    px, py, pz : ndarray
        点群の座標
    visibility : bool ndarray
        各点の可視性
    output_path : str
        出力LASファイルのパス
    header : laspy.LasHeader, optional
        元のLASファイルのヘッダー（スケール・オフセットを維持）
    """
    if not HAS_LASPY:
        print("[警告] laspyがインストールされていません。出力をスキップします。")
        return

    n_points = len(px)

    # Point Format 2 を使用（RGB対応）
    if header is not None:
        new_header = laspy.LasHeader(point_format=2, version="1.2")
        new_header.scales = header.scales
        new_header.offsets = header.offsets
    else:
        new_header = laspy.LasHeader(point_format=2, version="1.2")

    las = laspy.LasData(new_header)
    las.x = px
    las.y = py
    las.z = pz

    # 可視な点: 緑色 (0, 65535, 0)
    # 不可視な点: 濃い青 (0, 0, 32768)
    red = np.where(visibility, 0, 0).astype(np.uint16)
    green = np.where(visibility, 65535, 0).astype(np.uint16)
    blue = np.where(visibility, 0, 32768).astype(np.uint16)

    las.red = red
    las.green = green
    las.blue = blue

    las.write(output_path)
    print(f"  地形点群（可視性付き）を保存しました: {output_path}")
    print(f"    可視点: {visibility.sum()}/{n_points}")


def save_colored_waypoints(wp_x, wp_y, wp_z, visibility, output_path, header=None):
    """
    UAVウェイポイントを可視性に応じて赤/青で着色してLAS出力

    可視なウェイポイント: 赤色 (65535, 0, 0)
    不可視なウェイポイント: 青色 (0, 0, 65535)
    """
    if not HAS_LASPY:
        print("[警告] laspyがインストールされていません。出力をスキップします。")
        return

    n_points = len(wp_x)

    if header is not None:
        new_header = laspy.LasHeader(point_format=2, version="1.2")
        new_header.scales = header.scales
        new_header.offsets = header.offsets
    else:
        new_header = laspy.LasHeader(point_format=2, version="1.2")

    las = laspy.LasData(new_header)
    las.x = wp_x
    las.y = wp_y
    las.z = wp_z

    red = np.where(visibility, 65535, 0).astype(np.uint16)
    green = np.zeros(n_points, dtype=np.uint16)
    blue = np.where(visibility, 0, 65535).astype(np.uint16)

    las.red = red
    las.green = green
    las.blue = blue

    las.write(output_path)
    print(f"  ウェイポイント（可視性付き）を保存しました: {output_path}")
    print(f"    可視: {visibility.sum()}/{n_points}")


# ============================================================
# 可視化
# ============================================================
def visualize_results(dem, x_min, y_min, resolution, observers,
                      terrain_x, terrain_y, terrain_vis,
                      wp_x, wp_y, wp_vis,
                      output_path='viewshed_3d_result.png'):
    """解析結果を可視化する"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    extent = [x_min, x_min + dem.shape[1]*resolution,
              y_min, y_min + dem.shape[0]*resolution]

    # --- 左上: DEM + 監視者 ---
    ax = axes[0, 0]
    im = ax.imshow(dem, extent=extent, origin='lower', cmap='terrain', alpha=0.9)
    for obs in observers:
        ax.scatter(obs['x'], obs['y'], c='lime', marker='^', s=200,
                   edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_title('地形DEMと監視者位置')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='標高 (m)')

    # --- 右上: ViewShed Raster ---
    ax = axes[0, 1]
    # DEMベースのViewShed Rasterを生成
    vis_raster = viewshed_raster_multi_observers(observers, dem, x_min, y_min, resolution)
    vis_colors = np.zeros((*vis_raster.shape, 3))
    vis_colors[vis_raster] = [0, 1, 0]       # 可視: 緑
    vis_colors[~vis_raster] = [0, 0, 0.3]    # 不可視: 濃い青
    ax.imshow(vis_colors, extent=extent, origin='lower', alpha=0.9)
    for obs in observers:
        ax.scatter(obs['x'], obs['y'], c='red', marker='^', s=200,
                   edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_title(f'ViewShed Raster (可視セル: {vis_raster.sum()}/{vis_raster.size})')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')

    # --- 左下: 地形点群に可視性を反映 ---
    ax = axes[1, 0]
    ax.scatter(terrain_x[terrain_vis], terrain_y[terrain_vis], c='lime', s=1, alpha=0.6, label='可視')
    ax.scatter(terrain_x[~terrain_vis], terrain_y[~terrain_vis], c='navy', s=1, alpha=0.6, label='不可視')
    for obs in observers:
        ax.scatter(obs['x'], obs['y'], c='red', marker='^', s=200,
                   edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(f'地形点群の可視性 (可視: {terrain_vis.sum()}/{len(terrain_x)})')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.legend(markerscale=5)

    # --- 右下: UAVウェイポイントの可視性 ---
    ax = axes[1, 1]
    ax.imshow(dem, extent=extent, origin='lower', cmap='terrain', alpha=0.4)
    ax.scatter(wp_x[wp_vis], wp_y[wp_vis], c='red', s=50, alpha=0.9, label='可視', zorder=4)
    ax.scatter(wp_x[~wp_vis], wp_y[~wp_vis], c='blue', s=50, alpha=0.9, label='不可視', zorder=4)
    for obs in observers:
        ax.scatter(obs['x'], obs['y'], c='lime', marker='^', s=200,
                   edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_title(f'UAVウェイポイントの可視性 (可視: {wp_vis.sum()}/{len(wp_x)})')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可視化画像を保存しました: {output_path}")


# ============================================================
# サンプルデータ生成
# ============================================================
def generate_sample_terrain():
    """サンプル地形データを生成する"""
    np.random.seed(42)
    x_min, y_min = 0, 0
    x_max, y_max = 100, 100
    n_points = 5000

    x = np.random.uniform(x_min, x_max, n_points)
    y = np.random.uniform(y_min, y_max, n_points)
    z = 10 + 2 * np.sin(x / 15) * np.cos(y / 15) + np.random.normal(0, 0.3, n_points)

    hills = [(30, 40, 12, 8), (70, 60, 15, 10), (50, 20, 10, 6)]
    for hx, hy, hheight, hradius in hills:
        n_h = 500
        theta = np.random.uniform(0, 2*np.pi, n_h)
        r = np.random.uniform(0, hradius, n_h)
        hill_x = hx + r * np.cos(theta)
        hill_y = hy + r * np.sin(theta)
        hill_z = 10 + hheight * np.exp(-(r**2)/(2*(hradius/2)**2))
        x = np.concatenate([x, hill_x])
        y = np.concatenate([y, hill_y])
        z = np.concatenate([z, hill_z])

    return x, y, z


def generate_sample_waypoints():
    """サンプルUAVウェイポイントを生成する"""
    np.random.seed(123)
    n = 50
    wp_x = np.random.uniform(10, 90, n)
    wp_y = np.random.uniform(10, 90, n)
    wp_z = 18 + np.random.normal(0, 2, n)
    return wp_x, wp_y, wp_z


# ============================================================
# メイン処理
# ============================================================
def main():
    # --------------------------------------------------------
    # パラメータ設定
    # --------------------------------------------------------
    TERRAIN_LAZ = "terrain.laz"      # 地形点群のLAZファイル
    WAYPOINT_LAZ = "waypoints.laz"   # UAVウェイポイントのLAZファイル
    RESOLUTION = 1.0                 # DEM解像度（m）
    OBSERVER_HEIGHT = 1.7              # 監視者の観測高さ

    OBSERVERS = [
        {'x': 20.0, 'y': 20.0, 'z': 12.0},
        {'x': 80.0, 'y': 80.0, 'z': 11.0},
        {'x': 50.0, 'y': 10.0, 'z': 13.0},
    ]

    OUTPUT_TERRAIN = "terrain_viewshed.las"      # 可視性付き地形点群
    OUTPUT_WAYPOINTS = "waypoints_viewshed.las"   # 可視性付きウェイポイント
    OUTPUT_IMAGE = "viewshed_3d_result.png"

    # --------------------------------------------------------
    # データ読み込み
    # --------------------------------------------------------
    header = None
    if os.path.exists(TERRAIN_LAZ) and HAS_LASPY:
        print(f"地形データを読み込み中: {TERRAIN_LAZ}")
        tx, ty, tz, extras, header = read_laz_points(TERRAIN_LAZ)
    else:
        print("サンプル地形データを生成します。")
        tx, ty, tz = generate_sample_terrain()

    if os.path.exists(WAYPOINT_LAZ) and HAS_LASPY:
        print(f"ウェイポイントデータを読み込み中: {WAYPOINT_LAZ}")
        wpx, wpy, wpz, wpextras, wpheader = read_laz_points(WAYPOINT_LAZ)
    else:
        print("サンプルウェイポイントを生成します。")
        wpx, wpy, wpz = generate_sample_waypoints()

    # --------------------------------------------------------
    # DEM生成
    # --------------------------------------------------------
    print("\nDEMを生成中...")
    dem, x_min, y_min, n_cols, n_rows = create_dem_from_points(tx, ty, tz, resolution=RESOLUTION)
    print(f"  DEMサイズ: {n_cols} x {n_rows}")
    print(f"  標高範囲: {dem.min():.2f} ~ {dem.max():.2f} m")

    # --------------------------------------------------------
    # ViewShed解析
    # --------------------------------------------------------
    print("\nViewShed解析を実行中...")

    # DEM上のViewShed Rasterを生成
    vis_raster = viewshed_raster_multi_observers(OBSERVERS, dem, x_min, y_min, RESOLUTION, OBSERVER_HEIGHT)
    print(f"  可視セル数: {vis_raster.sum()}/{vis_raster.size}")

    # 地形点群に可視性をマッピング
    terrain_vis = map_visibility_to_points(tx, ty, vis_raster, x_min, y_min, RESOLUTION)
    print(f"  可視な地形点: {terrain_vis.sum()}/{len(tx)}")

    # UAVウェイポイントの可視性を判定（ウェイポイント間の視線追跡）
    wp_vis = viewshed_multi_observers(OBSERVERS, dem, x_min, y_min, RESOLUTION, wpx, wpy, wpz, OBSERVER_HEIGHT)
    print(f"  可視なウェイポイント: {wp_vis.sum()}/{len(wpx)}")

    # --------------------------------------------------------
    # 出力
    # --------------------------------------------------------
    print("\n出力ファイルを生成中...")
    save_colored_terrain_points(tx, ty, tz, terrain_vis, OUTPUT_TERRAIN, header)
    save_colored_waypoints(wpx, wpy, wpz, wp_vis, OUTPUT_WAYPOINTS, header)

    # --------------------------------------------------------
    # 可視化
    # --------------------------------------------------------
    print("\n可視化を生成中...")
    visualize_results(dem, x_min, y_min, RESOLUTION, OBSERVERS,
                      tx, ty, terrain_vis, wpx, wpy, wp_vis, OUTPUT_IMAGE)

    print("\n完了しました。")
    print(f"  出力ファイル:")
    print(f"    - {OUTPUT_TERRAIN}   (地形点群: 緑=可視, 青=不可視)")
    print(f"    - {OUTPUT_WAYPOINTS} (ウェイポイント: 赤=可視, 青=不可視)")
    print(f"    - {OUTPUT_IMAGE} (可視化画像)")


# viewshed_multi_observers 関数（ウェイポイント用）
def viewshed_multi_observers(observers, dem, x_min, y_min, resolution,
                             waypoint_x, waypoint_y, waypoint_z,
                             observer_height=1.7):
    """複数の監視者からUAVウェイポイントへの可視性を判定"""
    combined_visible = np.zeros(len(waypoint_x), dtype=bool)
    for obs in observers:
        vis = viewshed_2d(obs['x'], obs['y'], obs['z'], dem, x_min, y_min, resolution,
                          waypoint_x, waypoint_y, waypoint_z, observer_height)
        combined_visible |= vis
    return combined_visible


def viewshed_2d(observer_x, observer_y, observer_z, dem, x_min, y_min, resolution,
                waypoint_x, waypoint_y, waypoint_z, observer_height=1.7):
    """単一の監視者からUAVウェイポイントへの可視性を判定"""
    obs_elev = observer_z + observer_height
    visible = np.ones(len(waypoint_x), dtype=bool)
    n_rows, n_cols = dem.shape

    for i, (wx, wy, wz) in enumerate(zip(waypoint_x, waypoint_y, waypoint_z)):
        g0_col = int((observer_x - x_min) / resolution)
        g0_row = int((observer_y - y_min) / resolution)
        g1_col = int((wx - x_min) / resolution)
        g1_row = int((wy - y_min) / resolution)

        if not (0 <= g1_col < n_cols and 0 <= g1_row < n_rows):
            visible[i] = False
            continue

        line_cells = bresenham_line(g0_col, g0_row, g1_col, g1_row)
        if len(line_cells) < 2:
            continue

        dx = wx - observer_x
        dy = wy - observer_y
        total_dist = np.sqrt(dx**2 + dy**2)
        if total_dist == 0:
            continue

        d_elev = wz - obs_elev
        blocked = False

        for j, (col, row) in enumerate(line_cells[:-1]):
            if not (0 <= col < n_cols and 0 <= row < n_rows):
                continue

            cell_x = x_min + col * resolution + resolution / 2
            cell_y = y_min + row * resolution + resolution / 2
            dist = np.sqrt((cell_x - observer_x)**2 + (cell_y - observer_y)**2)
            t = dist / total_dist
            sight_elev = obs_elev + d_elev * t
            terrain_elev = dem[row, col]

            if terrain_elev > sight_elev + 1e-6:
                blocked = True
                break

        visible[i] = not blocked

    return visible


if __name__ == "__main__":
    main()
