#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViewShed解析スクリプト
- LAZ/LAS点群ファイルから地形DEMを生成
- 複数の監視者からUAVウェイポイントへの可視性を判定
- 可視なウェイポイントを赤色、不可視を青色で着色
"""

import numpy as np
import sys

# ============================================================
# ライブラリのインポート
# ============================================================
try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    print("[警告] laspyがインストールされていません。サンプルデータで動作します。")

try:
    import scipy
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
    print("[警告] matplotlibがインストールされていません。")


# ============================================================
# DEM生成関数
# ============================================================
def create_dem_from_points(x, y, z, resolution=1.0):
    """
    点群データからDEM（標高グリッド）を生成する

    Parameters:
    -----------
    x, y, z : array-like
        点群の座標
    resolution : float
        DEMの解像度（グリッドサイズ）

    Returns:
    --------
    dem : 2D numpy array
        標高グリッド
    x_min, y_min : float
        グリッドの原点
    n_cols, n_rows : int
        グリッドのサイズ
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    n_cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    n_rows = int(np.ceil((y_max - y_min) / resolution)) + 1

    dem = np.full((n_rows, n_cols), np.nan)

    # 点をグリッドに割り当て（最大標高を採用）
    for xi, yi, zi in zip(x, y, z):
        col = int((xi - x_min) / resolution)
        row = int((yi - y_min) / resolution)
        col = min(col, n_cols - 1)
        row = min(row, n_rows - 1)
        if np.isnan(dem[row, col]) or zi > dem[row, col]:
            dem[row, col] = zi

    # NaNを補間
    dem = fill_nan_dem(dem)

    return dem, x_min, y_min, n_cols, n_rows


def fill_nan_dem(dem):
    """DEMのNaNを近傍値で補間する"""
    if not HAS_SCIPY:
        # scipyがない場合は前方補間
        mask = np.isnan(dem)
        dem_filled = dem.copy()
        while mask.any():
            dem_filled[mask] = np.roll(dem_filled, 1, axis=0)[mask]
            dem_filled[mask] = np.roll(dem_filled, 1, axis=1)[mask]
            mask = np.isnan(dem_filled)
        return dem_filled

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
# ViewShed解析関数
# ============================================================
def bresenham_line(x0, y0, x1, y1):
    """
    Bresenhamの線画アルゴリズムでグリッド上の線のセルを取得
    """
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


def viewshed_2d(observer_x, observer_y, observer_z, dem, x_min, y_min, resolution,
                waypoint_x, waypoint_y, waypoint_z, observer_height=1.7):
    """
    単一の監視者からUAVウェイポイントへの可視性を判定

    Parameters:
    -----------
    observer_x, observer_y, observer_z : float
        監視者の位置
    dem : 2D array
        標高グリッド
    x_min, y_min : float
        DEMの原点
    resolution : float
        DEMの解像度
    waypoint_x, waypoint_y, waypoint_z : array
        UAVウェイポイントの座標
    observer_height : float
        監視者の目の高さ（観測高さ）

    Returns:
    --------
    visible : bool array
        各ウェイポイントが可視かどうか
    """
    obs_elev = observer_z + observer_height

    visible = np.ones(len(waypoint_x), dtype=bool)
    n_rows, n_cols = dem.shape

    for i, (wx, wy, wz) in enumerate(zip(waypoint_x, waypoint_y, waypoint_z)):
        # グリッド座標に変換
        g0_col = int((observer_x - x_min) / resolution)
        g0_row = int((observer_y - y_min) / resolution)
        g1_col = int((wx - x_min) / resolution)
        g1_row = int((wy - y_min) / resolution)

        # グリッド範囲のチェック
        if not (0 <= g1_col < n_cols and 0 <= g1_row < n_rows):
            visible[i] = False
            continue

        # 線上のセルを取得
        line_cells = bresenham_line(g0_col, g0_row, g1_col, g1_row)

        if len(line_cells) < 2:
            continue

        # 始点と終点の水平距離
        dx = wx - observer_x
        dy = wy - observer_y
        total_dist = np.sqrt(dx**2 + dy**2)
        if total_dist == 0:
            continue

        # 視線の傾き
        d_elev = wz - obs_elev

        blocked = False
        for j, (col, row) in enumerate(line_cells[:-1]):  # 終点は除く
            if not (0 <= col < n_cols and 0 <= row < n_rows):
                continue

            # このセルの実世界座標
            cell_x = x_min + col * resolution + resolution / 2
            cell_y = y_min + row * resolution + resolution / 2

            # 監視者からこのセルまでの距離
            dist = np.sqrt((cell_x - observer_x)**2 + (cell_y - observer_y)**2)
            t = dist / total_dist
            sight_elev = obs_elev + d_elev * t

            # 地形の高さ
            terrain_elev = dem[row, col]

            # 地形が視線より高い場合は遮られる
            if terrain_elev > sight_elev + 1e-6:
                blocked = True
                break

        visible[i] = not blocked

    return visible


def viewshed_multi_observers(observers, dem, x_min, y_min, resolution,
                             waypoint_x, waypoint_y, waypoint_z,
                             observer_height=1.7):
    """
    複数の監視者からのViewShed解析
    いずれかの監視者から可視なら可視とする

    Parameters:
    -----------
    observers : list of dict
        [{'x': float, 'y': float, 'z': float}, ...]
    dem : 2D array
        標高グッド
    x_min, y_min : float
        DEMの原点
    resolution : float
        DEMの解像度
    waypoint_x, waypoint_y, waypoint_z : array
        UAVウェイポイントの座標
    observer_height : float
        監視者の観測高さ

    Returns:
    --------
    visible : bool array
        いずれかの監視者から可視ならTrue
    """
    combined_visible = np.zeros(len(waypoint_x), dtype=bool)

    for obs in observers:
        vis = viewshed_2d(obs['x'], obs['y'], obs['z'], dem, x_min, y_min, resolution,
                          waypoint_x, waypoint_y, waypoint_z, observer_height)
        combined_visible |= vis

    return combined_visible


# ============================================================
# LAZ/LASファイル読み込み
# ============================================================
def read_laz_points(filepath):
    """
    LAZ/LASファイルから点群データを読み込む

    Parameters:
    -----------
    filepath : str
        LAZ/LASファイルのパス

    Returns:
    --------
    x, y, z : ndarray
        点群の座標
    """
    if not HAS_LASPY:
        raise ImportError("laspyがインストールされていません。\npip install laspy[lazrs] でインストールしてください。")

    las = laspy.read(filepath)
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    return x, y, z


def read_waypoints_from_laz(filepath):
    """
    UAVウェイポイントを含むLAZファイルを読み込む
    点群の分類（classification）が9（Water）やユーザ定義の場合を想定

    Parameters:
    -----------
    filepath : str
        LAZ/LASファイルのパス

    Returns:
    --------
    wp_x, wp_y, wp_z : ndarray
        ウェイポイントの座標
    """
    if not HAS_LASPY:
        raise ImportError("laspyがインストールされていません。")

    las = laspy.read(filepath)
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    # classification属性がある場合はウェイポイントを抽出
    if hasattr(las, 'classification'):
        classification = np.array(las.classification)
        # 例: classification == 15 をウェイポイントとする
        # 実際のデータに合わせて変更してください
        wp_mask = classification == 15
        if wp_mask.any():
            return x[wp_mask], y[wp_mask], z[wp_mask]

    # classificationがない場合は全点を返す
    return x, y, z


# ============================================================
# 可視化
# ============================================================
def visualize_viewshed(dem, x_min, y_min, resolution, observers,
                     waypoint_x, waypoint_y, visible, output_path='viewshed_result.png'):
    """
    ViewShed解析結果を可視化して保存する

    Parameters:
    -----------
    dem : 2D array
        標高グリッド
    x_min, y_min : float
        DEMの原点
    resolution : float
        DEMの解像度
    observers : list of dict
        監視者の位置
    waypoint_x, waypoint_y : array
        UAVウェイポイントのXY座標
    visible : bool array
        可視性の判定結果
    output_path : str
        出力画像のパス
    """
    if not HAS_MATPLOTLIB:
        print("[警告] matplotlibがインストールされていません。可視化をスキップします。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 左図: 地形 + 監視者 + 全ウェイポイント ---
    ax1 = axes[0]
    extent = [x_min, x_min + dem.shape[1]*resolution, y_min, y_min + dem.shape[0]*resolution]
    im1 = ax1.imshow(dem, extent=extent, origin='lower', cmap='terrain', alpha=0.8)
    plt.colorbar(im1, ax=ax1, label='標高 (m)')

    for obs in observers:
        ax1.scatter(obs['x'], obs['y'], c='lime', marker='^', s=200,
                    edgecolors='black', linewidths=1.5, zorder=5)
    ax1.scatter(waypoint_x, waypoint_y, c='gray', s=30, alpha=0.6, label='UAV Waypoints')

    ax1.set_title('地形と監視者・UAVウェイポイント')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')

    # --- 右図: ViewShed解析結果 ---
    ax2 = axes[1]
    im2 = ax2.imshow(dem, extent=extent, origin='lower', cmap='terrain', alpha=0.5)

    for obs in observers:
        ax2.scatter(obs['x'], obs['y'], c='lime', marker='^', s=200,
                    edgecolors='black', linewidths=1.5, zorder=5)

    # 可視なウェイポイントを赤色で表示
    ax2.scatter(waypoint_x[visible], waypoint_y[visible],
                c='red', s=50, alpha=0.9, label='可視 (Visible)', zorder=4)
    # 不可視なウェイポイントを青色で表示
    ax2.scatter(waypoint_x[~visible], waypoint_y[~visible],
                c='blue', s=50, alpha=0.9, label='不可視 (Not Visible)', zorder=4)

    # 視線を描画
    for obs in observers:
        for i, (wx, wy) in enumerate(zip(waypoint_x, waypoint_y)):
            if visible[i]:
                ax2.plot([obs['x'], wx], [obs['y'], wy], 'r-', alpha=0.1, linewidth=0.5)

    n_visible = visible.sum()
    ax2.set_title(f'ViewShed解析結果 (可視: {n_visible}/{len(waypoint_x)})')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可視化画像を保存しました: {output_path}")


# ============================================================
# 出力LASファイル生成（着色済み）
# ============================================================
def save_colored_waypoints(waypoint_x, waypoint_y, waypoint_z, visible, output_path):
    """
    可視性に応じて色付けしたウェイポイントをLASフイルとして保存する

    Parameters:
    -----------
    waypoint_x, waypoint_y, waypoint_z : array
        UAVウェイポイントの座標
    visible : bool array
        可視性の判定結果
    output_path : str
        出力LASファイルのパス
    """
    if not HAS_LASPY:
        print("[警告] laspyがインストールされていません。出力をスキップします。")
        return

    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)

    las.x = waypoint_x
    las.y = waypoint_y
    las.z = waypoint_z

    # 赤色（可視）: RGB = (65535, 0, 0)
    # 青色（不可視）: RGB = (0, 0, 65535)
    n_points = len(waypoint_x)
    red = np.where(visible, 65535, 0).astype(np.uint16)
    green = np.zeros(n_points, dtype=np.uint16)
    blue = np.where(visible, 0, 65535).astype(np.uint16)

    las.red = red
    las.green = green
    las.blue = blue

    las.write(output_path)
    print(f"着色済みウェイポイントを保存しました: {output_path}")


# ============================================================
# メイン処理
# ============================================================
def main():
    # --------------------------------------------------------
    # パラメータ設定（実際のデータに合わせて変更してください）
    # --------------------------------------------------------

    # 地形点群のLAZファイルパス
    TERRAIN_LAZ = "terrain.laz"  # 実際のファイルパスに変更

    # UAVウェイポインのLAZファイルパス
    WAYPOINT_LAZ = "waypoints.laz"  # 実際のファイルパスに変更

    # DEMの解像度（m）
    RESOLUTION = 1.0

    # 監視者の位置（実際のデータに合わせて設定）
    # 例: [{'x': 20.0, 'y': 30.0, 'z': 12.5}, ...]
    OBSERVERS = [
        {'x': 20.0, 'y': 20.0, 'z': 12.0},
        {'x': 80.0, 'y': 80.0, 'z': 11.0},
    ]

    # 監視者の観測高さ（目の高さ）
    OBSERVER_HEIGHT = 1.7

    # 出力ファイルパス
    OUTPUT_IMAGE = "viewshed_result.png"
    OUTPUT_LAS = "colored_waypoints.las"

    # --------------------------------------------------------
    # データ読み込み
    # --------------------------------------------------------

    # 実際のLAZファイルが存在する場合は読み込む
    import os

    if os.path.exists(TERRAIN_LAZ) and HAS_LASPY:
        print(f"地形データを読み込み中: {TERRAIN_LAZ}")
        tx, ty, tz = read_laz_points(TERRAIN_LAZ)
    else:
        print("サンプル地形データを生成します。")
        tx, ty, tz = generate_sample_terrain()

    if os.path.exists(WAYPOINT_LAZ) and HAS_LASPY:
        print(f"ウェイポイントデータを読み込み中: {WAYPOINT_LAZ}")
        wp_x, wp_y, wp_z = read_waypoints_from_laz(WAYPOINT_LAZ)
    else:
        print("サンプルウェイポイントを生成します。")
        wp_x, wp_y, wp_z = generate_sample_waypoints()

    # --------------------------------------------------------
    # DEM生成
    # --------------------------------------------------------
    print("DEMを生成中...")
    dem, x_min, y_min, n_cols, n_rows = create_dem_from_points(tx, ty, tz, resolution=RESOLUTION)
    print(f"  DEMサイズ: {n_cols} x {n_rows}")
    print(f"  標高範囲: {dem.min():.2f} ~ {dem.max():.2f} m")

    # --------------------------------------------------------
    # ViewShed解析
    # --------------------------------------------------------
    print("ViewShed解析を実行中...")
    visible = viewshed_multi_observers(OBSERVERS, dem, x_min, y_min, RESOLUTION,
                                       wp_x, wp_y, wp_z, observer_height=OBSERVER_HEIGHT)

    n_visible = visible.sum()
    print(f"  可視なウェイポイント: {n_visible}/{len(wp_x)}")
    print(f"  可視率: {100*n_visible/len(wp_x):.1f}%")

    # --------------------------------------------------------
    # 可視化
    # --------------------------------------------------------
    visualize_viewshed(dem, x_min, y_min, RESOLUTION, OBSERVERS,
                       wp_x, wp_y, visible, output_path=OUTPUT_IMAGE)

    # --------------------------------------------------------
    # 出力
    # --------------------------------------------------------
    save_colored_waypoints(wp_x, wp_y, wp_z, visible, output_path=OUTPUT_LAS)

    print("\n完了しました。")


# ============================================================
# サンプルデータ生成（デモ用）
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

    # 丘を追加
    hills = [
        (30, 40, 12, 8),   # (x, y, height, radius)
        (70, 60, 15, 10),
        (50, 20, 10, 6),
    ]
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
    n_waypoints = 50
    wp_x = np.random.uniform(10, 90, n_waypoints)
    wp_y = np.random.uniform(10, 90, n_waypoints)
    wp_z = 18 + np.random.normal(0, 2, n_waypoints)  # 地上18m程度
    return wp_x, wp_y, wp_z


if __name__ == "__main__":
    main()
