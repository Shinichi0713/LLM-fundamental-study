#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViewShed解析スクリプト - 点群（LAZ）直接判定版

機能:
1. 地形LAZデータおよびUAVウェイポイントLAZデータを読み込み
2. 点群（3D KD-Tree）を直接参照して視線遮蔽（Line of Sight）を判定
3. 地形点群およびウェイポイントに可視性に応じたRGBカラーを付与してLAS保存
   - 地形: 可視=緑 (0, 65535, 0) / 不可視=青 (0, 0, 32768)
   - ウェイポイント: 可視=赤 (65535, 0, 0) / 不可視=青 (0, 0, 65535)
"""

import numpy as np
import os
import sys

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    print("[警告] laspyがインストールされていません。")

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[エラー] scipyが必要です: pip install scipy")


# ============================================================
# 点群直接参照による ViewShed 解析 (3D Ray-Casting)
# ============================================================
def viewshed_pointcloud_direct(observer_x, observer_y, observer_z,
                               target_x, target_y, target_z,
                               kdtree, terrain_z,
                               observer_height=1.7,
                               search_radius=1.0,
                               sample_step=0.5):
    """
    点群（KD-Tree）を直接検索し、監視者から対象点への視線が遮られているか判定する

    Parameters:
    -----------
    observer_x, y, z : float
        監視者の座標
    target_x, y, z : ndarray
        判定対象（地形点群またはウェイポイント）のXY座標配列
    kdtree : scipy.spatial.cKDTree
        地形点群の2D (XY) KD-Tree
    terrain_z : ndarray
        地形点群のZ座標配列
    observer_height : float
        監視者の目の高さ (m)
    search_radius : float
        視線チェック時の近傍点探索半径 (m)
    sample_step : float
        視線上のサンプル間隔 (m)

    Returns:
    --------
    visibility : bool ndarray
        各対象点の可視性（True=可視）
    """
    obs_elev = observer_z + observer_height
    n_targets = len(target_x)
    visibility = np.ones(n_targets, dtype=bool)

    for i in range(n_targets):
        tx, ty, tz = target_x[i], target_y[i], target_z[i]

        dx = tx - observer_x
        dy = ty - observer_y
        dz = tz - obs_elev
        dist_2d = np.sqrt(dx**2 + dy**2)

        if dist_2d < 1e-3:
            continue

        # 視線上のサンプル数を計算
        n_steps = max(2, int(dist_2d / sample_step))
        ratios = np.linspace(0.05, 0.95, n_steps)  # 始点・終点付近を除く

        sample_x = observer_x + ratios * dx
        sample_y = observer_y + ratios * dy
        sample_z = obs_elev + ratios * dz

        sample_coords = np.column_stack((sample_x, sample_y))

        # 視線上の各点における近傍地形点を検索
        blocked = False
        for s_idx in range(len(sample_coords)):
            # 視線サンプル位置から一定半径内の地形点を検索
            neighbor_indices = kdtree.query_ball_point(sample_coords[s_idx], r=search_radius)

            if neighbor_indices:
                # 近傍地形点の最大標高を取得
                max_terrain_z = np.max(terrain_z[neighbor_indices])
                # 視線の高度より地形の標高が高ければ遮蔽とみなす
                if max_terrain_z > sample_z[s_idx] + 1e-3:
                    blocked = True
                    break

        visibility[i] = not blocked

    return visibility


def viewshed_multi_observers_pc(observers, terrain_x, terrain_y, terrain_z,
                                target_x, target_y, target_z,
                                observer_height=1.7, search_radius=1.0):
    """複数監視者からの可視性を判定 (OR条件)"""
    print("  [情報] 2D KD-Tree を構築中...")
    terrain_xy = np.column_stack((terrain_x, terrain_y))
    kdtree = cKDTree(terrain_xy)

    combined_vis = np.zeros(len(target_x), dtype=bool)

    for idx, obs in enumerate(observers):
        print(f"  [解析中] 監視者 {idx+1}/{len(observers)} ({obs['x']}, {obs['y']}, {obs['z']})...")
        vis = viewshed_pointcloud_direct(
            obs['x'], obs['y'], obs['z'],
            target_x, target_y, target_z,
            kdtree, terrain_z,
            observer_height=observer_height,
            search_radius=search_radius
        )
        combined_vis |= vis

    return combined_vis


# ============================================================
# LAS/LAZ ファイル処理
# ============================================================
def read_laz_points(filepath):
    """LAZ/LASファイルから点群データを読み込む"""
    if not HAS_LASPY:
        raise ImportError("laspyがインストールされていません。\npip install laspy[lazrs]")

    las = laspy.read(filepath)
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    return x, y, z, las.header


def save_colored_las(x, y, z, visibility, output_path, is_waypoint=False, header=None):
    """可視性に応じた色（RGB）を付与してLAS出力"""
    if not HAS_LASPY:
        return

    n_points = len(x)
    new_header = laspy.LasHeader(point_format=2, version="1.2")
    if header is not None:
        new_header.scales = header.scales
        new_header.offsets = header.offsets

    las = laspy.LasData(new_header)
    las.x = x
    las.y = y
    las.z = z

    if is_waypoint:
        # ウェイポイント: 可視=赤 (65535,0,0) / 不可視=青 (0,0,65535)
        red = np.where(visibility, 65535, 0).astype(np.uint16)
        green = np.zeros(n_points, dtype=np.uint16)
        blue = np.where(visibility, 0, 65535).astype(np.uint16)
    else:
        # 地形点群: 可視=緑 (0,65535,0) / 不可視=濃い青 (0,0,32768)
        red = np.zeros(n_points, dtype=np.uint16)
        green = np.where(visibility, 65535, 0).astype(np.uint16)
        blue = np.where(visibility, 0, 32768).astype(np.uint16)

    las.red = red
    las.green = green
    las.blue = blue

    las.write(output_path)
    print(f"  保存完了: {output_path} (可視: {visibility.sum()}/{n_points})")


# ============================================================
# メイン実行処理
# ============================================================
def main():
    TERRAIN_LAZ = "terrain.laz"
    WAYPOINT_LAZ = "waypoints.laz"

    OBSERVER_HEIGHT = 1.7
    SEARCH_RADIUS = 1.0  # 視線チェック時の近傍検索半径(m)

    OBSERVERS = [
        {'x': 20.0, 'y': 20.0, 'z': 12.0},
        {'x': 80.0, 'y': 80.0, 'z': 11.0},
        {'x': 50.0, 'y': 10.0, 'z': 13.0},
    ]

    OUTPUT_TERRAIN = "terrain_viewshed.las"
    OUTPUT_WAYPOINTS = "waypoints_viewshed.las"

    # 1. データの読み込み
    if os.path.exists(TERRAIN_LAZ):
        print(f"地形LAZデータの読み込み: {TERRAIN_LAZ}")
        tx, ty, tz, t_header = read_laz_points(TERRAIN_LAZ)
    else:
        print(f"[エラー] 地形LAZが見つかりません: {TERRAIN_LAZ}")
        sys.exit(1)

    if os.path.exists(WAYPOINT_LAZ):
        print(f"ウェイポイントLAZデータの読み込み: {WAYPOINT_LAZ}")
        wpx, wpy, wpz, wp_header = read_laz_points(WAYPOINT_LAZ)
    else:
        print(f"[エラー] ウェイポイントLAZが見つかりません: {WAYPOINT_LAZ}")
        sys.exit(1)

    # 2. 地形点群に対する ViewShed 解析
    print("\n--- 地形点群に対する直接 ViewShed 解析 ---")
    terrain_vis = viewshed_multi_observers_pc(
        OBSERVERS, tx, ty, tz, tx, ty, tz,
        observer_height=OBSERVER_HEIGHT, search_radius=SEARCH_RADIUS
    )

    # 3. UAVウェイポイントに対する ViewShed 解析
    print("\n--- UAVウェイポイントに対する ViewShed 解析 ---")
    wp_vis = viewshed_multi_observers_pc(
        OBSERVERS, tx, ty, tz, wpx, wpy, wpz,
        observer_height=OBSERVER_HEIGHT, search_radius=SEARCH_RADIUS
    )

    # 4. LAS出力
    print("\n--- 解析結果の出力 ---")
    save_colored_las(tx, ty, tz, terrain_vis, OUTPUT_TERRAIN, is_waypoint=False, header=t_header)
    save_colored_las(wpx, wpy, wpz, wp_vis, OUTPUT_WAYPOINTS, is_waypoint=True, header=wp_header)

    print("\n解析処理が完了しました。")


if __name__ == "__main__":
    main()