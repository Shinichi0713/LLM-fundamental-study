import numpy as np
import laspy
from scipy.interpolate import griddata

def load_laz_points(file_path):
    """LAZファイルからXYZ座標を取得"""
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    return las, points

def create_dem(points, grid_size=0.5):
    """
    点群データから指定グリッドサイズの2D DEM（標高マップ）を生成
    """
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    grid_x, grid_y = np.mgrid[
        x_min:x_max:grid_size,
        y_min:y_max:grid_size
    ]

    # 最近傍補間または線形補間でDEMを構築
    dem = griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method='nearest')

    return dem, (x_min, x_max, y_min, y_max), grid_size

def is_visible(observer_pos, target_pos, dem, extent, grid_size, observer_offset=1.5, sample_num=100):
    """
    1つの視点（observer）から特定のウェイポイント（target）が見えるか判断（Line of Sight）
    observer_pos: [x, y, z] (監視者の足元標高)
    observer_offset: 監視者の目の高さ (m)
    """
    x_min, x_max, y_min, y_max = extent

    # 監視者の視点位置（目の高さ分Zを上げる）
    obs_x, obs_y, obs_z = observer_pos[0], observer_pos[1], observer_pos[2] + observer_offset
    tgt_x, tgt_y, tgt_z = target_pos[0], target_pos[1], target_pos[2]

    # 視線上のサンプル点を生成
    sample_ratios = np.linspace(0.01, 0.99, sample_num)
    sample_x = obs_x + sample_ratios * (tgt_x - obs_x)
    sample_y = obs_y + sample_ratios * (tgt_y - obs_y)
    sample_z = obs_z + sample_ratios * (tgt_z - obs_z)

    # グリッド座標に変換
    grid_i = ((sample_x - x_min) / grid_size).astype(int)
    grid_j = ((sample_y - y_min) / grid_size).astype(int)

    # グリッドのインデックス範囲制限
    grid_i = np.clip(grid_i, 0, dem.shape[0] - 1)
    grid_j = np.clip(grid_j, 0, dem.shape[1] - 1)

    # 地形の標高を取得
    terrain_z = dem[grid_i, grid_j]

    # 視線の高度が地形の標高より低い場所（遮蔽）があれば不可視
    if np.any(sample_z < terrain_z):
        return False
    return True

def run_viewshed_analysis(laz_path, observers_pos, waypoints_pos, output_path="colored_waypoints.las"):
    """
    Viewshed解析を実行し、着色したウェイポイントLASを出力
    """
    # 1. LAZデータの読み込み
    las, points = load_laz_points(laz_path)
    print(f"Loaded {len(points)} points from {laz_path}")

    # 2. 地形DEMの生成 (グリッド解像度: 0.5m)
    grid_size = 0.5
    dem, extent, grid_size = create_dem(points, grid_size=grid_size)

    # 3. 各ウェイポイントの可視性判定
    # 赤: 可視 (RGB: 255, 0, 0) / 青: 不可視 (RGB: 0, 0, 255)
    colors = []
    visibility_results = []

    for idx, wp in enumerate(waypoints_pos):
        visible_from_any = False
        for obs in observers_pos:
            if is_visible(obs, wp, dem, extent, grid_size, observer_offset=1.7):
                visible_from_any = True
                break

        visibility_results.append(visible_from_any)

        # LASファイルのRGB値 (16-bitカラー: 0-65535表現に変換)
        if visible_from_any:
            colors.append([65535, 0, 0])      # 赤色 (255 -> 65535)
        else:
            colors.append([0, 0, 65535])      # 青色 (255 -> 65535)

    colors = np.array(colors, dtype=np.uint16)

    # 4. 結果をLASファイルとして書き出し
    header = laspy.LasHeader(point_format=3, version="1.2")
    out_las = laspy.LasData(header)

    waypoints_pos = np.array(waypoints_pos)
    out_las.x = waypoints_pos[:, 0]
    out_las.y = waypoints_pos[:, 1]
    out_las.z = waypoints_pos[:, 2]
    out_las.red = colors[:, 0]
    out_las.green = colors[:, 1]
    out_las.blue = colors[:, 2]

    out_las.write(output_path)
    print(f"Saved viewshed result to {output_path}")

    # 集計結果の表示
    visible_count = sum(visibility_results)
    print(f"解析結果: 全 {len(waypoints_pos)} ポイント中、{visible_count} 個が可視（赤）、{len(waypoints_pos) - visible_count} 個が不可視（青）です。")

# --------------------------------------------------
# サンプル実行用スクリプト
# --------------------------------------------------
if __name__ == "__main__":
    # 入力ファイル名
    laz_file = "terrain_data.laz"

    # 監視者の座標 [X, Y, Z]（複数指定可能）
    observers = [
        [100.0, 150.0, 10.5],
        [200.0, 250.0, 12.0]
    ]

    # UAVウェイポイントの座標リスト [X, Y, Z]
    waypoints = [
        [105.0, 155.0, 30.0],
        [150.0, 200.0, 25.0],
        [180.0, 220.0, 15.0],
        [300.0, 300.0, 40.0]
    ]

    # 解析の実行
    # run_viewshed_analysis(laz_file, observers, waypoints, output_path="colored_waypoints.las")