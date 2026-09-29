#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
laz_to_3dtiles.py

LAZ/LAS点群ファイル（または合成点群）を3D Tiles（PNTS形式）に変換するPythonコード。

機能:
  - LAZ/LASファイルの読み込み（laspyが必要）
  - laspyがない場合は合成点群データで動作確認可能
  - 点群をPNTS（Point Cloud Tile）バイナリ形式で書き出し
  - tileset.json（3D Tilesのエントリファイル）を生成
  - 座標の16bit量化による圧縮

使い方:
  1. LAZファイルを使う場合: pip install laspy[lazrs]
  2. python laz_to_3dtiles.py
  3. 出力された tileset.json をCesium等のビューアで読み込む

依存ライブラリ:
  - numpy
  - laspy[lazrs] （実際のLAZ読み込み時のみ）
  - json, struct, os, math （標準ライブラリ）
"""

import numpy as np
import json
import struct
import os
import math


# ============================================================
# 1. LAZ/LAS読み込み（laspyがあれば実ファイル、なければ合成点群）
# ============================================================

def read_laz_or_synthetic(filepath=None, num_points=100000, seed=42):
    """
    LAZファイルを読み込む。laspyがなければ合成点群を返す。

    Parameters
    ----------
    filepath : str or None
        入力LAZ/LASファイルパス。Noneの場合は合成点群を生成。
    num_points : int
        合成点群の点数（filepath=None時）
    seed : int
        乱数シード（filepath=None時）

    Returns
    -------
    points : ndarray (N, 3), float64
        点群のXYZ座標（地心座標系 EPSG:4978 を想定）
    colors : ndarray (N, 3), uint8
        RGBカラー（0-255）
    """
    try:
        import laspy
        print(f"LAZファイルを読み込み中: {filepath}")
        las = laspy.read(filepath)

        # XYZ座標（LASのスケール・オフセットを適用）
        x = las.x * las.header.scales[0] + las.header.offsets[0]
        y = las.y * las.header.scales[1] + las.header.offsets[1]
        z = las.z * las.header.scales[2] + las.header.offsets[2]
        points = np.column_stack([x, y, z])

        # カラーがあれば取得
        if hasattr(las, 'red'):
            r = np.array(las.red, dtype=np.uint8)
            g = np.array(las.green, dtype=np.uint8)
            b = np.array(las.blue, dtype=np.uint8)
            # 16bitカラーの場合は8bitに変換
            if las.red.max() > 255:
                r = (r / 256).astype(np.uint8)
                g = (g / 256).astype(np.uint8)
                b = (b / 256).astype(np.uint8)
            colors = np.column_stack([r, g, b])
        else:
            colors = np.tile(
                np.array([200, 200, 200], dtype=np.uint8), (len(points), 1)
            )

        print(f"  読み込み完了: {len(points)} 点")
        return points, colors

    except ImportError:
        print("laspy が未インストールのため、合成点群データを生成します。")
        print("実際のLAZを使う場合: pip install laspy[lazrs]")
        return generate_synthetic_pointcloud(num_points=num_points, seed=seed)


def generate_synthetic_pointcloud(num_points=100000, seed=42):
    """
    デモ用の合成点群データを生成する。
    地形（丘と谷）＋建物建物＋樹木の構造を含む。
    座標系: ローカル座標（メートル）

    Parameters
    ----------
    num_points : int
        生成する総点数
    seed : int
        乱数シード

    Returns
    -------
    points : ndarray (N, 3), float64
    colors : ndarray (N, 3), uint8
    """
    np.random.seed(seed)

    points_list = []
    colors_list = []

    # 1. 地面（起伏のある地形）
    n_ground = num_points // 2
    xg = np.random.uniform(-500, 500, n_ground)
    yg = np.random.uniform(-500, 500, n_ground)
    # 丘の地形
    zg = (30 * np.exp(-(xg**2 + yg**2) / 20000)
          + 10 * np.sin(xg / 50) * np.cos(yg / 50)
          + np.random.normal(0, 1, n_ground))

    ground_pts = np.column_stack([xg, yg, zg])
    ground_col = np.tile(
        np.array([139, 119, 101], dtype=np.uint8), (n_ground, 1)
    )  # 茶色系
    points_list.append(ground_pts)
    colors_list.append(ground_col)

    # 2. 建物（箱型の構造）
    n_buildings = 5
    for i in range(n_buildings):
        cx = np.random.uniform(-300, 300)
        cy = np.random.uniform(-300, 300)
        bw = np.random.uniform(20, 50)   # 幅
        bd = np.random.uniform(20, 50)   # 奥行き
        bh = np.random.uniform(30, 80)   # 高さ

        n_bp = num_points // (2 * n_buildings)
        xb = np.random.uniform(cx - bw / 2, cx + bw / 2, n_bp)
        yb = np.random.uniform(cy - bd / 2, cy + bd / 2, n_bp)
        zb = np.random.uniform(0, bh, n_bp)

        # 地面高を加算
        base_h = 30 * np.exp(-(cx**2 + cy**2) / 20000)
        zb += base_h

        building_pts = np.column_stack([xb, yb, zb])
        # 建物ごとに異なる色
        bcolor = np.array([
            np.random.randint(150, 220),
            np.random.randint(150, 220),
            np.random.randint(150, 220)
        ], dtype=np.uint8)
        building_col = np.tile(bcolor, (n_bp, 1))

        points_list.append(building_pts)
        colors_list.append(building_col)

    # 3. 樹木（円柱＋球状の冠）
    n_trees = 20
    for i in range(n_trees):
        cx = np.random.uniform(-400, 400)
        cy = np.random.uniform(-400, 400)
        th = np.random.uniform(8, 15)    # 幹高
        tr = np.random.uniform(3, 6)   # 冠半径

        n_tp = num_points // (4 * n_trees)
        # 幹
        xt = np.random.uniform(cx - 0.5, cx + 0.5, n_tp // 3)
        yt = np.random.uniform(cy - 0.5, cy + 0.5, n_tp // 3)
        base_h = 30 * np.exp(-(cx**2 + cy**2) / 20000)
        zt = np.random.uniform(base_h, base_h + th, n_tp // 3)

        # 冠（球）
        phi = np.random.uniform(0, 2 * np.pi, 2 * n_tp // 3)
        costheta = np.random.uniform(-1, 1, 2 * n_tp // 3)
        u = np.random.uniform(0, 1, 2 * n_tp // 3)
        theta = np.arccos(costheta)
        r = tr * np.cbrt(u)

        xc = cx + r * np.sin(theta) * np.cos(phi)
        yc = cy + r * np.sin(theta) * np.sin(phi)
        zc = base_h + th + r * np.cos(theta)

        tree_pts = np.column_stack([
            np.concatenate([xt, xc]),
            np.concatenate([yt, yc]),
            np.concatenate([zt, zc])
        ])
        tree_col = np.tile(
            np.array([34, 139, 34], dtype=np.uint8), (len(tree_pts), 1)
        )  # 緑

        points_list.append(tree_pts)
        colors_list.append(tree_col)

    points = np.vstack(points_list)
    colors = np.vstack(colors_list)

    # シャッフル
    perm = np.random.permutation(len(points))
    points = points[perm]
    colors = colors[perm]

    print(f"  合成点群生成完了: {len(points)} 点")
    return points, colors


# ============================================================
# 2. PNTSバイナリファイル生成
# ============================================================

def compute_bounding_box(points):
    """
    点群のバウンディングボックスを計算する。

    Parameters
    ----------
    points : ndarray (N, 3)

    Returns
    -------
    min_xyz : ndarray (3,)
    max_xyz : ndarray (3,)
    center : ndarray (3,)
    """
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    center = (min_xyz + max_xyz) / 2
    return min_xyz, max_xyz, center


def quantize_positions(points, min_xyz, max_xyz):
    """
    XYZ座標を16bit符号なし整数（0-65535）に量子化する。
    3D Tiles PNTSの標準的な圧縮手法。

    Parameters
    ----------
    points : ndarray (N, 3), float64
    min_xyz : ndarray (3,)
    max_xyz : ndarray (3,)

    Returns
    -------
    quantized : ndarray (N, 3), uint16
    """
    ranges = max_xyz - min_xyz
    ranges[ranges == 0] = 1.0  # ゼロ除算防止
    quantized = ((points - min_xyz) / ranges * 65535.0).astype(np.uint16)
    return quantized


def write_pnts_tile(points, colors, output_path):
    """
    点群データを PNTS (Point Cloud Tile) バイナリファイルとして書き出す。

    PNTSフォーマット (3D Tiles 1.0):
      Header (28 bytes)
      Feature Table JSON
      Feature Table Binary
      Batch Table JSON (optional)
      Batch Table Binary (optional)

    Parameters
    ----------
    points : ndarray (N, 3), float64
        XYZ座標（地心座標系 EPSG:4978 を想定、メートル単位）
    colors : ndarray (N, 3), uint8
        RGBカラー（0-255）
    output_path : str
        出力ファイルパス（.pnts）

    Returns
    -------
    min_xyz : ndarray (3,)
    max_xyz : ndarray (3,)
    center : ndarray (3,)
    """
    n_points = len(points)
    min_xyz, max_xyz, center = compute_bounding_box(points)

    # 座標を16bit量子化
    qpos = quantize_positions(points, min_xyz, max_xyz)

    # Feature Table JSON
    feature_table = {
        "POINTS_LENGTH": n_points,
        "QUANTIZED_VOLUME_OFFSET": {
            "0": float(min_xyz[0]),
            "1": float(min_xyz[1]),
            "2": float(min_xyz[2])
        },
        "QUANTIZED_VOLUME_SCALE": {
            "0": float(max_xyz[0] - min_xyz[0]),
            "1": float(max_xyz[1] - min_xyz[1]),
            "2": float(max_xyz[2] - min_xyz[2])
        },
        "POSITION_QUANTIZED": {
            "byteOffset": 0
        },
        "RGB": {
            "byteOffset": n_points * 3 * 2  # quantized positionの後
        }
    }

    feature_table_json_str = json.dumps(feature_table, separators=(',', ':'))
    # 4バイトアラインメント
    ft_json_padding = (4 - len(feature_table_json_str) % 4) % 4
    feature_table_json_str += ' ' * ft_json_padding
    feature_table_json_bytes = feature_table_json_str.encode('utf-8')

    # Feature Table Binary
    # POSITION_QUANTIZED: uint16 x 3 x N（各点6バイト）
    pos_binary = qpos.astype(np.uint16).tobytes()
    # RGB: uint8 x 3 x N（各点3バイト）
    rgb_binary = colors.astype(np.uint8).tobytes()

    # 4バイトアラインメント
    pos_padding = (4 - len(pos_binary) % 4) % 4
    pos_binary += b'\x00' * pos_padding

    feature_table_binary = pos_binary + rgb_binary
    ft_binary_padding = (4 - len(feature_table_binary) % 4) % 4
    feature_table_binary += b'\x00' * ft_binary_padding

    # Batch Table（空）
    batch_table_json_bytes = b'{}'
    bt_json_padding = (4 - len(batch_table_json_bytes) % 4) % 4
    batch_table_json_bytes += b' ' * bt_json_padding
    batch_table_binary = b''

    # Header
    magic = b'pnts'
    version = 1
    byte_length = (28
                   + len(feature_table_json_bytes)
                   + len(feature_table_binary)
                   + len(batch_table_json_bytes)
                   + len(batch_table_binary))

    header = struct.pack('<4sIIIIII',
                         magic,
                         version,
                         byte_length,
                         len(feature_table_json_bytes),
                         len(feature_table_binary),
                         len(batch_table_json_bytes),
                         len(batch_table_binary))

    # 書き出し
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(feature_table_json_bytes)
        f.write(feature_table_binary)
        f.write(batch_table_json_bytes)
        f.write(batch_table_binary)

    print(f"  PNTSファイルを書き出しました: {output_path}")
    print(f"    点数: {n_points}")
    print(f"    バウンディングボックス: min={min_xyz}, max={max_xyz}")
    print(f"    ファイルサイズ: {byte_length} bytes ({byte_length / 1024 / 1024:.2f} MB)")

    return min_xyz, max_xyz, center


# ============================================================
# 3. tileset.json 生成
# ============================================================

def write_tileset_json(min_xyz, max_xyz, pnts_filename, output_path,
                       geometric_error=100.0):
    """
    3D Tilesの tileset.json を書き出す。

    Parameters
    ----------
    min_xyz, max_xyz : ndarray (3,)
        点群のバウンディングボックス
    pnts_filename : str
        PNTSファイル名（tileset.jsonから相対パス）
    output_path : str
        tileset.jsonの出力パス
    geometric_error : float
        このタイルの幾何学的誤差（メートル）
    """
    # region（WGS84の経度緯度高さの範囲）に変換する簡易的な処理
    # 実際には EPSG:4978 の地心座標 → WGS84 の変換が必要
    # ここではローカル座標のまま region を仮設定する
    # region = [west, south, east, north, min_height, max_height]
    # 単位: ラジアン（west/south/east/north）、メートル（min/max_height）

    # 仮のWGS84座標（デモ用）- 実際には座標変換が必要
    lon_min, lat_min = 139.6917 - 0.01, 35.6895 - 0.01
    lon_max, lat_max = 139.6917 + 0.01, 35.6895 + 0.01

    tileset = {
        "asset": {
            "version": "1.0"
        },
        "geometricError": geometric_error,
        "root": {
            "refine": "REPLACE",
            "geometricError": geometric_error,
            "boundingVolume": {
                "region": [
                    math.radians(lon_min),
                    math.radians(lat_min),
                    math.radians(lon_max),
                    math.radians(lat_max),
                    float(min_xyz[2]),
                    float(max_xyz[2])
                ]
            },
            "content": {
                "uri": pnts_filename
            }
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tileset, f, indent=2, ensure_ascii=False)

    print(f"  tileset.json を書き出しました: {output_path}")


# ============================================================
# 4. メイン処理
# ============================================================

def laz_to_3dtiles(input_laz_path=None, output_dir="3dtiles_output",
                   num_points=100000, geometric_error=100.0):
    """
    LAZファイル（または合成点群）を3D Tiles（PNTS + tileset.json）に変換する。

    Parameters
    ----------
    input_laz_path : str or None
        入力LAZ/LASファイルパス。Noneの場合は合成点群を使用。
    output_dir : str
        出力ディレクトリ
    num_points : int
        合成点群の点数（LAZ未指定時）
    geometric_error : float
        タイルの幾何学的誤差（メートル）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 点群読み込み
    print("\n=== ステップ1: 点群読み込み ===")
    points, colors = read_laz_or_synthetic(input_laz_path, num_points=num_points)

    # 2. PNTS書き出し
    print("\n=== ステップ2: PNTSタイル生成 ===")
    pnts_path = os.path.join(output_dir, "points.pnts")
    min_xyz, max_xyz, center = write_pnts_tile(points, colors, pnts_path)

    # 3. tileset.json書き出し
    print("\n=== ステップ3: tileset.json生成 ===")
    tileset_path = os.path.join(output_dir, "tileset.json")
    write_tileset_json(min_xyz, max_xyz, "points.pnts", tileset_path, geometric_error)

    print("\n=== 変換完了 ===")
    print(f"出力ディレクトリ: {os.path.abspath(output_dir)}")
    print(f"  - {pnts_path}")
    print(f"  - {tileset_path}")
    print("\nCesiumで表示するには:")
    print("  Cesium.Viewerに tileset.json のURLを指定してください。")
    print("\n例（CesiumJS）:")
    print("""
  const viewer = new Cesium.Viewer('cesiumContainer');
  const tileset = viewer.scene.primitives.add(new Cesium.Cesium3DTileset({
      url: './3dtiles_output/tileset.json'
  }));
    """)


if __name__ == "__main__":
    # コマンドライン引数でLAZファイルパスを指定可能
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = None  # 合成点群を使用

    laz_to_3dtiles(
        input_laz_path=input_path,
        output_dir="3dtiles_output",
        num_points=50000,
        geometric_error=50.0
    )
