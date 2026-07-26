import os
import torch
import numpy as np
import plotly.graph_objects as go

# 1. 生成されたファイルのパスを指定
ply_file_path = '/content/DROID-SLAM/scene.ply'

if not os.path.exists(ply_file_path):
    print(f"エラー: {ply_file_path} が見つかりません。")
else:
    print("DROID-SLAMのシリアライズデータをロード中...")
    
    # torch.loadを使って、擬似PLYファイルの中身（辞書データ）を読み込む
    checkpoint = torch.load(ply_file_path, map_location='cpu')
    
    # 辞書のキーを確認しつつ、データをNumPyに変換
    # DROID-SLAMの内部形式: poses, tstamp, images, disps, intrinsics など
    disps = checkpoint['disps'].numpy()        # 視差マップ (N, H, W)
    images = checkpoint['images'].numpy()      # 画像 (N, 3, H, W)
    intrinsics = checkpoint['intrinsics'].numpy()  # カメラ内パラメータ (N, 4) [fx, fy, cx, cy]

    print(f"データロード完了: フレーム数 = {len(images)}")

    # 2. 視差（Disparity）から深度（Depth）への変換
    # DROID-SLAMの仕様に従い、Depth = 1.0 / Disparity
    depths = 1.0 / (disps + 1e-5)
    
    _, _, h, w = images.shape
    fx, fy, cx, cy = intrinsics[0]

    all_points = []
    all_colors = []

    print("各フレームの深度マップから3D座標を計算しています...")
    # ブラウザのハングを防ぐため、10フレーム間隔（stride=10）でサンプリング
    stride = 10 
    
    for i in range(0, len(images), stride):
        img = images[i].transpose(1, 2, 0) # (3, H, W) -> (H, W, 3)
        img = img.astype(np.uint8)
        depth = depths[i]
        
        # グリッド座標の作成
        yy, xx = np.mgrid[0:h, 0:w]
        
        # 信頼できる有効な深度値のマスク（0.1m〜8.0m）
        mask = (depth > 0.1) & (depth < 8.0)
        
        # 点数が多すぎるとブラウザがフリーズするため、さらに縦横ピクセルを間引く
        mask = mask & (xx % 6 == 0) & (yy % 6 == 0)
        
        # 3D空間への逆投影計算（カメラ座標系）
        z = depth[mask]
        x = (xx[mask] - cx) * z / fx
        y = (yy[mask] - cy) * z / fy
        
        pts = np.stack([x, y, z], axis=-1)
        cols = img[mask]
        
        all_points.append(pts)
        all_colors.append(cols)

    # 全フレームの点・色を結合
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    
    print(f"描画する総点数: {len(points)}")

    # 3. Plotlyによる3Dインタラクティブ可視化
    rgb_strings = [f'rgb({r},{g},{b})' for r, g, b in colors]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=1.2, color=rgb_strings, opacity=0.7)
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False), 
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0), width=800, height=600
    )
    
    print("可視化に成功しました！画面をドラッグして回転・拡大できます。")
    fig.show()

import torch
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation


# --------------------------
# load
# --------------------------

scene = torch.load("/content/DROID-SLAM/reconstruction.pth", map_location="cpu")

images = scene["images"].numpy()
disps = scene["disps"].numpy()
poses = scene["poses"].numpy()
intrinsics = scene["intrinsics"].numpy()

N = images.shape[0]

print("Frames:", N)


# --------------------------
# point cloud
# --------------------------

pcd = o3d.geometry.PointCloud()

all_xyz = []
all_rgb = []


for i in range(N):

    img = images[i].transpose(1,2,0)

    disp = disps[i]

    fx,fy,cx,cy = intrinsics[i]

    h,w = disp.shape

    # disparity -> depth
    depth = np.zeros_like(disp)

    mask = disp > 1e-6

    depth[mask] = 1.0 / disp[mask]

    # RGBをDepth解像度へ縮小
    img_small = cv2.resize(
        img,
        (w,h),
        interpolation=cv2.INTER_LINEAR
    )

    u,v = np.meshgrid(np.arange(w),np.arange(h))

    X = (u-cx)*depth/fx
    Y = (v-cy)*depth/fy
    Z = depth

    xyz = np.stack([X,Y,Z],axis=-1).reshape(-1,3)

    rgb = img_small.reshape(-1,3)/255.

    valid = np.isfinite(xyz).all(axis=1)
    valid &= (xyz[:,2]>0)
    valid &= (xyz[:,2]<20)

    xyz = xyz[valid]
    rgb = rgb[valid]

    # pose
    t = poses[i,:3]

    q = poses[i,3:]

    R = Rotation.from_quat(q).as_matrix()

    xyz = xyz @ R.T + t

    all_xyz.append(xyz)
    all_rgb.append(rgb)



points = np.concatenate(all_xyz,axis=0)
colors = np.concatenate(all_rgb,axis=0)

print(points.shape)


pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)


print("Voxel Downsample...")

pcd = pcd.voxel_down_sample(0.02)


print("Remove Outlier...")

pcd,_ = pcd.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)

print("Estimate Normal...")

pcd.estimate_normals()


o3d.io.write_point_cloud("scene.ply",pcd)

print("saved scene.ply")