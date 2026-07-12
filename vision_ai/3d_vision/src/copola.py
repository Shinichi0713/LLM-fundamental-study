import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# points3D.txt を読み込む関数
def read_points3d_txt(path):
    points = []
    with open(path, 'r') as f:
        for line in f:
            # コメント行や空行をスキップ
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                # フォーマット: POINT3D_ID X Y Z ...
                x, y, z = map(float, parts[1:4])
                points.append([x, y, z])
    return np.array(points)

# スパース点群を読み込み
points = read_points3d_txt("/content/south-building/sparse/0/points3D.txt")

# 点が多すぎる場合はサンプリング
if len(points) > 100000:
    points = points[np.random.choice(len(points), 100000, replace=False)]

# 3Dプロット
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("COLMAP Sparse Reconstruction")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_images_txt(path):
    cameras = []
    with open(path, 'r') as f:
        lines = f.readlines()
    # 各行のフォーマット: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.strip().split()
        if len(parts) >= 8:
            # クォータニオン (QW, QX, QY, QZ) と 平行移動 (TX, TY, TZ)
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cameras.append([tx, ty, tz])
    return np.array(cameras)

# カメラ位置を読み込み
cameras = read_images_txt("/content/south-building/sparse/0/images.txt")

# 点群を読み込み
points = read_points3d_txt("/content/south-building/sparse/0/points3D.txt")

# サンプリング（必要に応じて）
if len(points) > 100000:
    points = points[np.random.choice(len(points), 100000, replace=False)]

# 3Dプロット
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5, label='3D Points')
ax.scatter(cameras[:, 0], cameras[:, 1], cameras[:, 2], s=20, c='red', marker='^', label='Cameras')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.title("COLMAP Sparse Reconstruction with Cameras")
plt.show()