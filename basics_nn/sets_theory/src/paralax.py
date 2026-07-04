import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 画像読み込み（左・右）
img_left = cv2.imread('left.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('right.jpg', cv2.IMREAD_GRAYSCALE)

assert img_left is not None and img_right is not None, "画像の読み込みに失敗しました"

# 2. 特徴点検出と特徴量記述（SIFT）
sift = cv2.SIFT_create()
kp_left, desc_left = sift.detectAndCompute(img_left, None)
kp_right, desc_right = sift.detectAndCompute(img_right, None)

print(f"左画像の特徴点数: {len(kp_left)}")
print(f"右画像の特徴点数: {len(kp_right)}")

# 3. 特徴マッチング（Brute-Force + L2距離）
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(desc_left, desc_right, k=2)

# 4. Loweの比率テストで良いマッチのみを抽出
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(f"良いマッチ数: {len(good_matches)}")

# 5. 対応点の座標を取得
pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])

# 6. 視差（disparity）の計算（x方向の差）
# ステレオ画像が水平方向にずれている前提
disparities = pts_left[:, 0] - pts_right[:, 0]

print(f"視差の統計:")
print(f"  最小視差: {disparities.min():.2f} px")
print(f"  最大視差: {disparities.max():.2f} px")
print(f"  平均視差: {disparities.mean():.2f} px")

# 7. 簡易な視差マップの可視化（対応点のみ）
h, w = img_left.shape
disparity_map = np.zeros((h, w), dtype=np.float32)

for (xL, yL), (xR, yR), disp in zip(pts_left, pts_right, disparities):
    y, x = int(round(yL)), int(round(xL))
    if 0 <= y < h and 0 <= x < w:
        disparity_map[y, x] = disp

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_left, cmap='gray')
plt.title('Left Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_right, cmap='gray')
plt.title('Right Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(disparity_map, cmap='jet')
plt.colorbar(label='Disparity (pixels)')
plt.title('Disparity Map (Sparse)')
plt.axis('off')

plt.tight_layout()
plt.show()

# 高密度ステレオマッチング（SGBM）
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,   # 視差の探索範囲
    blockSize=11,
    P1=8*3*11**2,
    P2=32*3*11**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

disparity_sgbm = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

plt.imshow(disparity_sgbm, cmap='jet')
plt.colorbar(label='Disparity (pixels)')
plt.title('Dense Disparity Map (SGBM)')
plt.axis('off')
plt.show()

# 高密度ステレオマッチング（SGBM）
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,   # 視差の探索範囲
    blockSize=11,
    P1=8*3*11**2,
    P2=32*3*11**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

disparity_sgbm = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

plt.imshow(disparity_sgbm, cmap='jet')
plt.colorbar(label='Disparity (pixels)')
plt.title('Dense Disparity Map (SGBM)')
plt.axis('off')
plt.show()