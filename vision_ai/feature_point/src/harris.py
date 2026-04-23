import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
assert img is not None, "画像が読み込めませんでした"

# Harrisコーナー検出
dst = cv2.cornerHarris(img.astype(np.float32), blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)  # コーナーを強調

# 閾値以上の点をコーナーとしてマーク
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_color[dst > 0.01 * dst.max()] = [0, 0, 255]  # 赤点

plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Harris corners")
plt.axis("off")
plt.show()