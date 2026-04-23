import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
assert img is not None, "画像が読み込めませんでした"

# LBPパラメータ
radius = 1
n_points = 8 * radius

# LBP画像を計算
lbp = local_binary_pattern(img, n_points, radius, method="uniform")

# 可視化
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(lbp, cmap="gray")
plt.title("LBP image")
plt.tight_layout()
plt.show()