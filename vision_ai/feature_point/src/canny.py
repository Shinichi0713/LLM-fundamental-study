import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像読み込み（グレースケール）
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
assert img is not None, "画像が読み込めませんでした"

# Sobelエッジ（x方向・y方向）
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobelx**2 + sobely**2)  # 勾配の大きさ

# Cannyエッジ
edges_canny = cv2.Canny(img, threshold1=50, threshold2=150)

# 可視化
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.subplot(1, 3, 2)
plt.imshow(sobel_mag, cmap="gray")
plt.title("Sobel magnitude")
plt.subplot(1, 3, 3)
plt.imshow(edges_canny, cmap="gray")
plt.title("Canny edges")
plt.tight_layout()
plt.show()