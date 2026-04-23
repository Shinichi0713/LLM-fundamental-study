import cv2
import numpy as np
from skimage.filters import gabor
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
assert img is not None, "画像が読み込めませんでした"

# Gaborフィルタ適用（実部・虚部）
freq = 0.1
theta = np.pi / 4  # 45度
real, imag = gabor(img, frequency=freq, theta=theta)

# 応答の大きさ
gabor_mag = np.sqrt(real**2 + imag**2)

# 可視化
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.subplot(1, 3, 2)
plt.imshow(real, cmap="gray")
plt.title("Gabor real")
plt.subplot(1, 3, 3)
plt.imshow(gabor_mag, cmap="gray")
plt.title("Gabor magnitude")
plt.tight_layout()
plt.show()