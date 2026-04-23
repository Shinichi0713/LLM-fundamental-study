import cv2
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
assert img is not None, "画像が読み込めませんでした"

# FAST検出器
fast = cv2.FastFeatureDetector_create(threshold=30)

# キーポイント検出
keypoints = fast.detect(img, None)

# 描画
img_kp = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0))

plt.imshow(img_kp, cmap="gray")
plt.title("FAST keypoints")
plt.axis("off")
plt.show()