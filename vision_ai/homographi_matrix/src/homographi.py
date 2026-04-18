import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_homography(image_path, H, output_size=(400, 400)):
    """
    ホモグラフィー行列 H を画像に適用し、変換前後の画像を可視化する
    
    Parameters
    ----------
    image_path : str
        入力画像のパス
    H : np.ndarray, shape (3, 3)
        ホモグラフィー行列
    output_size : tuple (width, height)
        出力画像のサイズ
    """
    # 画像読み込み
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCVはBGRなのでRGBに変換

    h, w = img.shape[:2]

    # 元画像の4隅の座標（同次座標）
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ], dtype=np.float32).T  # shape: (3, 4)

    # ホモグラフィー変換を適用
    transformed_corners = H @ corners  # shape: (3, 4)
    transformed_corners = transformed_corners / transformed_corners[2]  # 正規化

    # 変換後の四角形の範囲を取得
    x_min = np.min(transformed_corners[0])
    x_max = np.max(transformed_corners[0])
    y_min = np.min(transformed_corners[1])
    y_max = np.max(transformed_corners[1])

    # 出力画像のサイズに合わせてホモグラフィーを調整（平行移動とスケール）
    scale_x = output_size[0] / (x_max - x_min)
    scale_y = output_size[1] / (y_max - y_min)
    scale = min(scale_x, scale_y)

    # スケールと平行移動を追加したホモグラフィー
    T = np.array([
        [scale, 0, -x_min * scale],
        [0, scale, -y_min * scale],
        [0, 0, 1]
    ])
    H_adjusted = T @ H

    # 画像をホモグラフィー変換
    warped = cv2.warpPerspective(
        img, H_adjusted, output_size,
        flags=cv2.INTER_LINEAR
    )

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(warped)
    axes[1].set_title("Transformed by Homography")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 適当なホモグラフィー行列の例
    # ここでは「少し回転＋スケール＋せん断」を含む変換を設定
    H_example = np.array([
        [1.2, 0.1, 50],
        [-0.1, 0.9, 30],
        [0.0002, 0.0001, 1]
    ], dtype=np.float32)

    # 画像ファイルのパスを指定
    image_path = "your_image.jpg"  # ここを実際の画像パスに変更

    visualize_homography(image_path, H_example)