import cv2
import numpy as np
from scipy.optimize import least_squares

def extract_features(image_path):
    """画像から特徴点と記述子を抽出（SIFT）"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc, img.shape

def match_features(desc1, desc2):
    """2つの記述子をマッチング（BFMatcher + Lowe's ratio test）"""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def estimate_pose(kp1, kp2, matches, K):
    """マッチング結果からエッセンシャル行列とカメラ姿勢を推定"""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # エッセンシャル行列（RANSAC）
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    mask = mask.ravel().astype(bool)

    # カメラ姿勢の復元（4通りの解から正しいものを選択）
    _, R, t, _ = cv2.recoverPose(E, pts1[mask], pts2[mask], K)
    return R, t, pts1[mask], pts2[mask]

def triangulate_points(K, R1, t1, R2, t2, pts1, pts2):
    """2つのカメラから三角測量で3D点を計算"""
    # カメラ行列 P1, P2
    P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
    P2 = K @ np.hstack((R2, t2.reshape(3, 1)))

    # 同次座標に変換
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3)

    # 三角測量
    points4d = cv2.triangulatePoints(P1, P2, pts1_h.T, pts2_h.T)
    points3d = points4d[:3] / points4d[3]  # 同次座標 → 3D座標
    return points3d.T

def reprojection_error(params, points2d, K, n_cameras, n_points, camera_indices, point_indices):
    """バンドル調整用：再投影誤差を計算"""
    # params: [cam_params (R,t for each camera), points3d]
    camera_params = params[:6 * n_cameras].reshape(n_cameras, 6)
    points3d = params[6 * n_cameras:].reshape(n_points, 3)

    errors = []
    for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        R_vec = camera_params[cam_idx, :3]
        t = camera_params[cam_idx, 3:]
        R, _ = cv2.Rodrigues(R_vec)
        P = K @ np.hstack((R, t.reshape(3, 1)))
        pt3d_h = np.hstack((points3d[pt_idx], 1.0))
        proj = P @ pt3d_h
        proj = proj[:2] / proj[2]
        errors.extend(proj - points2d[i])
    return np.array(errors)

def bundle_adjustment(points3d, points2d_list, camera_params_list, K, camera_indices, point_indices):
    """簡易バンドル調整（Levenberg-Marquardt）"""
    n_cameras = len(camera_params_list)
    n_points = len(points3d)

    # 初期パラメータ（カメラパラメータ + 3D点）
    x0 = np.hstack((
        np.vstack(camera_params_list).ravel(),
        points3d.ravel()
    ))

    res = least_squares(
        reprojection_error, x0,
        args=(np.vstack(points2d_list), K, n_cameras, n_points, camera_indices, point_indices),
        method='lm', verbose=0
    )
    return res.x

def main():
    # 画像パス（2枚）
    img1_path = "image1.jpg"
    img2_path = "image2.jpg"

    # カメラ内部パラメータ（例：焦点距離 f=800, 主点 cx=cy=400 の仮定）
    K = np.array([
        [800, 0, 400],
        [0, 800, 400],
        [0, 0, 1]
    ])

    # 1. 特徴点抽出
    kp1, desc1, shape1 = extract_features(img1_path)
    kp2, desc2, shape2 = extract_features(img2_path)

    # 2. 特徴点マッチング
    matches = match_features(desc1, desc2)
    print(f"Good matches: {len(matches)}")

    # 3. カメラ姿勢推定（第1カメラを世界座標系の原点とする）
    R1 = np.eye(3)
    t1 = np.zeros(3)
    R2, t2, pts1, pts2 = estimate_pose(kp1, kp2, matches, K)

    # 4. 三角測量で3D点群を計算
    points3d = triangulate_points(K, R1, t1, R2, t2, pts1, pts2)
    print(f"Reconstructed 3D points: {points3d.shape}")

    # 5. （簡易）バンドル調整
    # カメラパラメータ（Rodriguesベクトル + 並進ベクトル）
    rvec1, _ = cv2.Rodrigues(R1)
    rvec2, _ = cv2.Rodrigues(R2)
    camera_params_list = [
        np.hstack((rvec1.ravel(), t1)),
        np.hstack((rvec2.ravel(), t2))
    ]

    # 対応する2D点のリストとインデックス
    points2d_list = [pts1, pts2]
    camera_indices = [0] * len(pts1) + [1] * len(pts2)
    point_indices = list(range(len(pts1))) + list(range(len(pts2)))

    # バンドル調整実行
    optimized_params = bundle_adjustment(
        points3d, points2d_list, camera_params_list, K,
        camera_indices, point_indices
    )

    # 最適化後の3D点を抽出
    n_cameras = 2
    optimized_points3d = optimized_params[6 * n_cameras:].reshape(points3d.shape)
    print("Bundle adjustment completed.")

    # ここで optimized_points3d を可視化（例：matplotlibで3Dプロット）など

if __name__ == "__main__":
    main()