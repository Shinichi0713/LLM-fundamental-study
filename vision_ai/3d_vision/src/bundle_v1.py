import numpy as np
from scipy.optimize import least_squares

def rodrigues_to_rotation(rvec):
    """Rodriguesベクトルから回転行列を計算"""
    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        return np.eye(3)
    u = rvec / theta
    ux = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])
    R = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.outer(u, u) + ux * np.sin(theta)
    return R

def rotation_to_rodrigues(R):
    """回転行列からRodriguesベクトルを計算"""
    # 簡易実装（小さい回転を仮定）
    # 実用では cv2.Rodrigues を使うのが安全
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta < 1e-6:
        return np.zeros(3)
    lnR = (R - R.T) / (2 * np.sin(theta)) * theta
    rvec = np.array([lnR[2, 1], lnR[0, 2], lnR[1, 0]])
    return rvec

def project_point(K, rvec, t, point3d):
    """3D点をカメラに投影して2D座標を計算"""
    R = rodrigues_to_rotation(rvec)
    # カメラ行列 P = K [R|t]
    P = K @ np.hstack((R, t.reshape(3, 1)))
    point_h = np.hstack((point3d, 1.0))
    proj_h = P @ point_h
    proj = proj_h[:2] / proj_h[2]
    return proj

def bundle_adjustment_residuals(params, observations, camera_indices, point_indices, n_cameras, n_points, fixed_cameras=None, fixed_points=None):
    """
    バンドル調整の残差（再投影誤差）を計算する関数
    
    Parameters:
    - params: 最適化変数 [cam_params, points3d]
    - observations: 観測された2D点のリスト (N, 2)
    - camera_indices: 各観測が属するカメラのインデックス (N,)
    - point_indices: 各観測が属する3D点のインデックス (N,)
    - n_cameras: カメラ数
    - n_points: 3D点数
    - fixed_cameras: 固定するカメラのインデックスリスト（内部パラメータや姿勢を固定）
    - fixed_points: 固定する3D点のインデックスリスト
    """
    # パラメータの分解
    # cam_params: [f, cx, cy, rvec(3), t(3)] の形式を仮定（1カメラあたり 1+2+3+3=9次元）
    cam_param_dim = 9
    points3d_dim = 3

    cam_params = params[:cam_param_dim * n_cameras].reshape(n_cameras, cam_param_dim)
    points3d = params[cam_param_dim * n_cameras:].reshape(n_points, points3d_dim)

    # 固定パラメータの処理（簡易版：ここでは固定カメラ・固定点の扱いは省略）
    # 実装したい場合は、fixed_cameras/fixed_points に基づき cam_params/points3d を上書きする

    residuals = []
    for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        # カメラパラメータの取得
        f, cx, cy, rx, ry, rz, tx, ty, tz = cam_params[cam_idx]
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        rvec = np.array([rx, ry, rz])
        t = np.array([tx, ty, tz])

        # 3D点の取得
        pt3d = points3d[pt_idx]

        # 投影して2D座標を計算
        proj = project_point(K, rvec, t, pt3d)

        # 観測値との差（再投影誤差）
        obs = observations[i]
        residuals.extend(proj - obs)

    return np.array(residuals)

def run_bundle_adjustment(initial_cam_params, initial_points3d, observations, camera_indices, point_indices):
    """
    バンドル調整を実行する関数
    
    Parameters:
    - initial_cam_params: 初期カメラパラメータ (n_cameras, 9)
    - initial_points3d: 初期3D点群 (n_points, 3)
    - observations: 観測2D点 (N, 2)
    - camera_indices: 各観測のカメラインデックス (N,)
    - point_indices: 各観測の点インデックス (N,)
    """
    n_cameras = initial_cam_params.shape[0]
    n_points = initial_points3d.shape[0]

    # 最適化変数の初期値
    x0 = np.hstack((
        initial_cam_params.ravel(),
        initial_points3d.ravel()
    ))

    # 最小二乗法でバンドル調整
    res = least_squares(
        bundle_adjustment_residuals, x0,
        args=(observations, camera_indices, point_indices, n_cameras, n_points),
        method='lm',  # Levenberg–Marquardt
        verbose=2
    )

    # 最適化後のパラメータを分解
    cam_param_dim = 9
    optimized_cam_params = res.x[:cam_param_dim * n_cameras].reshape(n_cameras, cam_param_dim)
    optimized_points3d = res.x[cam_param_dim * n_cameras:].reshape(n_points, 3)

    return optimized_cam_params, optimized_points3d, res

# ===== 使用例 =====

def example_usage():
    # 仮のデータ生成（テスト用）
    n_cameras = 2
    n_points = 10
    n_observations = n_cameras * n_points  # 各カメラがすべての点を見ていると仮定

    # カメラパラメータの初期値（内部＋外部）
    # [f, cx, cy, rvec(3), t(3)]
    initial_cam_params = np.zeros((n_cameras, 9))
    initial_cam_params[:, 0] = 800  # f
    initial_cam_params[:, 1] = 400  # cx
    initial_cam_params[:, 2] = 400  # cy
    # 外部パラメータは適当な値（実用ではSfMの結果から初期化）

    # 3D点の初期値（適当な点群）
    initial_points3d = np.random.randn(n_points, 3) * 10

    # 観測2D点（ここでは「真値＋ノイズ」を模擬）
    observations = []
    camera_indices = []
    point_indices = []
    for cam_idx in range(n_cameras):
        for pt_idx in range(n_points):
            # カメラパラメータ
            f, cx, cy, rx, ry, rz, tx, ty, tz = initial_cam_params[cam_idx]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            rvec = np.array([rx, ry, rz])
            t = np.array([tx, ty, tz])
            pt3d = initial_points3d[pt_idx]
            # 投影
            proj = project_point(K, rvec, t, pt3d)
            # ノイズを加えて観測値とする
            obs = proj + np.random.randn(2) * 0.5
            observations.append(obs)
            camera_indices.append(cam_idx)
            point_indices.append(pt_idx)

    observations = np.array(observations)

    # バンドル調整実行
    optimized_cam_params, optimized_points3d, res = run_bundle_adjustment(
        initial_cam_params, initial_points3d,
        observations, camera_indices, point_indices
    )

    print("Initial residual norm:", np.linalg.norm(bundle_adjustment_residuals(
        np.hstack((initial_cam_params.ravel(), initial_points3d.ravel())),
        observations, camera_indices, point_indices, n_cameras, n_points
    )))
    print("Optimized residual norm:", res.cost)

if __name__ == "__main__":
    example_usage()