# Colab上でCOLMAPをソースからビルドする例
"""
# 依存ライブラリのインストール
!apt-get update
!apt-get install -y \
  cmake \
  build-essential \
  libboost-program-options-dev \
  libboost-filesystem-dev \
  libboost-graph-dev \
  libboost-system-dev \
  libboost-test-dev \
  libeigen3-dev \
  libsuitesparse-dev \
  libfreeimage-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev \
  libcgal-dev \
  libcgal-qt5-dev \
  libatlas-base-dev \
  libsuitesparse-dev

# Ceres Solver（最適化ライブラリ）のインストール
!apt-get install -y libceres-dev

# COLMAPのソースコードを取得
!git clone https://github.com/colmap/colmap.git
%cd colmap

# ビルドディレクトリ作成
!mkdir build
%cd build

# CMakeでビルド設定（CUDAはColab環境でうまくいかない場合があるため、OFFにしておく例）
!cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ENABLED=OFF \
  -DCUDA_ARCHS=""

# ビルド実行（並列ビルドで高速化）
!make -j$(nproc)

# インストール（/usr/local/bin などにバイナリを配置）
!sudo make install

# パスを通す（任意）
import os
os.environ['PATH'] = '/usr/local/bin:' + os.environ['PATH']

# インストール確認
!colmap -h


from google.colab import drive
drive.mount('/content/drive')

# 例: /content/drive/MyDrive/colmap_images に画像を置いている場合
image_dir = '/content/drive/MyDrive/colmap_images'
project_dir = '/content/colmap_project'  # 作業ディレクトリ

import os
os.makedirs(project_dir, exist_ok=True)

# プロジェクトディレクトリに移動
%cd /content/colmap_project

# 1. 特徴抽出（feature_extractor）
!colmap feature_extractor \
  --database_path database.db \
  --image_path "$image_dir" \
  --ImageReader.single_camera 1

# 2. 特徴マッチング（exhaustive_matcher）
!colmap exhaustive_matcher \
  --database_path database.db

# 3. SfMによるスパース復元（incremental_mapper）
!colmap mapper \
  --database_path database.db \
  --image_path "$image_dir" \
  --output_path sparse

# 4. MVSによる高密度復元（image_undistorter + patch_match_stereo + stereo_fusion）
# 4-1. 画像の歪み補正
!colmap image_undistorter \
  --image_path "$image_dir" \
  --input_path sparse/0 \
  --output_path dense \
  --output_type COLMAP

# 4-2. 高密度ステレオマッチング（パッチマッチ）
!colmap patch_match_stereo \
  --workspace_path dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

# 4-3. 点群の融合
!colmap stereo_fusion \
  --workspace_path dense \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path dense/fused.ply

print("Dense point cloud saved to dense/fused.ply")

# 生成された点群ファイルの確認
import os
ply_path = '/content/colmap_project/dense/fused.ply'
if os.path.exists(ply_path):
    print(f"Point cloud generated: {ply_path}")
    # サイズ確認
    size_mb = os.path.getsize(ply_path) / (1024*1024)
    print(f"File size: {size_mb:.2f} MB")
else:
    print("Point cloud file not found.")

# Driveにコピー（任意）
!cp "$ply_path" "/content/drive/MyDrive/"
"""

