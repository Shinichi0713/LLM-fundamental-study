!wget https://huggingface.co/datasets/aialliance/GEOBench-VLM/resolve/main/Single.parquet?download=true
!unzip /content/Temporal.zip?download=true
!wget https://huggingface.co/datasets/aialliance/GEOBench-VLM/resolve/main/Temporal.zip?download=true

# Boost関連パッケージをインストール
!sudo apt update
!sudo apt install -y libboost-all-dev
!sudo apt install -y libeigen3-dev
!sudo apt install -y libopenimageio-dev
!sudo apt install -y libopenexr-dev
!sudo apt install -y openimageio-tools
!sudo apt install -y libsuitesparse-dev
!sudo apt install -y libceres-dev
!sudo apt install -y libcgal-dev
!sudo apt install -y qt5-default libqt5opengl5-dev
!sudo apt install -y qtbase5-dev qttools5-dev libqt5opengl5-dev
!sudo apt install -y libqt5svg5-dev
!sudo apt install -y libglew-dev libmetis-dev


%cd /content/colmap/build
!rm -rf CMakeCache.txt CMakeFiles/
!cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..
!ninja


!wget https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip
!unzip south-building.zip

%cd /content/colmap/build/src/colmap/exe

# 1. 特徴抽出
!/content/colmap/build/src/colmap/exe/colmap feature_extractor \
  --database_path /content/south-building/database.db \
  --image_path /content/south-building/images

# 2. 特徴マッチング
!/content/colmap/build/src/colmap/exe/colmap exhaustive_matcher \
  --database_path /content/south-building/database.db

# 3. SfM（スパース再構成）
!mkdir -p /content/south-building/sparse
!/content/colmap/build/src/colmap/exe/colmap mapper \
  --database_path /content/south-building/database.db \
  --image_path /content/south-building/images \
  --output_path /content/south-building/sparse

# 4. 画像の歪み補正
!mkdir -p /content/south-building/dense
!/content/colmap/build/src/colmap/exe/colmap image_undistorter \
  --image_path /content/south-building/images \
  --input_path /content/south-building/sparse/0 \
  --output_path /content/south-building/dense \
  --output_type COLMAP

# 5. MVS（デンス再構成）
!/content/colmap/build/src/colmap/exe/colmap patch_match_stereo \
  --workspace_path /content/south-building/dense

!/content/colmap/build/src/colmap/exe/colmap stereo_fusion \
  --workspace_path /content/south-building/dense \
  --output_path /content/south-building/dense/fused.ply
