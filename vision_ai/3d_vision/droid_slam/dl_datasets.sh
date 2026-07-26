# 1. データの保存先ディレクトリを作成
!mkdir -p /content/datasets/TUM_RGBD
%cd /content/datasets/TUM_RGBD

# 2. TUM RGB-Dの検証用軽量シーケンス（fr1/xyz）をダウンロード
# （カメラがXYZ軸方向に少しずつ動く、SLAMの初期テストに最適なデータです：約440MB）
!wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz

# 3. アーカイブの解凍
print("解凍中...")
!tar -xf rgbd_dataset_freiburg1_xyz.tgz

# 4. ダウンロードしたファイルの削除（Colabのストレージ容量節約のため）
!rm rgbd_dataset_freiburg1_xyz.tgz

# 5. 中身の確認
print("データセットの準備が完了しました。構造は以下の通りです：")
!ls -l rgbd_dataset_freiburg1_xyz/

# GPUの確認（必ずGPUインスタンスを選択してください）
!nvidia-smi

# DROID-SLAMリポジトリのクローン
%cd /content
!git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
%cd DROID-SLAM

# 必要なPythonパッケージのインストール
!pip install open3d
!pip install trimesh
!pip install gdown

# DROID-SLAMに必須な「torch geometry（lietorch）」のビルドとインストール
# ※ColabのPytorch/CUDAバージョンに合わせてコンパイルするため少々時間がかかります（数分）
%cd thirdparty/lietorch
!python setup.py install
%cd /content/DROID-SLAM

# DROID-SLAM 本体のC++/CUDA拡張のビルド
!python setup.py install

# 重み保存用ディレクトリの作成
!mkdir -p /content/DROID-SLAM/checkpoints
%cd /content/DROID-SLAM/checkpoints

# 公式のGoogle Driveから重みをダウンロード（gdownを利用）
!gdown --id 1O9gInV94v8YgYw3Xw4E_T7g0g9K-v6-R -O droid.pth

%cd /content/DROID-SLAM

# TUM RGB-Dのカメラパラメータ、先ほどDLしたデータセット、および重みファイルを指定して実行
!python demo.py \
  --imagedir=/content/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz/rgb \
  --calib=calib/tum1.txt \
  --weights=checkpoints/droid.pth \
  --reconstruct

# 保存先ディレクトリに移動
!mkdir -p /content/DROID-SLAM/checkpoints
%cd /content/DROID-SLAM/checkpoints

# Hugging FaceにホストされているDROID-SLAM公式のdroid.pthをダウンロード
!wget -O droid.pth https://huggingface.co/datasets/vinesmsuic/droid-slam-weights/resolve/main/droid.pth

# ダウンロードが成功したか確認（約100MB〜200MBほどのファイルがあればOKです）
print("チェックポイントの確認:")
!ls -lh droid.pth

# 元のディレクトリに戻る
%cd /content/DROID-SLAM

import os

# ディレクトリを作成
os.makedirs('/content/DROID-SLAM/calib', exist_ok=True)

# TUM Freiburg1 (fr1) のカメラパラメータを書き込む
# 記述形式: fx fy cx cy
# TUM fr1の標準値: fx=517.3, fy=516.5, cx=318.6, cy=255.3
calib_content = "517.3 516.5 318.6 255.3"

calib_path = '/content/DROID-SLAM/calib/tum1.txt'
with open(calib_path, 'w') as f:
    f.write(calib_content)

print(f"キャリブレーションファイルを新規作成しました: {calib_path}")
print(f"中身: {calib_content}")

%cd /content/DROID-SLAM/checkpoints

# 1. 一度壊れたファイルを完全に削除
!rm -f droid.pth

# 2. 確実な公式提供の別ミラーURLから再ダウンロード
!wget --no-check-certificate -O droid.pth "https://huggingface.co/vslamlab/droidslam/resolve/main/droid.pth?download=true"

# 3. ファイルサイズの確認（正常なら 70MB〜80MB 前後あります）
# もしここで 0 バイトだったり数KBならダウンロードに失敗しています
print("再ダウンロードした重みのサイズを確認します:")
!ls -lh droid.pth

%cd /content/DROID-SLAM

import torch

# ColabのPyTorchとCUDAのバージョンを自動取得
pyt_version = torch.__version__.split('+')[0]
cuda_version = torch.version.cuda.replace('.', '')

print(f"PyTorch Version: {pyt_version}")
print(f"CUDA Version: {cuda_version}")

# バージョンに適合するtorch-scatterをインストール
# （URL例: https://data.pyg.org/whl/torch-2.5.0+cu124.html のような形を自動生成します）
!pip install torch-scatter -f https://data.pyg.org/whl/torch-{pyt_version}+cu{cuda_version}.html