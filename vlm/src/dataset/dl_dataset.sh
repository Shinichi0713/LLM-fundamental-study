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


