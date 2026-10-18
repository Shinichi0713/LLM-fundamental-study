# 1. データ準備
python prepare_cifar10.py

# 2. ビルド（libtorch のパスは環境に合わせてください）
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build . --parallel

# 3. 実行
./dynalay