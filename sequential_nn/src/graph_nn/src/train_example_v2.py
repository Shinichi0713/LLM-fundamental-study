import matplotlib.pyplot as plt

# =====================================================================
# 読み込んだデータ（画像とラベル）を10枚並べて表示
# =====================================================================
plt.figure(figsize=(15, 3))

# 先頭から10枚の画像を表示
for index in range(10):
    # 8x8のグリッド状にサブプロットを配置 (1行10列)
    plt.subplot(1, 10, index + 1)
    
    # digits.images には、すでに 8x8 の行列データが入っています
    # cmap=plt.cm.gray_r で「白黒反転（文字を黒、背景を白）」にして見やすくします
    plt.imshow(digits.images[index], cmap=plt.cm.gray_r, interpolation='nearest')
    
    # 各画像の上部に、対応するラベル（目的変数）を表示
    plt.title(f"Label: {digits.target[index]}", fontsize=12, fontweight="bold")
    
    # グラフの目盛り（軸の数字）は邪魔なので非表示にします
    plt.axis('off')

plt.suptitle("Input Data (Features: 8x8 Pixels) & Target Labels", fontsize=14, y=1.1, fontweight="bold")
plt.show()

# =====================================================================
# データの「中身の形状（次元）」もテキストで確認
# =====================================================================
print("\n=== データ構造の詳細 ===")
print(f"1枚目の画像（行列）の形状: {digits.images[0].shape} (縦8ピクセル × 横8ピクセル)")
print(f"GNNに入力した説明変数（1次元化）の形状: {digits.data[0].shape} (64次元のベクトル)")
print(f"1枚目の画像の実データ（ピクセル値の一例）:\n{digits.data[0].round().astype(int).reshape(8,8)}")


