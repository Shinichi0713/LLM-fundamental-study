import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. 時系列データの生成 (サイン波)
# ----------------------------------------------------
def create_time_series(length):
    """サイン波に基づいた時系列データを生成する"""
    time = np.linspace(0, 100, length)
    series = np.sin(time) + np.random.normal(0, 0.1, length) # ノイズを少し加える
    return series.astype(np.float32)

SERIES_LENGTH = 1000
RAW_DATA = create_time_series(SERIES_LENGTH)

# ----------------------------------------------------
# 2. データの整形（シーケンス化）
# ----------------------------------------------------
def create_sequences(data, sequence_length):
    """
    時系列データをGRUが学習できる (入力シーケンス, ターゲット) のペアに変換する
    例: sequence_length=10 の場合
    入力: [t0, t1, ..., t9] -> ターゲット: [t10]
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # 入力X: iから i+sequence_length までのシーケンス
        X.append(data[i : i + sequence_length])
        # ターゲットy: 次のステップ (i+sequence_length) の値
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# 時系列長（過去どれくらいの期間を見て予測するか）
SEQUENCE_LENGTH = 50
X, y = create_sequences(RAW_DATA, SEQUENCE_LENGTH)

# 訓練データとテストデータに分割
TRAIN_SIZE = int(0.8 * len(X))
X_train, X_test = X[:TRAIN_SIZE], X[TRAIN_SIZE:]
y_train, y_test = y[:TRAIN_SIZE], y[TRAIN_SIZE:]

# GRUの入力形式にリシェイプ: (サンプル数, タイムステップ, 特徴量)
# 現在は単変量時系列なので、特徴量は1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"訓練データ形状 (X_train): {X_train.shape}")
print(f"ターゲット形状 (y_train): {y_train.shape}")

# ----------------------------------------------------
# 3. GRUモデルの構築と訓練
# ----------------------------------------------------
def build_gru_model(input_shape):
    model = Sequential([
        # 最初のGRUレイヤ: return_sequences=Trueで次のGRUレイヤにシーケンス全体を渡す
        GRU(units=64, return_sequences=True, input_shape=input_shape),
        # 2番目のGRUレイヤ: return_sequences=Falseで最後の出力のみを後続に渡す
        GRU(units=64),
        # 全結合層: 最終的な出力（次のステップの値）は1つ
        Dense(units=1) 
    ])
    
    # 損失関数: 平均二乗誤差 (回帰タスクのため)
    model.compile(optimizer='adam', loss='mse')
    return model

INPUT_SHAPE = (SEQUENCE_LENGTH, 1)
model = build_gru_model(INPUT_SHAPE)
model.summary()

# モデルの訓練
EPOCHS = 20
BATCH_SIZE = 32
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)

# ----------------------------------------------------
# 4. 予測と結果の可視化
# ----------------------------------------------------
# テストデータで予測を実行
y_pred = model.predict(X_test)

# グラフ描画
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Values (Test Data)')
plt.plot(y_pred, label='GRU Predictions')
plt.title(f'GRU Time Series Prediction (Sequence Length: {SEQUENCE_LENGTH})')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 損失の推移（学習曲線）の描画
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Progression')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()