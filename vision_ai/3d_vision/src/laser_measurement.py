import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def generate_mock_laser_signal(
    signal_length=1000, peak_pos=452.3, noise_level=0.15
):
    """レーザー反射光の模擬信号を生成（ガウス形状のピーク + ノイズ）"""
    x = np.arange(signal_length)
    # 理想的なガウスピーク（レーザービームのプロファイル）
    sigma = 12.0
    amplitude = 1.0
    true_signal = amplitude * np.exp(-((x - peak_pos) ** 2) / (2 * sigma**2))

    # ノイズの加算
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, size=signal_length)

    # ベースライン（環境光・オフセット）の追加
    baseline = 0.05 + 0.02 * np.sin(2 * np.pi * x / 500)

    return x, true_signal + noise + baseline


def subpixel_peak_fit(y, peak_idx):
    """3点放物線補間（Parabolic Interpolation）によるサブピクセル精度のピーク検出

    ピーク位置前後の3点を用いて二次関数フィッティングを行い、
    真のピーク位置（小数精度）を算出する。
    """
    if peak_idx <= 0 or peak_idx >= len(y) - 1:
        return float(peak_idx)

    y0 = y[peak_idx - 1]
    y1 = y[peak_idx]
    y2 = y[peak_idx + 1]

    # 二次曲線の頂点オフセット: delta = (y0 - y2) / (2 * (y0 - 2*y1 + y2))
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(peak_idx)

    delta = (y0 - y2) / (2.0 * denom)
    return peak_idx + delta


def process_laser_signal(
    x, raw_signal, sigma=3.0, height_threshold=0.3, distance=20
):
    """レーザー波形の信号処理パイプライン

    1. ガウシアンフィルタによるノイズ除去
    2. ピーク検出（閾値設定）
    3. 放物線補間による精密ピーク位置の推定
    """
    # 1. ノイズ除去（ガウシアンフィルタリング）
    filtered_signal = gaussian_filter1d(raw_signal, sigma=sigma)

    # 2. 整数精度のピーク検出
    peaks_idx, properties = find_peaks(
        filtered_signal, height=height_threshold, distance=distance
    )

    # 3. サブピクセル精度の計算
    subpixel_peaks = []
    for idx in peaks_idx:
        sub_pos = subpixel_peak_fit(filtered_signal, idx)
        subpixel_peaks.append(sub_pos)

    return filtered_signal, peaks_idx, np.array(subpixel_peaks)


# ---------------------------------------------------------
# 実行および可視化
# ---------------------------------------------------------
if __name__ == "__main__":
    # 真のピーク位置を 452.35 と設定して模擬信号作成
    true_peak = 452.35
    x, raw_signal = generate_mock_laser_signal(peak_pos=true_peak)

    # 信号処理の実行
    filtered_signal, peaks_idx, subpixel_peaks = process_laser_signal(
        x, raw_signal, sigma=3.0, height_threshold=0.3
    )

    # 結果表示
    print(f"真のピーク位置: {true_peak:.3f}")
    if len(peaks_idx) > 0:
        print(f"検出ピーク (整数精度): {peaks_idx[0]}")
        print(f"検出ピーク (サブピクセル精度): {subpixel_peaks[0]:.3f}")
        print(f"測定誤差: {abs(subpixel_peaks[0] - true_peak):.4f} index")

    # グラフ描画
    plt.figure(figsize=(10, 5))
    plt.plot(x, raw_signal, label="Raw Signal (with noise)", alpha=0.5, color="gray")
    plt.plot(
        x,
        filtered_signal,
        label="Filtered Signal (Gaussian Filter)",
        color="blue",
        linewidth=1.5,
    )

    # 検出ピークのプロット
    if len(subpixel_peaks) > 0:
        plt.plot(
            peaks_idx,
            filtered_signal[peaks_idx],
            "x",
            label="Integer Peak (find_peaks)",
            color="red",
            markersize=10,
        )
        # サブピクセル補間位置での強さを描画用に補間
        sub_y = np.interp(subpixel_peaks, x, filtered_signal)
        plt.plot(
            subpixel_peaks,
            sub_y,
            "o",
            label="Subpixel Peak (Parabolic Fit)",
            color="green",
            markersize=8,
        )

    plt.axvline(
        x=true_peak,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"True Position ({true_peak})",
    )
    plt.title("Laser Signal Processing & Peak Detection")
    plt.xlabel("Sample Index / Distance Pixel")
    plt.ylabel("Intensity / Signal Amplitude")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()