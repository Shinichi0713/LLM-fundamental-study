# Google Colab で mp4 を gif に変換するコード

# 1. ffmpeg と Pillow のインストール（必要な場合）
!apt-get update > /dev/null 2>&1
!apt-get install -y ffmpeg > /dev/null 2>&1
!pip install Pillow > /dev/null 2>&1

# 2. mp4 を gif に変換する関数
import subprocess
from PIL import Image

def mp4_to_gif(mp4_path, gif_path, fps=10, scale=None):
    """
    mp4 を gif に変換する
    
    Parameters
    ----------
    mp4_path : str
        入力 mp4 ファイルのパス
    gif_path : str
        出力 gif ファイルのパス
    fps : int, default 10
        gif のフレームレート（低くすると軽くなる）
    scale : tuple or None, default None
        リサイズサイズ (width, height)。None なら元サイズ
    """
    # 一時ディレクトリにフレームを保存
    import os
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    # ffmpeg で mp4 をフレーム画像に分解
    if scale is None:
        scale_cmd = ""
    else:
        scale_cmd = f",scale={scale[0]}:{scale[1]}"
    
    cmd = [
        "ffmpeg", "-i", mp4_path,
        "-vf", f"fps={fps}{scale_cmd}",
        f"{temp_dir}/frame_%04d.png"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # フレーム画像を読み込んで gif に変換
    frames = []
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".png")])
    
    for frame_file in frame_files:
        frame_path = os.path.join(temp_dir, frame_file)
        img = Image.open(frame_path)
        frames.append(img.copy())
        img.close()
    
    # gif として保存
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000//fps,  # ミリ秒単位
            loop=0
        )
        print(f"GIF saved: {gif_path}")
    else:
        print("No frames found.")
    
    # 一時ファイルを削除
    import shutil
    shutil.rmtree(temp_dir)

# 3. 実行例
# 例: kalman_vs_lowpass_circle.mp4 を kalman_vs_lowpass_circle.gif に変換
mp4_to_gif(
    mp4_path="kalman_vs_lowpass_circle.mp4",
    gif_path="kalman_vs_lowpass_circle.gif",
    fps=10,                    # gif のフレームレート（軽量化のため低め推奨）
    scale=(400, 300)           # リサイズ（幅, 高さ）。None なら元サイズ
)

# 4. 生成された gif を Colab 上で表示
from IPython.display import Image as IPImage
IPImage("kalman_vs_lowpass_circle.gif")