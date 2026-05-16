import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
from google.colab import drive

# Google Drive をマウント（動画を保存するため）
drive.mount('/content/drive')

# 動画保存先ディレクトリ
VIDEO_DIR = "/content/drive/MyDrive/rl_boxing/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# 環境とエージェントの準備
env = get_env()
agent = MAPPOAgent(action_space_n=18)

# 最新のチェックポイントをロード
def find_latest_checkpoint(checkpoint_dir):
    import glob
    import re
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "mappo_agent_iter_*.pth"))
    if not checkpoints:
        return None, 0
    ids = [int(re.findall(r"iter_(\d+)", f)[0]) for f in checkpoints]
    latest_idx = np.argmax(ids)
    return checkpoints[latest_idx], ids[latest_idx]

CHECKPOINT_DIR = "/content/drive/MyDrive/rl_boxing/checkpoints"
checkpoint_path, _ = find_latest_checkpoint(CHECKPOINT_DIR)

if checkpoint_path:
    print(f"Loading checkpoint: {checkpoint_path}")
    # trainer 経由でロードする場合（あなたのコードに合わせてください）
    # trainer.load_checkpoint(checkpoint_path)
    # agent = trainer.agent
    # あるいは agent.load_state_dict(torch.load(checkpoint_path))
    print("Checkpoint loaded.")
else:
    print("No checkpoint found.")
    exit()

# 動画保存用の環境ラッパー
# render_mode="rgb_array" を指定するのがポイント
env = gym.wrappers.RecordVideo(
    env,
    video_folder=VIDEO_DIR,
    episode_trigger=lambda episode_id: True,  # 全エピソード保存
    disable_logger=True,
)

# 評価モード（推論モード）に切り替え
agent.eval()

# 1エピソード分プレイして動画を保存
obs, info = env.reset()
terminated, truncated = False, False
total_reward = 0.0

while not (terminated or truncated):
    # エージェントが action を返すメソッドに合わせて修正してください
    # 例: action = agent.act(obs)
    action = agent.act(obs)  # ここは実装に合わせて変更

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

env.close()

print(f"Episode finished. Total reward: {total_reward:.2f}")
print(f"Video saved to {VIDEO_DIR}")