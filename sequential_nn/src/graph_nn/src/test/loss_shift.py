import re
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. ログデータの流し込み
# ---------------------------------------------------------
# 提供されたログをそのままトリプルクォーテーション内に貼り付けています
log_data = """
Episode 0: Reward = -126.81, Captures = 0, Avg Reward = -126.81, Avg Captures = 0.00
Actor Loss: -0.0154 | Critic Loss: 0.1027 | Avg Entropy: 1.6085
Episode 1: Reward = -96.68, Captures = 0, Avg Reward = -111.75, Avg Captures = 0.00
Actor Loss: -0.0157 | Critic Loss: 0.1005 | Avg Entropy: 1.6085
Episode 2: Reward = -125.40, Captures = 0, Avg Reward = -116.30, Avg Captures = 0.00
Actor Loss: -0.0158 | Critic Loss: 0.0900 | Avg Entropy: 1.6085
Episode 3: Reward = -158.32, Captures = 0, Avg Reward = -126.80, Avg Captures = 0.00
Actor Loss: -0.0179 | Critic Loss: 0.1070 | Avg Entropy: 1.6085
Episode 4: Reward = -95.05, Captures = 0, Avg Reward = -120.45, Avg Captures = 0.00
Actor Loss: -0.0167 | Critic Loss: 0.1072 | Avg Entropy: 1.6086
Episode 5: Reward = -142.06, Captures = 0, Avg Reward = -124.05, Avg Captures = 0.00
Actor Loss: -0.0152 | Critic Loss: 0.0926 | Avg Entropy: 1.6085
Episode 6: Reward = -140.07, Captures = 0, Avg Reward = -126.34, Avg Captures = 0.00
Actor Loss: -0.0163 | Critic Loss: 0.0854 | Avg Entropy: 1.6086
Episode 7: Reward = -121.78, Captures = 0, Avg Reward = -125.77, Avg Captures = 0.00
Actor Loss: -0.0175 | Critic Loss: 0.0947 | Avg Entropy: 1.6086
Episode 8: Reward = -149.45, Captures = 0, Avg Reward = -128.40, Avg Captures = 0.00
Actor Loss: -0.0142 | Critic Loss: 0.0880 | Avg Entropy: 1.6086
Episode 9: Reward = -116.19, Captures = 0, Avg Reward = -127.18, Avg Captures = 0.00
Actor Loss: -0.0142 | Critic Loss: 0.0961 | Avg Entropy: 1.6086
Episode 10: Reward = -148.00, Captures = 0, Avg Reward = -129.07, Avg Captures = 0.00
Actor Loss: -0.0157 | Critic Loss: 0.0856 | Avg Entropy: 1.6086
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 11: Reward = -133.20, Captures = 0, Avg Reward = -129.42, Avg Captures = 0.00
Actor Loss: -0.0155 | Critic Loss: 0.0845 | Avg Entropy: 1.6086
Episode 12: Reward = -133.22, Captures = 0, Avg Reward = -129.71, Avg Captures = 0.00
Actor Loss: -0.0152 | Critic Loss: 0.0966 | Avg Entropy: 1.6086
Episode 13: Reward = -72.73, Captures = 0, Avg Reward = -125.64, Avg Captures = 0.00
Actor Loss: -0.0169 | Critic Loss: 0.0961 | Avg Entropy: 1.6086
Episode 14: Reward = -139.48, Captures = 0, Avg Reward = -126.56, Avg Captures = 0.00
Actor Loss: -0.0142 | Critic Loss: 0.0665 | Avg Entropy: 1.6086
Episode 15: Reward = -143.87, Captures = 0, Avg Reward = -127.64, Avg Captures = 0.00
Actor Loss: -0.0168 | Critic Loss: 0.0616 | Avg Entropy: 1.6086
Episode 16: Reward = -116.86, Captures = 0, Avg Reward = -127.01, Avg Captures = 0.00
Actor Loss: -0.0170 | Critic Loss: 0.0652 | Avg Entropy: 1.6086
Episode 17: Reward = -138.34, Captures = 0, Avg Reward = -127.64, Avg Captures = 0.00
Actor Loss: -0.0179 | Critic Loss: 0.0519 | Avg Entropy: 1.6086
Episode 18: Reward = -118.95, Captures = 0, Avg Reward = -127.18, Avg Captures = 0.00
Actor Loss: -0.0165 | Critic Loss: 0.0555 | Avg Entropy: 1.6086
Episode 19: Reward = -117.86, Captures = 0, Avg Reward = -126.72, Avg Captures = 0.00
Actor Loss: -0.0175 | Critic Loss: 0.0448 | Avg Entropy: 1.6086
Episode 20: Reward = -148.68, Captures = 0, Avg Reward = -127.76, Avg Captures = 0.00
Actor Loss: -0.0160 | Critic Loss: 0.0433 | Avg Entropy: 1.6086
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 21: Reward = -111.50, Captures = 0, Avg Reward = -127.02, Avg Captures = 0.00
Actor Loss: -0.0169 | Critic Loss: 0.0430 | Avg Entropy: 1.6086
Episode 22: Reward = -141.88, Captures = 0, Avg Reward = -127.67, Avg Captures = 0.00
Actor Loss: -0.0159 | Critic Loss: 0.0384 | Avg Entropy: 1.6086
Episode 23: Reward = -141.98, Captures = 0, Avg Reward = -128.27, Avg Captures = 0.00
Actor Loss: -0.0170 | Critic Loss: 0.0294 | Avg Entropy: 1.6086
Episode 24: Reward = -136.10, Captures = 0, Avg Reward = -128.58, Avg Captures = 0.00
Actor Loss: -0.0160 | Critic Loss: 0.0316 | Avg Entropy: 1.6086
Episode 25: Reward = -113.64, Captures = 0, Avg Reward = -128.00, Avg Captures = 0.00
Actor Loss: -0.0179 | Critic Loss: 0.0453 | Avg Entropy: 1.6086
Episode 26: Reward = -69.11, Captures = 0, Avg Reward = -125.82, Avg Captures = 0.00
Actor Loss: -0.0172 | Critic Loss: 0.0737 | Avg Entropy: 1.6086
Episode 27: Reward = -125.86, Captures = 0, Avg Reward = -125.82, Avg Captures = 0.00
Actor Loss: -0.0154 | Critic Loss: 0.0458 | Avg Entropy: 1.6086
Episode 28: Reward = -114.83, Captures = 0, Avg Reward = -125.45, Avg Captures = 0.00
Actor Loss: -0.0158 | Critic Loss: 0.0589 | Avg Entropy: 1.6085
Episode 29: Reward = -141.32, Captures = 0, Avg Reward = -125.97, Avg Captures = 0.00
Actor Loss: -0.0189 | Critic Loss: 0.0324 | Avg Entropy: 1.6085
Episode 30: Reward = -139.23, Captures = 0, Avg Reward = -126.40, Avg Captures = 0.00
Actor Loss: -0.0166 | Critic Loss: 0.0327 | Avg Entropy: 1.6085
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 31: Reward = -176.79, Captures = 0, Avg Reward = -127.98, Avg Captures = 0.00
Actor Loss: -0.0150 | Critic Loss: 0.0383 | Avg Entropy: 1.6084
Episode 32: Reward = -121.66, Captures = 0, Avg Reward = -127.79, Avg Captures = 0.00
Actor Loss: -0.0169 | Critic Loss: 0.0380 | Avg Entropy: 1.6084
Episode 33: Reward = -118.52, Captures = 0, Avg Reward = -127.51, Avg Captures = 0.00
Actor Loss: -0.0176 | Critic Loss: 0.0560 | Avg Entropy: 1.6083
Episode 34: Reward = -117.43, Captures = 0, Avg Reward = -127.23, Avg Captures = 0.00
Actor Loss: -0.0206 | Critic Loss: 0.0485 | Avg Entropy: 1.6083
Episode 35: Reward = -128.28, Captures = 0, Avg Reward = -127.25, Avg Captures = 0.00
Actor Loss: -0.0182 | Critic Loss: 0.0354 | Avg Entropy: 1.6082
Episode 36: Reward = -106.43, Captures = 0, Avg Reward = -126.69, Avg Captures = 0.00
Actor Loss: -0.0201 | Critic Loss: 0.0605 | Avg Entropy: 1.6080
Episode 37: Reward = -77.97, Captures = 0, Avg Reward = -125.41, Avg Captures = 0.00
Actor Loss: -0.0170 | Critic Loss: 0.0656 | Avg Entropy: 1.6080
Episode 38: Reward = -139.04, Captures = 0, Avg Reward = -125.76, Avg Captures = 0.00
Actor Loss: -0.0175 | Critic Loss: 0.0356 | Avg Entropy: 1.6080
Episode 39: Reward = -115.49, Captures = 0, Avg Reward = -125.50, Avg Captures = 0.00
Actor Loss: -0.0201 | Critic Loss: 0.0476 | Avg Entropy: 1.6078
Episode 40: Reward = -109.83, Captures = 0, Avg Reward = -125.12, Avg Captures = 0.00
Actor Loss: -0.0164 | Critic Loss: 0.0586 | Avg Entropy: 1.6078
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 41: Reward = -118.95, Captures = 0, Avg Reward = -124.97, Avg Captures = 0.00
Actor Loss: -0.0183 | Critic Loss: 0.0338 | Avg Entropy: 1.6076
Episode 42: Reward = -71.75, Captures = 0, Avg Reward = -123.74, Avg Captures = 0.00
Actor Loss: -0.0167 | Critic Loss: 0.0718 | Avg Entropy: 1.6079
Episode 43: Reward = -149.73, Captures = 0, Avg Reward = -124.33, Avg Captures = 0.00
Actor Loss: -0.0191 | Critic Loss: 0.0313 | Avg Entropy: 1.6079
Episode 44: Reward = -141.59, Captures = 0, Avg Reward = -124.71, Avg Captures = 0.00
Actor Loss: -0.0157 | Critic Loss: 0.0321 | Avg Entropy: 1.6080
Episode 45: Reward = -137.33, Captures = 0, Avg Reward = -124.98, Avg Captures = 0.00
Actor Loss: -0.0177 | Critic Loss: 0.0346 | Avg Entropy: 1.6080
Episode 46: Reward = -136.45, Captures = 0, Avg Reward = -125.23, Avg Captures = 0.00
Actor Loss: -0.0167 | Critic Loss: 0.0393 | Avg Entropy: 1.6081
Episode 47: Reward = -136.99, Captures = 0, Avg Reward = -125.47, Avg Captures = 0.00
Actor Loss: -0.0190 | Critic Loss: 0.0306 | Avg Entropy: 1.6080
Episode 48: Reward = -143.60, Captures = 0, Avg Reward = -125.84, Avg Captures = 0.00
Actor Loss: -0.0173 | Critic Loss: 0.0284 | Avg Entropy: 1.6079
Episode 49: Reward = -121.92, Captures = 0, Avg Reward = -125.76, Avg Captures = 0.00
Actor Loss: -0.0170 | Critic Loss: 0.0450 | Avg Entropy: 1.6080
Episode 50: Reward = -111.40, Captures = 0, Avg Reward = -125.48, Avg Captures = 0.00
Actor Loss: -0.0196 | Critic Loss: 0.0332 | Avg Entropy: 1.6079

reward evaluate team play

Episode 10: Reward = 372.71, Captures = 0, Avg Reward = 372.71, Avg Captures = 0.00
Actor Loss: -0.0156 | Critic Loss: 0.2885 | Avg Entropy: 1.6090
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 11: Reward = 433.66, Captures = 0, Avg Reward = 403.19, Avg Captures = 0.00
Actor Loss: -0.0162 | Critic Loss: 0.4005 | Avg Entropy: 1.6090
Episode 12: Reward = 388.98, Captures = 0, Avg Reward = 398.45, Avg Captures = 0.00
Actor Loss: -0.0150 | Critic Loss: 0.2632 | Avg Entropy: 1.6090
Episode 13: Reward = 421.15, Captures = 0, Avg Reward = 404.13, Avg Captures = 0.00
Actor Loss: -0.0150 | Critic Loss: 0.3387 | Avg Entropy: 1.6090
Episode 14: Reward = 407.61, Captures = 0, Avg Reward = 404.82, Avg Captures = 0.00
Actor Loss: -0.0161 | Critic Loss: 0.2681 | Avg Entropy: 1.6090
Episode 15: Reward = 459.86, Captures = 0, Avg Reward = 413.99, Avg Captures = 0.00
Actor Loss: -0.0169 | Critic Loss: 0.3423 | Avg Entropy: 1.6090
Episode 16: Reward = 419.74, Captures = 0, Avg Reward = 414.82, Avg Captures = 0.00
Actor Loss: -0.0164 | Critic Loss: 0.2845 | Avg Entropy: 1.6090
Episode 17: Reward = 452.44, Captures = 0, Avg Reward = 419.52, Avg Captures = 0.00
Actor Loss: -0.0169 | Critic Loss: 0.3028 | Avg Entropy: 1.6090
Episode 18: Reward = 401.61, Captures = 0, Avg Reward = 417.53, Avg Captures = 0.00
Actor Loss: -0.0171 | Critic Loss: 0.2398 | Avg Entropy: 1.6090
Episode 19: Reward = 444.40, Captures = 0, Avg Reward = 420.22, Avg Captures = 0.00
Actor Loss: -0.0160 | Critic Loss: 0.2387 | Avg Entropy: 1.6090
Episode 20: Reward = 397.86, Captures = 0, Avg Reward = 418.18, Avg Captures = 0.00
Actor Loss: -0.0172 | Critic Loss: 0.1992 | Avg Entropy: 1.6090
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 21: Reward = 409.27, Captures = 0, Avg Reward = 417.44, Avg Captures = 0.00
Actor Loss: -0.0176 | Critic Loss: 0.1969 | Avg Entropy: 1.6089
Episode 22: Reward = 420.36, Captures = 0, Avg Reward = 417.67, Avg Captures = 0.00
Actor Loss: -0.0167 | Critic Loss: 0.2046 | Avg Entropy: 1.6089
Episode 23: Reward = 403.41, Captures = 0, Avg Reward = 416.65, Avg Captures = 0.00
Actor Loss: -0.0174 | Critic Loss: 0.1649 | Avg Entropy: 1.6089
Episode 24: Reward = 412.87, Captures = 0, Avg Reward = 416.40, Avg Captures = 0.00
Actor Loss: -0.0193 | Critic Loss: 0.1705 | Avg Entropy: 1.6088
Episode 25: Reward = 394.27, Captures = 0, Avg Reward = 415.01, Avg Captures = 0.00
Actor Loss: -0.0204 | Critic Loss: 0.1373 | Avg Entropy: 1.6087
Episode 26: Reward = 420.40, Captures = 0, Avg Reward = 415.33, Avg Captures = 0.00
Actor Loss: -0.0192 | Critic Loss: 0.1375 | Avg Entropy: 1.6086
Episode 27: Reward = 383.56, Captures = 0, Avg Reward = 413.56, Avg Captures = 0.00
Actor Loss: -0.0212 | Critic Loss: 0.1177 | Avg Entropy: 1.6085
Episode 28: Reward = 472.43, Captures = 0, Avg Reward = 416.66, Avg Captures = 0.00
Actor Loss: -0.0235 | Critic Loss: 0.1513 | Avg Entropy: 1.6084
Episode 29: Reward = 385.57, Captures = 0, Avg Reward = 415.11, Avg Captures = 0.00
Actor Loss: -0.0247 | Critic Loss: 0.1289 | Avg Entropy: 1.6081
Episode 30: Reward = 488.29, Captures = 0, Avg Reward = 418.59, Avg Captures = 0.00
Actor Loss: -0.0199 | Critic Loss: 0.1280 | Avg Entropy: 1.6080
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 31: Reward = 554.19, Captures = 0, Avg Reward = 424.76, Avg Captures = 0.00
Actor Loss: -0.0224 | Critic Loss: 0.2417 | Avg Entropy: 1.6075
Episode 32: Reward = 415.82, Captures = 0, Avg Reward = 424.37, Avg Captures = 0.00
Actor Loss: -0.0243 | Critic Loss: 0.1457 | Avg Entropy: 1.6071
Episode 33: Reward = 437.09, Captures = 0, Avg Reward = 424.90, Avg Captures = 0.00
Actor Loss: -0.0230 | Critic Loss: 0.1556 | Avg Entropy: 1.6070
Episode 34: Reward = 400.89, Captures = 0, Avg Reward = 423.94, Avg Captures = 0.00
Actor Loss: -0.0267 | Critic Loss: 0.1740 | Avg Entropy: 1.6066
Episode 35: Reward = 401.85, Captures = 0, Avg Reward = 423.09, Avg Captures = 0.00
Actor Loss: -0.0268 | Critic Loss: 0.1140 | Avg Entropy: 1.6067
Episode 36: Reward = 372.28, Captures = 0, Avg Reward = 421.21, Avg Captures = 0.00
Actor Loss: -0.0218 | Critic Loss: 0.1351 | Avg Entropy: 1.6065
Episode 37: Reward = 463.51, Captures = 0, Avg Reward = 422.72, Avg Captures = 0.00
Actor Loss: -0.0202 | Critic Loss: 0.1646 | Avg Entropy: 1.6062
Episode 38: Reward = 450.87, Captures = 0, Avg Reward = 423.69, Avg Captures = 0.00
Actor Loss: -0.0246 | Critic Loss: 0.1203 | Avg Entropy: 1.6067
Episode 39: Reward = 567.83, Captures = 0, Avg Reward = 428.49, Avg Captures = 0.00
Actor Loss: -0.0260 | Critic Loss: 0.2477 | Avg Entropy: 1.6066
Episode 40: Reward = 417.55, Captures = 0, Avg Reward = 428.14, Avg Captures = 0.00
Actor Loss: -0.0234 | Critic Loss: 0.1536 | Avg Entropy: 1.6067
Checkpoint saved: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth
Episode 41: Reward = 364.30, Captures = 0, Avg Reward = 426.15, Avg Captures = 0.00
Actor Loss: -0.0290 | Critic Loss: 0.1175 | Avg Entropy: 1.6069
Episode 42: Reward = 658.34, Captures = 0, Avg Reward = 433.18, Avg Captures = 0.00
Actor Loss: -0.0238 | Critic Loss: 0.3344 | Avg Entropy: 1.6068
Episode 43: Reward = 519.16, Pooling = 0, Avg Reward = 435.71, Avg Captures = 0.00
Actor Loss: -0.0285 | Critic Loss: 0.1981 | Avg Entropy: 1.6069
Episode 44: Reward = 459.94, Captures = 0, Avg Reward = 436.40, Avg Captures = 0.00
Actor Loss: -0.0226 | Critic Loss: 0.1403 | Avg Entropy: 1.6072
Episode 45: Reward = 610.95, Captures = 0, Avg Reward = 441.25, Avg Captures = 0.00
Actor Loss: -0.0280 | Critic Loss: 0.2423 | Avg Entropy: 1.6069
Episode 46: Reward = 486.87, Captures = 0, Avg Reward = 442.48, Avg Captures = 0.00
Actor Loss: -0.0298 | Critic Loss: 0.1489 | Avg Entropy: 1.6075
Episode 47: Reward = 386.94, Captures = 0, Avg Reward = 441.02, Avg Captures = 0.00
Actor Loss: -0.0254 | Critic Loss: 0.1293 | Avg Entropy: 1.6072
Episode 48: Reward = 561.70, Captures = 0, Avg Reward = 444.12, Avg Captures = 0.00
Actor Loss: -0.0272 | Critic Loss: 0.2120 | Avg Entropy: 1.6068
Episode 49: Reward = 619.12, Captures = 0, Avg Reward = 448.49, Avg Captures = 0.00
Actor Loss: -0.0297 | Critic Loss: 0.2478 | Avg Entropy: 1.6073
Episode 50: Reward = 431.19, Captures = 0, Avg Reward = 448.07, Avg Captures = 0.00
Actor Loss: -0.0308 | Critic Loss: 0.1478 | Avg Entropy: 1.6073
"""

# ---------------------------------------------------------
# 2. 正規表現を用いたデータの抽出
# ---------------------------------------------------------
# 各フェーズ（Gelu変更後、Team Play評価）に分けてデータを保持
phases = {"Gelu_Only": {}, "Team_Play": {}}
current_phase = "Gelu_Only"

# 正規表現パターン
ep_pattern = re.compile(r"Episode\s+(\d+):\s+Reward\s+=\s+([-\d.]+),\s+Captures.*Avg\s+Reward\s+=\s+([-\d.]+)")
loss_pattern = re.compile(r"Actor\s+Loss:\s+([-\d.]+)\s+\|\s+Critic\s+Loss:\s+([-\d.]+)\s+\|\s+Avg\s+Entropy:\s+([-\d.]+)")

lines = log_data.strip().split('\n')
for line in lines:
    if "reward evaluate team play" in line:
        current_phase = "Team_Play"
        continue
        
    ep_match = ep_pattern.search(line)
    if ep_match:
        ep = int(ep_match.group(1))
        rew = float(ep_match.group(2))
        avg_rew = float(ep_match.group(3))
        
        if ep not in phases[current_phase]:
            phases[current_phase][ep] = {}
        phases[current_phase][ep]['reward'] = rew
        phases[current_phase][ep]['avg_reward'] = avg_rew
        
    loss_match = loss_pattern.search(line)
    if loss_match:
        # 直前のEpisodeマッチと紐付けるため、現在のフェーズの最大（最新）のepを取得
        if phases[current_phase]:
            ep = max(phases[current_phase].keys())
            phases[current_phase][ep]['actor_loss'] = float(loss_match.group(1))
            phases[current_phase][ep]['critic_loss'] = float(loss_match.group(2))
            phases[current_phase][ep]['entropy'] = float(loss_match.group(3))

# ---------------------------------------------------------
# 3. グラフ描画用のリスト作成
# ---------------------------------------------------------
def dict_to_lists(d):
    # エピソード順にソートしてリスト化
    sorted_eps = sorted([k for k, v in d.items() if 'actor_loss' in v]) # 完全にデータが揃っているもの
    return {
        'ep': sorted_eps,
        'reward': [d[e]['reward'] for e in sorted_eps],
        'avg_reward': [d[e]['avg_reward'] for e in sorted_eps],
        'actor_loss': [d[e]['actor_loss'] for e in sorted_eps],
        'critic_loss': [d[e]['critic_loss'] for e in sorted_eps],
        'entropy': [d[e]['entropy'] for e in sorted_eps],
    }

gelu_data = dict_to_lists(phases["Gelu_Only"])
team_data = dict_to_lists(phases["Team_Play"])

# ---------------------------------------------------------
# 4. Matplotlibによる可視化
# ---------------------------------------------------------
fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex='col')
fig.suptitle('Training Metrics Comparison: Gelu Only vs Team Play', fontsize=16, fontweight='bold')

# --- 左列: Gelu Only フェーズ ---
# Reward
axs[0, 0].plot(gelu_data['ep'], gelu_data['reward'], label='Reward', color='lightblue', alpha=0.6, linestyle='--')
axs[0, 0].plot(gelu_data['ep'], gelu_data['avg_reward'], label='Avg Reward', color='blue', linewidth=2)
axs[0, 0].set_title('Hybrid Reward: Reward')
axs[0, 0].grid(True, linestyle=':')
axs[0, 0].legend()

# Losses
axs[1, 0].plot(gelu_data['ep'], gelu_data['actor_loss'], label='Actor Loss', color='darkorange')
axs[1, 0].plot(gelu_data['ep'], gelu_data['critic_loss'], label='Critic Loss', color='red')
axs[1, 0].set_title('Hybrid Reward: Loss')
axs[1, 0].grid(True, linestyle=':')
axs[1, 0].legend()

# Entropy
axs[2, 0].plot(gelu_data['ep'], gelu_data['entropy'], color='purple', marker='o', markersize=3)
axs[2, 0].set_title('Hybrid Reward: Avg Entropy')
axs[2, 0].set_xlabel('Episode')
axs[2, 0].grid(True, linestyle=':')


# --- 右列: Team Play フェーズ ---
# Reward
axs[0, 1].plot(team_data['ep'], team_data['reward'], label='Reward', color='lightgreen', alpha=0.6, linestyle='--')
axs[0, 1].plot(team_data['ep'], team_data['avg_reward'], label='Avg Reward', color='green', linewidth=2)
axs[0, 1].set_title('Team Play: Reward')
axs[0, 1].grid(True, linestyle=':')
axs[0, 1].legend()

# Losses
axs[1, 1].plot(team_data['ep'], team_data['actor_loss'], label='Actor Loss', color='darkorange')
axs[1, 1].plot(team_data['ep'], team_data['critic_loss'], label='Critic Loss', color='red')
axs[1, 1].set_title('Team Play: Loss')
axs[1, 1].grid(True, linestyle=':')
axs[1, 1].legend()

# Entropy
axs[2, 1].plot(team_data['ep'], team_data['entropy'], color='purple', marker='o', markersize=3)
axs[2, 1].set_title('Team Play: Avg Entropy')
axs[2, 1].set_xlabel('Episode')
axs[2, 1].grid(True, linestyle=':')

plt.tight_layout()
plt.show()