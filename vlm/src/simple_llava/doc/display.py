import re
import matplotlib.pyplot as plt

# =====================================================================
# 1. ログデータの定義
# =====================================================================
mappo_log = """
Episode 110: Reward = 77.56, Captures = 8, Avg Reward = 77.56, Avg Captures = 8.00
Actor Loss: -0.0195 | Critic Loss: 0.4405 | Avg Entropy: 1.6042
💾 チェックポイントを保存しました: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth (Episode 110)
Episode 111: Reward = 125.10, Captures = 20, Avg Reward = 101.33, Avg Captures = 14.00
Actor Loss: -0.0182 | Critic Loss: 0.4253 | Avg Entropy: 1.6050
Episode 112: Reward = 77.02, Captures = 5, Avg Reward = 93.23, Avg Captures = 11.00
Actor Loss: -0.0239 | Critic Loss: 0.3923 | Avg Entropy: 1.6054
Episode 113: Reward = 87.08, Captures = 9, Avg Reward = 91.69, Avg Captures = 10.50
Actor Loss: -0.0213 | Critic Loss: 0.3429 | Avg Entropy: 1.6057
Episode 114: Reward = 74.46, Captures = 7, Avg Reward = 88.24, Avg Captures = 9.80
Actor Loss: -0.0199 | Critic Loss: 0.3045 | Avg Entropy: 1.6057
Episode 115: Reward = 73.03, Captures = 8, Avg Reward = 85.71, Avg Captures = 9.50
Actor Loss: -0.0254 | Critic Loss: 0.2739 | Avg Entropy: 1.6056
Episode 116: Reward = 138.32, Captures = 21, Avg Reward = 93.22, Avg Captures = 11.14
Actor Loss: -0.0182 | Critic Loss: 0.2328 | Avg Entropy: 1.6051
Episode 117: Reward = 68.30, Captures = 6, Avg Reward = 90.11, Avg Captures = 10.50
Actor Loss: -0.0284 | Critic Loss: 0.1840 | Avg Entropy: 1.6043
Episode 118: Reward = 125.08, Captures = 15, Avg Reward = 93.99, Avg Captures = 11.00
Actor Loss: -0.0287 | Critic Loss: 0.1585 | Avg Entropy: 1.6050
Episode 119: Reward = 53.15, Captures = 4, Avg Reward = 89.91, Avg Captures = 10.30
Actor Loss: -0.0257 | Critic Loss: 0.1032 | Avg Entropy: 1.6054
Episode 120: Reward = 120.64, Captures = 18, Avg Reward = 92.70, Avg Captures = 11.00
Actor Loss: -0.0148 | Critic Loss: 0.1163 | Avg Entropy: 1.5055
💾 チェックポイントを保存しました: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth (Episode 120)
Episode 121: Reward = 74.65, Captures = 8, Avg Reward = 91.20, Avg Captures = 10.75
Actor Loss: -0.0253 | Critic Loss: 0.0571 | Avg Entropy: 1.6052
Episode 122: Reward = 61.11, Captures = 5, Avg Reward = 88.88, Avg Captures = 10.31
Actor Loss: -0.0300 | Critic Loss: 0.0385 | Avg Entropy: 1.6046
Episode 123: Reward = 70.68, Captures = 6, Avg Reward = 87.58, Avg Captures = 10.00
Actor Loss: -0.0291 | Critic Loss: 0.0369 | Avg Entropy: 1.6043
Episode 124: Reward = 64.69, Captures = 6, Avg Reward = 86.06, Avg Captures = 9.73
Actor Loss: -0.0317 | Critic Loss: 0.0315 | Avg Entropy: 1.6042
Episode 125: Reward = 61.29, Captures = 5, Avg Reward = 84.51, Avg Captures = 9.44
Actor Loss: -0.0328 | Critic Loss: 0.0286 | Avg Entropy: 1.6039
Episode 126: Reward = 81.39, Captures = 9, Avg Reward = 84.33, Avg Captures = 9.41
Actor Loss: -0.0293 | Critic Loss: 0.0479 | Avg Entropy: 1.6038
Episode 127: Reward = 85.70, Captures = 9, Avg Reward = 84.40, Avg Captures = 9.39
Actor Loss: -0.0237 | Critic Loss: 0.0549 | Avg Entropy: 1.6039
Episode 128: Reward = 68.96, Captures = 6, Avg Reward = 83.59, Avg Captures = 9.21
Actor Loss: -0.0306 | Critic Loss: 0.0195 | Avg Entropy: 1.6036
Episode 129: Reward = 138.38, Captures = 17, Avg Reward = 86.33, Avg Captures = 9.60
Actor Loss: -0.0273 | Critic Loss: 0.1167 | Avg Entropy: 1.6026
Episode 130: Reward = 70.94, Captures = 6, Avg Reward = 85.60, Avg Captures = 9.43
Actor Loss: -0.0300 | Critic Loss: 0.0166 | Avg Entropy: 1.6022
💾 チェックポイントを保存しました: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth (Episode 130)
Episode 131: Reward = 138.70, Captures = 16, Avg Reward = 88.01, Avg Captures = 9.73
Actor Loss: -0.0185 | Critic Loss: 0.1030 | Avg Entropy: 1.6005
Episode 132: Reward = 70.48, Captures = 6, Avg Reward = 87.25, Avg Captures = 9.57
Actor Loss: -0.0252 | Critic Loss: 0.0223 | Avg Entropy: 1.6004
Episode 133: Reward = 51.60, Captures = 3, Avg Reward = 85.76, Avg Captures = 9.29
Actor Loss: -0.0347 | Critic Loss: 0.0127 | Avg Entropy: 1.6011
Episode 134: Reward = 97.47, Captures = 10, Avg Reward = 86.23, Avg Captures = 9.32
Actor Loss: -0.0227 | Critic Loss: 0.0678 | Avg Entropy: 1.6010
Episode 135: Reward = 161.13, Captures = 21, Avg Reward = 89.11, Avg Captures = 9.77
Actor Loss: -0.0160 | Critic Loss: 0.1108 | Avg Entropy: 1.5999
Episode 136: Reward = 66.25, Captures = 6, Avg Reward = 88.26, Avg Captures = 9.63
Actor Loss: -0.0250 | Critic Loss: 0.0250 | Avg Entropy: 1.5992
Episode 137: Reward = 67.91, Captures = 4, Avg Reward = 87.54, Avg Captures = 9.43
Actor Loss: -0.0328 | Critic Loss: 0.0230 | Avg Entropy: 1.5990
Episode 138: Reward = 139.80, Captures = 17, Avg Reward = 89.34, Avg Captures = 9.69
Actor Loss: -0.0130 | Critic Loss: 0.1091 | Avg Entropy: 1.5969
Episode 139: Reward = 93.71, Captures = 10, Avg Reward = 89.49, Avg Captures = 9.70
Actor Loss: -0.0276 | Critic Loss: 0.0447 | Avg Entropy: 1.5974
Episode 140: Reward = 73.89, Captures = 7, Avg Reward = 88.98, Avg Captures = 9.61
Actor Loss: -0.0220 | Critic Loss: 0.0389 | Avg Entropy: 1.5966
💾 チェックポイントを保存しました: /content/drive/MyDrive/rl_pursuit/mappo_episode.pth (Episode 140)
Episode 141: Reward = 126.91, Captures = 18, Avg Reward = 90.17, Avg Captures = 9.88
Actor Loss: -0.0283 | Critic Loss: 0.0724 | Avg Entropy: 1.5950
Episode 142: Reward = 99.14, Captures = 11, Avg Reward = 90.44, Avg Captures = 9.91
Actor Loss: -0.0274 | Critic Loss: 0.0458 | Avg Entropy: 1.5931
Episode 143: Reward = 88.59, Captures = 9, Avg Reward = 90.39, Avg Captures = 9.88
Actor Loss: -0.0158 | Critic Loss: 0.0322 | Avg Entropy: 1.5929
Episode 144: Reward = 169.21, Captures = 21, Avg Reward = 92.64, Avg Captures = 10.20
Actor Loss: -0.0034 | Critic Loss: 0.1129 | Avg Entropy: 1.5915
Episode 145: Reward = 102.39, Captures = 11, Avg Reward = 92.91, Avg Captures = 10.22
Actor Loss: -0.0165 | Critic Loss: 0.0504 | Avg Entropy: 1.5906
Episode 146: Reward = 115.00, Captures = 13, Avg Reward = 93.51, Avg Captures = 10.30
Actor Loss: -0.0236 | Critic Loss: 0.0676 | Avg Entropy: 1.5901
Episode 147: Reward = 257.18, Captures = 29, Avg Reward = 97.81, Avg Captures = 10.79
Actor Loss: -0.0180 | Critic Loss: 0.2259 | Avg Entropy: 1.5872
Episode 148: Reward = 86.43, Captures = 8, Avg Reward = 97.52, Avg Captures = 10.72
Actor Loss: -0.0208 | Critic Loss: 0.0365 | Avg Entropy: 1.5863
Episode 149: Reward = 128.48, Captures = 14, Avg Reward = 98.29, Avg Captures = 10.80
Actor Loss: -0.0140 | Critic Loss: 0.0601 | Avg Entropy: 1.5853
Episode 150: Reward = 60.22, Captures = 6, Avg Reward = 97.37, Avg Captures = 10.68
Actor Loss: -0.0238 | Critic Loss: 0.0453 | Avg Entropy: 1.5831
"""

mat_log = """
update=1/1000 actor_loss=-0.0078 critic_loss=0.2344 entropy=1.4653
[episode 1] reward_sum=188.59 mean_reward=23.573 captures=31
update=2/1000 actor_loss=-0.0107 critic_loss=0.1990 entropy=1.4769
[episode 2] reward_sum=178.28 mean_reward=22.285 captures=29
update=3/1000 actor_loss=-0.0094 critic_loss=0.2919 entropy=1.4764
[episode 3] reward_sum=240.47 mean_reward=30.059 captures=31
update=4/1000 actor_loss=-0.0086 critic_loss=0.3692 entropy=1.4717
[episode 4] reward_sum=261.96 mean_reward=32.745 captures=34
update=5/1000 actor_loss=-0.0105 critic_loss=0.3509 entropy=1.4814
[episode 5] reward_sum=270.88 mean_reward=33.859 captures=34
update=6/1000 actor_loss=-0.0107 critic_loss=0.3405 entropy=1.4757
[episode 6] reward_sum=271.66 mean_reward=33.957 captures=31
update=7/1000 actor_loss=-0.0117 critic_loss=0.2905 entropy=1.4738
[episode 7] reward_sum=250.40 mean_reward=31.300 captures=36
update=8/1000 actor_loss=-0.0096 critic_loss=0.2518 entropy=1.4820
[episode 8] reward_sum=187.56 mean_reward=23.445 captures=34
update=9/1000 actor_loss=-0.0108 critic_loss=0.3792 entropy=1.4918
[episode 9] reward_sum=292.69 mean_reward=36.586 captures=33
update=10/1000 actor_loss=-0.0127 critic_loss=0.2428 entropy=1.5012
[episode 10] reward_sum=203.54 mean_reward=25.443 captures=27
update=11/1000 actor_loss=-0.0111 critic_loss=0.2271 entropy=1.4921
[episode 11] reward_sum=198.92 mean_reward=24.866 captures=29
update=12/1000 actor_loss=-0.0120 critic_loss=0.2699 entropy=1.4859
[episode 12] reward_sum=197.44 mean_reward=24.680 captures=30
update=13/1000 actor_loss=-0.0082 critic_loss=0.2619 entropy=1.4733
[episode 13] reward_sum=230.52 mean_reward=28.815 captures=29
update=14/1000 actor_loss=-0.0094 critic_loss=0.2760 entropy=1.4857
[episode 14] reward_sum=214.77 mean_reward=26.847 captures=38
update=15/1000 actor_loss=-0.0112 critic_loss=0.3479 entropy=1.4842
[episode 15] reward_sum=259.27 mean_reward=32.409 captures=39
update=16/1000 actor_loss=-0.0101 critic_loss=0.2600 entropy=1.5027
[episode 16] reward_sum=221.08 mean_reward=27.635 captures=34
update=17/1000 actor_loss=-0.0101 critic_loss=0.2227 entropy=1.4983
[episode 17] reward_sum=191.87 mean_reward=23.983 captures=29
update=18/1000 actor_loss=-0.0118 critic_loss=0.2030 entropy=1.4991
[episode 18] reward_sum=187.96 mean_reward=23.495 captures=25
update=19/1000 actor_loss=-0.0121 critic_loss=0.2482 entropy=1.4972
[episode 19] reward_sum=199.57 mean_reward=24.946 captures=25
update=20/1000 actor_loss=-0.0115 critic_loss=0.2098 entropy=1.4943
[episode 20] reward_sum=205.41 mean_reward=25.677 captures=25
update=21/1000 actor_loss=-0.0141 critic_loss=0.1636 entropy=1.4937
[episode 21] reward_sum=167.62 mean_reward=20.953 captures=26
update=22/1000 actor_loss=-0.0143 critic_loss=0.3473 entropy=1.4909
[episode 22] reward_sum=289.19 mean_reward=36.149 captures=29
update=23/1000 actor_loss=-0.0099 critic_loss=0.3670 entropy=1.4867
[episode 23] reward_sum=314.92 mean_reward=39.366 captures=38
update=24/1000 actor_loss=-0.0116 critic_loss=0.4661 entropy=1.4927
[episode 24] reward_sum=332.96 mean_reward=41.620 captures=34
update=25/1000 actor_loss=-0.0142 critic_loss=0.4783 entropy=1.4852
[episode 25] reward_sum=407.64 mean_reward=50.956 captures=39
update=26/1000 actor_loss=-0.0128 critic_loss=0.4075 entropy=1.4824
[episode 26] reward_sum=269.21 mean_reward=33.651 captures=29
update=27/1000 actor_loss=-0.0136 critic_loss=0.3699 entropy=1.4898
[episode 27] reward_sum=281.17 mean_reward=35.147 captures=35
update=28/1000 actor_loss=-0.0110 critic_loss=0.6121 entropy=1.5027
[episode 28] reward_sum=457.04 mean_reward=57.130 captures=40
update=29/1000 actor_loss=-0.0123 critic_loss=0.4213 entropy=1.5117
[episode 29] reward_sum=378.53 mean_reward=47.316 captures=37
update=30/1000 actor_loss=-0.0150 critic_loss=0.3587 entropy=1.5190
[episode 30] reward_sum=278.51 mean_reward=34.814 captures=38
update=31/1000 actor_loss=-0.0155 critic_loss=0.4204 entropy=1.5106
[episode 31] reward_sum=293.49 mean_reward=36.686 captures=37
update=32/1000 actor_loss=-0.0123 critic_loss=0.3509 entropy=1.5064
[episode 32] reward_sum=291.12 mean_reward=36.390 captures=38
update=33/1000 actor_loss=-0.0139 critic_loss=0.4952 entropy=1.5117
[episode 33] reward_sum=352.14 mean_reward=44.018 captures=38
update=34/1000 actor_loss=-0.0119 critic_loss=0.3523 entropy=1.5179
[episode 34] reward_sum=219.41 mean_reward=27.427 captures=35
update=35/1000 actor_loss=-0.0136 critic_loss=0.4574 entropy=1.5179
[episode 35] reward_sum=347.50 mean_reward=43.437 captures=37
update=36/1000 actor_loss=-0.0139 critic_loss=0.1886 entropy=1.5251
[episode 36] reward_sum=126.98 mean_reward=15.872 captures=20
update=37/1000 actor_loss=-0.0129 critic_loss=0.2740 entropy=1.5237
[episode 37] reward_sum=218.39 mean_reward=27.299 captures=32
update=38/1000 actor_loss=-0.0143 critic_loss=0.3288 entropy=1.5056
[episode 38] reward_sum=252.67 mean_reward=31.584 captures=34
update=39/1000 actor_loss=-0.0155 critic_loss=0.2828 entropy=1.5060
[episode 39] reward_sum=259.32 mean_reward=32.415 captures=29
update=40/1000 actor_loss=-0.0114 critic_loss=0.3161 entropy=1.5138
[episode 40] reward_sum=258.44 mean_reward=32.306 captures=29
update=41/1000 actor_loss=-0.0139 critic_loss=0.2560 entropy=1.5266
[episode 41] reward_sum=202.49 mean_reward=25.312 captures=28
update=42/1000 actor_loss=-0.0116 critic_loss=0.2467 entropy=1.5262
[episode 42] reward_sum=199.35 mean_reward=24.918 captures=31
update=43/1000 actor_loss=-0.0180 critic_loss=0.1465 entropy=1.5262
[episode 43] reward_sum=116.16 mean_reward=14.519 captures=15
update=44/1000 actor_loss=-0.0140 critic_loss=0.1957 entropy=1.5309
[episode 44] reward_sum=175.09 mean_reward=21.886 captures=26
update=45/1000 actor_loss=-0.0149 critic_loss=0.2613 entropy=1.5133
[episode 45] reward_sum=210.46 mean_reward=26.307 captures=32
update=46/1000 actor_loss=-0.0168 critic_loss=0.2434 entropy=1.5139
[episode 46] reward_sum=207.99 mean_reward=25.998 captures=31
update=47/1000 actor_loss=-0.0155 critic_loss=0.2532 entropy=1.5150
[episode 47] reward_sum=199.85 mean_reward=24.981 captures=30
update=48/1000 actor_loss=-0.0142 critic_loss=0.1449 entropy=1.5056
[episode 48] reward_sum=132.22 mean_reward=16.528 captures=20
update=49/1000 actor_loss=-0.0146 critic_loss=0.3037 entropy=1.4876
[episode 49] reward_sum=270.01 mean_reward=33.752 captures=33
update=50/1000 actor_loss=-0.0138 critic_loss=0.2507 entropy=1.4868
[episode 50] reward_sum=184.27 mean_reward=23.034 captures=27
update=51/1000 actor_loss=-0.0134 critic_loss=0.2651 entropy=1.4651
[episode 51] reward_sum=210.35 mean_reward=26.293 captures=29
"""

# =====================================================================
# 2. パース処理（正規表現による数値抽出）
# =====================================================================
# --- MAPPOパース ---
mappo_rewards = [float(x) for x in re.findall(r"Episode \d+: Reward = ([\d.]+)", mappo_log)]
mappo_entropies = [float(x) for x in re.findall(r"Avg Entropy: ([\d.]+)", mappo_log)]

# --- MATパース ---
mat_rewards = [float(x) for x in re.findall(r"reward_sum=([\d.]+)", mat_log)]
mat_entropies = [float(x) for x in re.findall(r"entropy=([\d.]+)", mat_log)]

# 開始位置を 0 (相対ステップ) に揃えるインデックス
mappo_steps = list(range(len(mappo_rewards)))
mat_steps = list(range(len(mat_rewards)))

# =====================================================================
# 3. グラフ描画
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左：報酬の比較グラフ ---
axes[0].plot(mappo_steps, mappo_rewards, label="MAPPO", color="#1f77b4", marker="o", markersize=3, alpha=0.8)
axes[0].plot(mat_steps, mat_rewards, label="MAT", color="#ff7f0e", marker="s", markersize=3, alpha=0.8)
axes[0].set_title("Reward Comparison (Aligned Start)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Relative Steps (from Start of Log)", fontsize=10)
axes[0].set_ylabel("Total Episode Reward", fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].legend(fontsize=10)

# --- 右：エントロピーの比較グラフ ---
axes[1].plot(mappo_steps, mappo_entropies, label="MAPPO", color="#1f77b4", marker="o", markersize=3, alpha=0.8)
axes[1].plot(mat_steps, mat_entropies, label="MAT", color="#ff7f0e", marker="s", markersize=3, alpha=0.8)
axes[1].set_title("Entropy Comparison (Aligned Start)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Relative Steps (from Start of Log)", fontsize=10)
axes[1].set_ylabel("Average Entropy", fontsize=10)
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.show()