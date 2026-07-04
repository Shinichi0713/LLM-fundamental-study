import re
import matplotlib.pyplot as plt
import pandas as pd

# 1. ログデータの定義
log_case1 = """
Episode 42: Reward = 467.37, Captures = 0, Avg Reward = 489.93, Avg Captures = 0.00
Actor Loss: -0.0172 | Critic Loss: 0.6201 | Avg Entropy: 1.6013
Episode 43: Reward = 481.78, Captures = 0, Avg Reward = 487.90, Avg Captures = 0.00
Actor Loss: -0.0141 | Critic Loss: 0.5737 | Avg Entropy: 1.6010
Episode 44: Reward = 594.01, Captures = 0, Avg Reward = 509.12, Avg Captures = 0.00
Actor Loss: -0.0121 | Critic Loss: 0.6822 | Avg Entropy: 1.6009
Episode 45: Reward = 471.01, Captures = 0, Avg Reward = 502.77, Avg Captures = 0.00
Actor Loss: -0.0188 | Critic Loss: 0.5120 | Avg Entropy: 1.6008
Episode 46: Reward = 484.04, Captures = 0, Avg Reward = 500.09, Avg Captures = 0.00
Actor Loss: -0.0132 | Critic Loss: 0.5283 | Avg Entropy: 1.6011
Episode 47: Reward = 529.22, Captures = 0, Avg Reward = 503.73, Avg Captures = 0.00
Actor Loss: -0.0148 | Critic Loss: 0.5817 | Avg Entropy: 1.6020
Episode 48: Reward = 395.21, Captures = 0, Avg Reward = 491.68, Avg Captures = 0.00
Actor Loss: -0.0140 | Critic Loss: 0.3669 | Avg Entropy: 1.6028
Episode 49: Reward = 383.41, Captures = 0, Avg Reward = 480.85, Avg Captures = 0.00
Actor Loss: -0.0196 | Critic Loss: 0.3435 | Avg Entropy: 1.6029
Episode 50: Reward = 413.43, Captures = 0, Avg Reward = 474.72, Avg Captures = 0.00
Actor Loss: -0.0137 | Critic Loss: 0.3308 | Avg Entropy: 1.6027
Episode 51: Reward = 464.33, Captures = 0, Avg Reward = 473.85, Avg Captures = 0.00
Actor Loss: -0.0169 | Critic Loss: 0.3864 | Avg Entropy: 1.6019
Episode 52: Reward = 616.90, Captures = 0, Avg Reward = 484.86, Avg Captures = 0.00
Actor Loss: -0.0192 | Critic Loss: 0.5948 | Avg Entropy: 1.6015
Episode 53: Reward = 676.31, Captures = 0, Avg Reward = 498.53, Avg Captures = 0.00
Actor Loss: -0.0163 | Critic Loss: 0.6437 | Avg Entropy: 1.6017
Episode 54: Reward = 507.19, Captures = 0, Avg Reward = 499.11, Avg Captures = 0.00
Actor Loss: -0.0147 | Critic Loss: 0.3696 | Avg Entropy: 1.6016
Episode 55: Reward = 503.13, Captures = 0, Avg Reward = 499.36, Avg Captures = 0.00
Actor Loss: -0.0154 | Critic Loss: 0.3156 | Avg Entropy: 1.6012
Episode 56: Reward = 368.84, Captures = 0, Avg Reward = 491.68, Avg Captures = 0.00
Actor Loss: -0.0122 | Critic Loss: 0.1700 | Avg Entropy: 1.6011
Episode 57: Reward = 532.65, Captures = 0, Avg Reward = 493.96, Avg Captures = 0.00
Actor Loss: -0.0165 | Critic Loss: 0.3028 | Avg Entropy: 1.6011
Episode 58: Reward = 468.54, Captures = 0, Avg Reward = 492.62, Avg Captures = 0.00
Actor Loss: -0.0164 | Critic Loss: 0.1974 | Avg Entropy: 1.6011
Episode 59: Reward = 526.99, Captures = 0, Avg Reward = 494.34, Avg Captures = 0.00
Actor Loss: -0.0198 | Critic Loss: 0.2353 | Avg Entropy: 1.6013
Episode 60: Reward = 641.95, Captures = 0, Avg Reward = 501.37, Avg Captures = 0.00
Actor Loss: -0.0219 | Critic Loss: 0.4083 | Avg Entropy: 1.6021
Episode 61: Reward = 559.82, Captures = 0, Avg Reward = 504.03, Avg Captures = 0.00
Actor Loss: -0.0132 | Critic Loss: 0.2081 | Avg Entropy: 1.6033
Episode 62: Reward = 624.36, Captures = 0, Avg Reward = 509.26, Avg Captures = 0.00
Actor Loss: -0.0168 | Critic Loss: 0.2913 | Avg Entropy: 1.6038
Episode 63: Reward = 495.37, Captures = 0, Avg Reward = 508.68, Avg Captures = 0.00
Actor Loss: -0.0195 | Critic Loss: 0.2081 | Avg Entropy: 1.6039
Episode 64: Reward = 531.21, Captures = 0, Avg Reward = 509.58, Avg Captures = 0.00
Actor Loss: -0.0176 | Critic Loss: 0.2721 | Avg Entropy: 1.6033
Episode 65: Reward = 474.34, Captures = 0, Avg Reward = 508.23, Avg Captures = 0.00
Actor Loss: -0.0130 | Critic Loss: 0.1952 | Avg Entropy: 1.6031
Episode 66: Reward = 418.90, Captures = 0, Avg Reward = 504.92, Avg Captures = 0.00
Actor Loss: -0.0136 | Critic Loss: 0.1979 | Avg Entropy: 1.6028
Episode 67: Reward = 453.41, Captures = 0, Avg Reward = 503.08, Avg Captures = 0.00
Actor Loss: -0.0195 | Critic Loss: 0.1530 | Avg Entropy: 1.6033
Episode 68: Reward = 445.16, Captures = 0, Avg Reward = 501.08, Avg Captures = 0.00
Actor Loss: -0.0165 | Critic Loss: 0.1520 | Avg Entropy: 1.6027
Episode 69: Reward = 498.93, Captures = 0, Avg Reward = 501.01, Avg Captures = 0.00
Actor Loss: -0.0140 | Critic Loss: 0.1843 | Avg Entropy: 1.6020
Episode 70: Reward = 729.47, Captures = 0, Avg Reward = 508.38, Avg Captures = 0.00
Actor Loss: -0.0190 | Critic Loss: 0.4764 | Avg Entropy: 1.6023
Episode 71: Reward = 636.09, Captures = 0, Avg Reward = 512.37, Avg Captures = 0.00
Actor Loss: -0.0184 | Critic Loss: 0.3889 | Avg Entropy: 1.6028
Episode 72: Reward = 378.70, Captures = 0, Avg Reward = 508.32, Avg Captures = 0.00
Actor Loss: -0.0153 | Critic Loss: 0.2261 | Avg Entropy: 1.6025
Episode 73: Reward = 519.80, Captures = 0, Avg Reward = 508.66, Avg Captures = 0.00
Actor Loss: -0.0161 | Critic Loss: 0.2039 | Avg Entropy: 1.6018
Episode 74: Reward = 514.13, Captures = 0, Avg Reward = 508.81, Avg Captures = 0.00
Actor Loss: -0.0112 | Critic Loss: 0.2549 | Avg Entropy: 1.6006
Episode 75: Reward = 748.89, Captures = 0, Avg Reward = 515.48, Avg Captures = 0.00
Actor Loss: -0.0102 | Critic Loss: 0.3488 | Avg Entropy: 1.5983
Episode 76: Reward = 578.63, Captures = 0, Avg Reward = 517.19, Avg Captures = 0.00
Actor Loss: -0.0142 | Critic Loss: 0.2685 | Avg Entropy: 1.5976
Episode 77: Reward = 513.31, Captures = 0, Avg Reward = 517.09, Avg Captures = 0.00
Actor Loss: -0.0099 | Critic Loss: 0.1858 | Avg Entropy: 1.5975
Episode 78: Reward = 406.15, Captures = 0, Avg Reward = 514.24, Avg Captures = 0.00
Actor Loss: -0.0173 | Critic Loss: 0.2041 | Avg Entropy: 1.5976
Episode 79: Reward = 438.37, Captures = 0, Avg Reward = 512.35, Avg Captures = 0.00
Actor Loss: -0.0100 | Critic Loss: 0.1494 | Avg Entropy: 1.5975
Episode 80: Reward = 516.01, Captures = 0, Avg Reward = 512.43, Avg Captures = 0.00
Actor Loss: -0.0112 | Critic Loss: 0.2158 | Avg Entropy: 1.5972
Episode 81: Reward = 574.32, Captures = 0, Avg Reward = 513.91, Avg Captures = 0.00
Actor Loss: -0.0081 | Critic Loss: 0.2786 | Avg Entropy: 1.5962
Episode 82: Reward = 421.64, Captures = 0, Avg Reward = 511.76, Avg Captures = 0.00
Actor Loss: -0.0115 | Critic Loss: 0.1929 | Avg Entropy: 1.5960
Episode 83: Reward = 683.22, Captures = 0, Avg Reward = 515.66, Avg Captures = 0.00
Actor Loss: -0.0114 | Critic Loss: 0.3166 | Avg Entropy: 1.5967
Episode 84: Reward = 569.79, Captures = 0, Avg Reward = 516.86, Avg Captures = 0.00
Actor Loss: -0.0188 | Critic Loss: 0.2154 | Avg Entropy: 1.5965
Episode 85: Reward = 596.64, Captures = 0, Avg Reward = 518.60, Avg Captures = 0.00
Actor Loss: -0.0154 | Critic Loss: 0.3315 | Avg Entropy: 1.5961
Episode 86: Reward = 686.69, Captures = 0, Avg Reward = 522.17, Avg Captures = 0.00
Actor Loss: -0.0114 | Critic Loss: 0.2667 | Avg Entropy: 1.5953
Episode 87: Reward = 625.19, Captures = 0, Avg Reward = 524.32, Avg Captures = 0.00
Actor Loss: -0.0158 | Critic Loss: 0.4506 | Avg Entropy: 1.5953
Episode 88: Reward = 724.67, Captures = 0, Avg Reward = 528.41, Avg Captures = 0.00
Actor Loss: -0.0159 | Critic Loss: 0.4264 | Avg Entropy: 1.5953
Episode 89: Reward = 599.71, Captures = 0, Avg Reward = 529.83, Avg Captures = 0.00
Actor Loss: -0.0109 | Critic Loss: 0.3060 | Avg Entropy: 1.5954
Episode 90: Reward = 608.99, Captures = 0, Avg Reward = 531.39, Avg Captures = 0.00
Actor Loss: -0.0130 | Critic Loss: 0.2900 | Avg Entropy: 1.5958
Episode 91: Reward = 808.99, Captures = 0, Avg Reward = 536.72, Avg Captures = 0.00
Actor Loss: -0.0145 | Critic Loss: 0.4835 | Avg Entropy: 1.5952
Episode 92: Reward = 694.68, Captures = 0, Avg Reward = 539.70, Avg Captures = 0.00
Actor Loss: -0.0067 | Critic Loss: 0.3934 | Avg Entropy: 1.5945
Episode 93: Reward = 733.47, Captures = 0, Avg Reward = 543.29, Avg Captures = 0.00
Actor Loss: -0.0112 | Critic Loss: 0.3703 | Avg Entropy: 1.5936
Episode 94: Reward = 835.73, Captures = 0, Avg Reward = 548.61, Avg Captures = 0.00
Actor Loss: -0.0093 | Critic Loss: 0.4313 | Avg Entropy: 1.5934
Episode 95: Reward = 365.75, Captures = 0, Avg Reward = 545.34, Avg Captures = 0.00
Actor Loss: -0.0087 | Critic Loss: 0.3830 | Avg Entropy: 1.5925
Episode 96: Reward = 687.91, Captures = 0, Avg Reward = 547.85, Avg Captures = 0.00
Actor Loss: -0.0096 | Critic Loss: 0.3101 | Avg Entropy: 1.5948
Episode 97: Reward = 725.37, Captures = 0, Avg Reward = 550.91, Avg Captures = 0.00
Actor Loss: -0.0133 | Critic Loss: 0.4029 | Avg Entropy: 1.5959
Episode 98: Reward = 533.66, Captures = 0, Avg Reward = 550.61, Avg Captures = 0.00
Actor Loss: -0.0139 | Critic Loss: 0.2719 | Avg Entropy: 1.5960
Episode 99: Reward = 629.45, Captures = 0, Avg Reward = 551.93, Avg Captures = 0.00
Actor Loss: -0.0173 | Critic Loss: 0.3451 | Avg Entropy: 1.5968
Episode 100: Reward = 591.20, Captures = 0, Avg Reward = 552.57, Avg Captures = 0.00
Actor Loss: -0.0187 | Critic Loss: 0.3149 | Avg Entropy: 1.5975
"""

log_case2 = """
Episode 110: Reward = 77.56, Captures = 8, Avg Reward = 77.56, Avg Captures = 8.00
Actor Loss: -0.0195 | Critic Loss: 0.4405 | Avg Entropy: 1.6042
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
Actor Loss: -0.0148 | Critic Loss: 0.1163 | Avg Entropy: 1.6055
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

# 2. 正規表現を用いたパース用関数
def parse_log(log_text):
    # エピソード行とロス行を結合して処理しやすくする
    lines = [line.strip() for line in log_text.strip().split("\n") if line.strip() and "チェックポイント" not in line]
    
    data = []
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        ep_line = lines[i]
        loss_line = lines[i+1]
        
        # 数値の抽出
        ep_match = re.search(r"Episode (\d+): Reward = ([\d\.-]+), Captures = (\d+), Avg Reward = ([\d\.-]+), Avg Captures = ([\d\.-]+)", ep_line)
        loss_match = re.search(r"Actor Loss: ([\d\.-]+) \| Critic Loss: ([\d\.-]+) \| Avg Entropy: ([\d\.-]+)", loss_line)
        
        if ep_match and loss_match:
            data.append({
                "Episode": int(ep_match.group(1)),
                "Reward": float(ep_match.group(2)),
                "Captures": int(ep_match.group(3)),
                "Avg_Reward": float(ep_match.group(4)),
                "Avg_Captures": float(ep_match.group(5)),
                "Actor_Loss": float(loss_match.group(1)),
                "Critic_Loss": float(loss_match.group(2)),
                "Avg_Entropy": float(loss_match.group(3))
            })
            
    df = pd.DataFrame(data)
    # 開始ステップを0に揃えるための相対インデックスを作成
    df["Relative_Step"] = range(len(df))
    return df

df1 = parse_log(log_case1)
df2 = parse_log(log_case2)

# 3. グラフの描画設定 (5つの指標を可視化)
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
axes = axes.flatten()

metrics = [
    ("Reward", "Episode Reward", "blue", "orange"),
    ("Captures", "Episode Captures", "blue", "orange"),
    ("Actor_Loss", "Actor Loss", "blue", "orange"),
    ("Critic_Loss", "Critic Loss", "blue", "orange"),
    ("Avg_Entropy", "Average Entropy", "blue", "orange")
]

for ax, (col, title, c1, c2) in zip(axes[:5], metrics):
    ax.plot(df1["Relative_Step"], df1[col], label="Case 1 (Ep 42-100)", color=c1, alpha=0.7)
    ax.plot(df2["Relative_Step"], df2[col], label="Case 2 (Ep 110-150)", color=c2, alpha=0.7)
    
    # 報酬とキャプチャには移動平均 (Avg) も点線で追加
    if col in ["Reward", "Captures"]:
        ax.plot(df1["Relative_Step"], df1[f"Avg_{col}"], color=c1, linestyle="--", label=f"Case 1 Avg")
        ax.plot(df2["Relative_Step"], df2[f"Avg_{col}"], color=c2, linestyle="--", label=f"Case 2 Avg")
        
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Steps (Aligned Start)")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

# 最後の空きサブプロットを非表示にする
axes[5].axis("off")

plt.tight_layout()
plt.show()