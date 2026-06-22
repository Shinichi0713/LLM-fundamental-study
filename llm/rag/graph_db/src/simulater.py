import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output

# =====================================================================
# 1. スケールフリーネットワーク（現実社会に近いグラフ）の自動生成
# =====================================================================
np.random.seed(42)
num_nodes = 50

# バラバシ・アルバートモデル（富める者がさらに富むルールで、ハブが生まれるグラフ構造）
G = nx.barabasi_albert_graph(n=num_nodes, m=2, seed=42)

# ノードの状態を定義 (0: 健康(青), 1: 感染(赤), 2: ロックダウン隔離(黒))
status = {i: 0 for i in range(num_nodes)}

# ★グラフ理論の主役「ハブノード」を見つける
# つながりの数（次数）が最も多いノードを特定する
degrees = dict(G.degree())
hub_node = max(degrees, key=degrees.get)

# 感染源の投入：あえて影響力の低い「端っこのノード」から感染をスタートさせます
patient_zero = [i for i in range(num_nodes) if G.degree(i) == 2][0]
status[patient_zero] = 1

# 描画用の固定位置を計算
pos = nx.spring_layout(G, seed=42)

# =====================================================================
# 2. タイムステップ・シミュレーション
# =====================================================================
infection_rate = 0.4  # 感染確率
lockdown_activated = False

print("シミュレーションを開始します...")

for step in range(1, 16):
    clear_output(wait=True)
    
    # 新しく感染するノードを記録するリスト
    new_infected = []
    
    # グラフのルールに従って感染を伝播（メッセージパッシングに近い処理）
    for node in G.nodes():
        if status[node] == 1: # もし自分が感染していたら
            for neighbor in G.neighbors(node):
                if status[neighbor] == 0: # 隣人が健康なら
                    if np.random.rand() < infection_rate:
                        new_infected.append(neighbor)
                        
    # 状態の更新
    for node in new_infected:
        status[node] = 1
        
    # ★グラフ理論による「戦略的ロックダウン（介入）」
    # ステップ5に達したら、最も危険な「ハブノード」に繋がるエッジ（ルート）を遮断する
    if step == 5:
        lockdown_activated = True
        # ハブノードの周りのエッジをすべて削除し、ハブ自体を孤立（ステータス2）させる
        hub_neighbors = list(G.neighbors(hub_node))
        for neighbor in hub_neighbors:
            G.remove_edge(hub_node, neighbor)
        status[hub_node] = 2 # 隔離状態
        
    # =====================================================================
    # 3. リアルタイム可視化
    # =====================================================================
    plt.figure(figsize=(10, 8))
    
    # ステータスに応じた色分け（0=青、1=赤、2=黒）
    node_colors = []
    for i in range(num_nodes):
        if status[i] == 0: node_colors.append("#3498db") # 健康
        elif status[i] == 1: node_colors.append("#e74c3c") # 感染
        else: node_colors.append("#2c3e50") # 隔離（ハブ）
        
    # ハブノードだけ少し大きく表示
    node_sizes = [1000 if i == hub_node else 300 for i in range(num_nodes)]
    
    # グラフの描画
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="white", linewidths=1)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray")
    
    # 特徴的なノードにラベルをつける
    labels = {patient_zero: "Start", hub_node: "HUB"}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="white", font_weight="bold")
    
    title_text = f"Graph Virus Simulation - Step: {step}\n"
    if step < 5:
        title_text += "⚠️ 感染が徐々に拡大しています..."
    elif step == 5:
        title_text += "🔥 グラフ理論発動：危険な【HUBノード】のネットワークを緊急遮断！"
    else:
        title_text += "🛡️ ロックダウン成功。ハブを切り離したため、右側のコミュニティが守られました"
        
    plt.title(title_text, fontsize=12, fontweight="bold", loc="left")
    plt.axis("off")
    plt.show()
    
    # 変化が早い場合は全て感染して終了
    if status.count(1) == num_nodes - 1:
        print("全員に感染が広がりました。")
        break
        
    time.sleep(0.8)