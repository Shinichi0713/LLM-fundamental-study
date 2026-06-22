import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output

# =====================================================================
# 1. 迷路環境（グラフ構造）の構築
# =====================================================================
# 6x6のグリッド迷路
grid_size = 6
G = nx.grid_2d_graph(grid_size, grid_size)

# 迷路の「壁」を設定（通行不可にするためエッジを削除）
walls = [
    ((1, 1), (1, 2)), ((1, 2), (1, 3)),
    ((2, 3), (2, 4)), ((3, 1), (3, 2)),
    ((4, 3), (4, 4)), ((3, 4), (4, 4)),
    ((4, 1), (5, 1)), ((4, 2), (5, 2))
]
for u, v in walls:
    if G.has_edge(u, v):
        G.remove_edge(u, v)

# ★グラフ理論のおもしろ要素「泥沼のトラップ地帯」を設定★
# 特定のマスを通るエッジには、通常の5倍の移動コスト（重み）を設定する
mud_nodes = [(2, 1), (2, 2), (3, 2), (3, 3)]

for u, v in G.edges():
    # どちらかのノードが泥沼ならコストを5、それ以外は1にする
    if u in mud_nodes or v in mud_nodes:
        G[u][v]['weight'] = 5.0
    else:
        G[u][v]['weight'] = 1.0

# スタートとゴールの位置
start_node = (0, 0)
goal_node = (5, 5)

# ★グラフ理論（ダイクストラ法）で最短経路を一瞬で計算！
# 'weight'（重み）を考慮して最小コストのルートを弾き出す
shortest_path = nx.dijkstra_path(G, source=start_node, target=goal_node, weight='weight')

# =====================================================================
# 2. 自動走破シミュレーションとリアルタイム描画
# =====================================================================
# 各マスの固定配置座標
pos = {node: (node[1], grid_size - 1 - node[0]) for node in G.nodes()}

print("迷路の探索を開始します...")

for step, current_pos in enumerate(shortest_path):
    clear_output(wait=True)
    plt.figure(figsize=(8, 8))
    
    # --- マスの色分け ---
    node_colors = []
    for node in G.nodes():
        if node == current_pos: node_colors.append("#f1c40f")    # プレイヤー（黄色）
        elif node == start_node: node_colors.append("#2ecc71")   # スタート（緑）
        elif node == goal_node: node_colors.append("#e74c3c")    # ゴール（赤）
        elif node in mud_nodes: node_colors.append("#95a5a6")    # 泥沼トラップ（グレー）
        else: node_colors.append("#ecf0f1")                     # 通常の床（白）

    # ノードとエッジの描画
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, edgecolors="#bdc3c7")
    
    # エッジ（道）の描画。泥沼を通るルートは点線にして不穏さを演出
    normal_edges = [(u, v) for u, v in G.edges() if G[u][v]['weight'] == 1.0]
    muddy_edges = [(u, v) for u, v in G.edges() if G[u][v]['weight'] == 5.0]
    
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=2, edge_color="#2c3e50")
    nx.draw_networkx_edges(G, pos, edgelist=muddy_edges, width=2, edge_color="#7f8c8d", style="dashed")
    
    # プレイヤーの位置を★マークで強調
    nx.draw_networkx_labels(G, pos, labels={current_pos: "⭐"}, font_size=14)
    
    # スタート、ゴール、泥沼のテキストラベル
    text_labels = {start_node: "Start", goal_node: "Goal"}
    for mn in mud_nodes:
        if mn != current_pos:
            text_labels[mn] = "Mud"
    nx.draw_networkx_labels(G, pos, labels=text_labels, font_size=8, font_weight="bold")

    plt.title(f"Graph Pathfinding (Dijkstra) - Step: {step}\n"
              f"⭐ 現在地: {current_pos} | 泥沼（Mud）を巧妙に迂回しながら進んでいます", 
              fontsize=12, fontweight="bold", loc="left")
    plt.axis("off")
    plt.show()
    
    time.sleep(0.6)

print("🎉 無事に最短ルートでゴールに到達しました！")