import matplotlib.pyplot as plt
import networkx as nx

# 1. 空の無向グラフ（方向のないグラフ）を作成
G = nx.Graph()

G.add_node("Alice")
G.add_nodes_from(["Bob", "Charlie", "David", "Ellen"])

G.add_edge("Alice", "Bob")
G.add_edge("Alice", "Charlie")
G.add_edge("Bob", "David")
G.add_edge("Charlie", "Ellen")
G.add_edge("David", "Ellen")

# 5. 可視化（描画）の設定
plt.figure(figsize=(6, 6))  # 描画サイズを指定

# ノードの配置（レイアウト）を計算（スプリングレイアウト：バランスよく配置する設定）
pos = nx.spring_layout(G, seed=42)  # seedを固定すると毎回同じ配置になります

# グラフを描画
nx.draw(
    G,
    pos,
    with_labels=True,  # ノードに名前（ラベル）を表示する
    node_color="lightblue",  # ノードの色
    node_size=2000,  # ノードの大きさ
    font_size=12,  # 文字の大きさ
    font_weight="bold",  # 文字の太さ
    edge_color="gray",  # 線の色
    width=2,  # 線の太さ
)

# 画面に表示
plt.title("Friendship Network Practice")
plt.show()


import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()

G.add_nodes_from(["Apple", "Steve Jobs", "iPhone", "iOS"])

G.add_edge("Steve Jobs", "Apple")
G.add_edge("Apple", "iPhone")
G.add_edge("iPhone", "iOS")

# 4. 可視化の設定
plt.figure(figsize=(8, 6))

# ノードの位置を決定
pos = nx.spring_layout(G, seed=42)


# 有向グラフをきれいに描画するための調整
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightgreen",  # 有向グラフ用に色をグリーンに変更
    node_size=3000,
    font_size=10,
    font_weight="bold",
    edge_color="black",
    width=2,
    arrows=True,  # 矢印を表示（有向グラフではデフォルトでTrueですが明示）
    arrowsize=20,  # 矢印のサイズを大きくして見やすくする
    arrowstyle="-|>",  # 矢印の形状を指定
    connectionstyle="arc3,rad=0.1",  # 直線ではなく少しカーブさせて矢印の向きをわかりやすくする
)

# 画面に表示
plt.title("Directed Graph (Knowledge Graph Practice)")
plt.show()

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community

# 1. 組み込みの空手クラブデータセットを読み込む
G = nx.karate_club_graph()

# 2. 高度な解析：コミュニティ検出（Girvan-Newman法）を実行
communities_generator = community.girvan_newman(G)
top_level_communities = next(communities_generator)
community_lists = sorted(map(list, top_level_communities))

# 3. 検出された2つのグループごとにノードの色を分ける設定
color_map = []
for node in G.nodes():
    if node in community_lists[0]:
        color_map.append("orange")      # グループ1
    else:
        color_map.append("lightgreen")  # グループ2（タイポを修正しました）

# 4. 可視化
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=color_map,  # AIが分類したグループごとに色分け
    node_size=600,
    font_size=10,
    font_color="black",
    edge_color="silver",
)

plt.title("Karate Club Community Detection Practice")
plt.show()

# 5. 分析結果の出力
print(f"グループ1の人数: {len(community_lists[0])}人")
print(f"グループ2の人数: {len(community_lists[1])}人")

import matplotlib.pyplot as plt
import networkx as nx

# 1. 組み込みのレ・ミゼラブル データセットを読み込む
# (登場人物どうしの共起関係ネットワーク)
G = nx.les_miserables_graph()

# 2. アルゴリズムの実行
# ① 度数中心性（単純にたくさんの人と関わっている度合い）
degree_cent = nx.degree_centrality(G)

# ② PageRank（重要な人物とつながっている度合い）
pagerank_cent = nx.pagerank(G)

# 3. 結果をランキング形式で並び替えて上位5名を出力
print("--- 度数中心性（関わりの多さ） TOP 5 ---")
sorted_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
for i, (name, score) in enumerate(sorted_degree[:5], 1):
    print(f"{i}位: {name:<20} (スコア: {score:.3f})")

print("\n--- PageRank（ネットワーク的な重要度） TOP 5 ---")
sorted_pagerank = sorted(pagerank_cent.items(), key=lambda x: x[1], reverse=True)
for i, (name, score) in enumerate(sorted_pagerank[:5], 1):
    print(f"{i}位: {name:<20} (スコア: {score:.3f})")

# 4. 可視化（PageRankのスコアに応じてノードの大きさを変える）
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)

# ノードの大きさをPageRankのスコアに比例させる（見やすくするために10000倍に拡大）
node_sizes = [pagerank_cent[node] * 10000 for node in G.nodes()]

# 上位3名だけラベルを大きく表示するための準備
labels = {node: node if pagerank_cent[node] > 0.04 else "" for node in G.nodes()}

nx.draw(
    G,
    pos,
    labels=labels,       # 重要な人物のみ名前を表示（画面の混雑を防ぐため）
    node_size=node_sizes,# PageRankが高いほど大きな丸になる
    node_color="tomato",
    edge_color="gainsboro",
    alpha=0.8,
    font_size=12,
    font_weight="bold"
)

plt.title("Les Miserables Network - Node Size by PageRank")
plt.show()

import matplotlib.pyplot as plt
import networkx as nx
from cdlib import algorithms

# 1. 複雑なコミュニティを持つグラフの自動生成（LFRベンチマーク風）
# 20人ずつのグループが3つ（計60人）あり、グループ内は高密度、グループ間は低密度で繋ぐ
sizes = [20, 20, 20]
# グループ内のつながる確率: 0.4 / グループ間のつながる確率: 0.03
probs = [[0.4, 0.03, 0.03], 
         [0.03, 0.4, 0.03], 
         [0.03, 0.03, 0.4]]

G = nx.stochastic_block_model(sizes, probs, seed=42)

# 2. レイデン法（Leiden）を実行
# cdlibのleidenアルゴリズムを呼び出します
leiden_communities = algorithms.leiden(G)

# 検出されたコミュニティ（グループのリスト）を取得
# 例: [[0, 1, 2...], [20, 21, 22...], ...]
communities = leiden_communities.communities
print(f"レイデン法が検出したコミュニティ数: {len(communities)}個")

# 3. ノードごとに色を割り振るためのカラーマップを作成
# 何個のグループに分かれても対応できるように、Matplotlibのカラーグラデーション（cm）を使用
cmap = plt.get_cmap("viridis", len(communities))
color_map = {}

for group_idx, community in enumerate(communities):
    for node in community:
        color_map[node] = cmap(group_idx)

# ノードのID順に色を並び替えたリスト
node_colors = [color_map[node] for node in sorted(G.nodes())]

# 4. 美しく可視化
plt.figure(figsize=(10, 10))

# ネットワークをきれいに広げるためのレイアウト計算
pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_colors,  # レイデン法が決めた色
    node_size=300,
    font_size=8,
    font_color="white",
    edge_color="gainsboro",  # 複雑な線を薄いグレーにして見やすくする
    alpha=0.9
)

plt.title("Leiden Algorithm - Community Detection Experiment", fontsize=14)
plt.show()