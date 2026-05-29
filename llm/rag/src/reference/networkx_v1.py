import networkx as nx

# 無向グラフの作成
G = nx.Graph()

# ノードの追加
G.add_node("A")
G.add_node("B")
G.add_node("C")

# エッジの追加
G.add_edge("A", "B")
G.add_edge("B", "C")
G.add_edge("A", "C")

print("ノード一覧:", list(G.nodes()))
print("エッジ一覧:", list(G.edges()))
print("次数（degree）:", dict(G.degree()))

# ドキュメントノードとエンティティノードを持つグラフ
G = nx.Graph()

# ドキュメントノード
G.add_node("doc1", type="DOCUMENT", text="Apple Inc. is headquartered in Cupertino.")
G.add_node("doc2", type="DOCUMENT", text="Steve Jobs co-founded Apple.")

# エンティティノード
G.add_node("Apple Inc.", type="ENTITY", label="ORG")
G.add_node("Cupertino", type="ENTITY", label="GPE")
G.add_node("Steve Jobs", type="ENTITY", label="PERSON")

# エッジ（ドキュメントがエンティティを言及）
G.add_edge("doc1", "Apple Inc.", relation="MENTIONS")
G.add_edge("doc1", "Cupertino", relation="MENTIONS")
G.add_edge("doc2", "Apple Inc.", relation="MENTIONS")
G.add_edge("doc2", "Steve Jobs", relation="MENTIONS")

# ノード属性の確認
for n in G.nodes():
    print(f"ノード {n}: {G.nodes[n]}")


def graph_rag_with_ollama(query: str, context_texts: list, model: str = "llama3.1:8b"):
    # コンテキストを結合
    context_str = "\n\n".join(context_texts)
    prompt = f"""
以下の文書を参考に、質問に答えてください。

参考文書:
{context_str}

質問: {query}

回答:
"""
    return ask_ollama_lib(prompt, model=model)

# 実行例
contexts = [
    "Apple Inc. is headquartered in Cupertino, California.",
    "Cupertino is a city in California."
]
answer = graph_rag_with_ollama("Where is Apple Inc. headquartered?", contexts)
print(answer)

# ドキュメントノードとエンティティノードを持つグラフ
G = nx.Graph()

# ドキュメントノード
G.add_node("doc1", type="DOCUMENT", text="Apple Inc. is headquartered in Cupertino.")
G.add_node("doc2", type="DOCUMENT", text="Steve Jobs co-founded Apple.")

# エンティティノード
G.add_node("Apple Inc.", type="ENTITY", label="ORG")
G.add_node("Cupertino", type="ENTITY", label="GPE")
G.add_node("Steve Jobs", type="ENTITY", label="PERSON")

# エッジ（ドキュメントがエンティティを言及）
G.add_edge("doc1", "Apple Inc.", relation="MENTIONS")
G.add_edge("doc1", "Cupertino", relation="MENTIONS")
G.add_edge("doc2", "Apple Inc.", relation="MENTIONS")
G.add_edge("doc2", "Steve Jobs", relation="MENTIONS")

# ノード属性の確認
for n in G.nodes():
    print(f"ノード {n}: {G.nodes[n]}")

# 例：小さなソーシャルネットワーク
G_social = nx.Graph()
G_social.add_edges_from([
    ("Alice", "Bob"),
    ("Alice", "Charlie"),
    ("Bob", "David"),
    ("Charlie", "David"),
    ("David", "Eve")
])

# 次数中心性（Degree Centrality）
degree_centrality = nx.degree_centrality(G_social)
print("次数中心性:", degree_centrality)

# 媒介中心性（Betweenness Centrality）
betweenness_centrality = nx.betweenness_centrality(G_social)
print("媒介中心性:", betweenness_centrality)

# 近接中心性（Closeness Centrality）
closeness_centrality = nx.closeness_centrality(G_social)
print("近接中心性:", closeness_centrality)

# 例：コミュニティ構造を持つグラフ
G_comm = nx.karate_club_graph()  # 有名なザッカーバーグの空手クラブネットワーク

# Louvain法によるコミュニティ検出（communityライブラリが必要）
# !pip install python-louvain
import community as community_louvain

partition = community_louvain.best_partition(G_comm)
print("コミュニティ割り当て:", partition)

# コミュニティごとのノード数
from collections import Counter
comm_counts = Counter(partition.values())
print("各コミュニティのサイズ:", comm_counts)

import matplotlib.pyplot as plt

# 小さなグラフを描画
G_viz = nx.Graph()
G_viz.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")])

# レイアウトを計算
pos = nx.spring_layout(G_viz)

# 描画
plt.figure(figsize=(6, 4))
nx.draw(G_viz, pos, with_labels=True, node_color="lightblue", 
        node_size=500, font_size=12, font_weight="bold")
plt.title("シンプルなグラフ")
plt.show()

# 属性付きグラフの可視化（Graph RAG風）
G_attr = nx.Graph()
G_attr.add_node("doc1", type="DOCUMENT")
G_attr.add_node("Apple", type="ENTITY")
G_attr.add_node("Cupertino", type="ENTITY")
G_attr.add_edge("doc1", "Apple")
G_attr.add_edge("doc1", "Cupertino")

pos = nx.spring_layout(G_attr)

# ノードの色をタイプで変える
node_colors = []
for n in G_attr.nodes():
    if G_attr.nodes[n]["type"] == "DOCUMENT":
        node_colors.append("red")
    else:
        node_colors.append("lightgreen")

plt.figure(figsize=(6, 4))
nx.draw(G_attr, pos, with_labels=True, node_color=node_colors, 
        node_size=800, font_size=10)
plt.title("Graph RAG風のグラフ（ドキュメントとエンティティ）")
plt.show()