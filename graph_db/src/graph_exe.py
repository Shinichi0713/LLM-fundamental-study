import networkx as nx
import matplotlib.pyplot as plt

# 1. グラフ（データベース）の初期化
G = nx.Graph()

# 2. ノード（データ）の追加
G.add_node("User_A", label="Person", name="Alice")
G.add_node("User_B", label="Person", name="Bob")
G.add_node("Item_X", label="Product", name="Smartphone")

# 3. エッジ（関係性）の追加
G.add_edge("User_A", "User_B", relation="FRIEND")
G.add_edge("User_A", "Item_X", relation="PURCHASED")

# 4. 可視化
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()