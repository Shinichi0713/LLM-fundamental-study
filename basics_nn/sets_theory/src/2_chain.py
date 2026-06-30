import matplotlib.pyplot as plt
import networkx as nx

def visualize_chain_in_poset():
    """
    半順序集合（ハッセ図）の中で、鎖（全順序部分集合）のイメージを可視化する。
    """
    # 半順序集合の例（ハッセ図）
    # 要素: a < b < c < d（鎖）, かつ e, f は比較不能
    G = nx.DiGraph()
    G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])
    G.add_nodes_from(['e', 'f'])  # 比較不能な要素

    pos = {
        'a': (0, 0),
        'b': (0, 1),
        'c': (0, 2),
        'd': (0, 3),
        'e': (-1, 1.5),
        'f': (1, 1.5)
    }

    plt.figure(figsize=(6, 6))
    # すべてのノード（薄い色）
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)

    # 鎖 a < b < c < d を強調（オレンジ）
    chain_nodes = ['a', 'b', 'c', 'd']
    nx.draw_networkx_nodes(G, pos, nodelist=chain_nodes,
                           node_color='orange', node_size=800)
    nx.draw_networkx_edges(G, pos, edgelist=[('a','b'),('b','c'),('c','d')],
                           edge_color='orange', arrows=True, arrowsize=20, width=2)

    plt.title("半順序集合における鎖（オレンジ）のイメージ")
    plt.axis('off')
    plt.show()

visualize_chain_in_poset()


import matplotlib.pyplot as plt
import numpy as np

def visualize_cyclic_group(n=6):
    """
    巡回群 ℤ/nℤ（mod n の足し算）を円周上の点として可視化する。
    これは有限可換群の典型例。
    """
    # 0,1,...,n-1 を円周上に配置
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=100, color='red', zorder=3)
    for i in range(n):
        plt.text(x[i]*1.1, y[i]*1.1, str(i), fontsize=12,
                 ha='center', va='center')

    circle = plt.Circle((0,0), 1, fill=False, color='black', linestyle='--')
    plt.gca().add_artist(circle)
    plt.axis('equal')
    plt.title(f'有限可換群 ℤ/{n}ℤ のイメージ\n（mod {n} の足し算）')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# 例：ℤ/6ℤ
visualize_cyclic_group(6)

import matplotlib.pyplot as plt
import networkx as nx

def visualize_poset_with_maximal():
    """
    半順序集合（ハッセ図）で、極大元のイメージを可視化する。
    ここでは有限の場合を例示（Zornの補題は自明に成立）。
    """
    # 半順序集合の例（ハッセ図）
    # 要素: a < b, a < c, b < d, c < d
    # d が極大元
    G = nx.DiGraph()
    G.add_edges_from([('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd')])

    pos = {
        'a': (0, 0),
        'b': (-1, 1),
        'c': (1, 1),
        'd': (0, 2)
    }

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)

    # 極大元 d を強調
    nx.draw_networkx_nodes(G, pos, nodelist=['d'], node_color='red', node_size=800)

    plt.title("半順序集合における極大元のイメージ\n（Zornの補題は有限では自明）")
    plt.axis('off')
    plt.show()

visualize_poset_with_maximal()

def visualize_chain_and_upper_bound():
    """
    半順序集合の鎖と、その上界のイメージを可視化する。
    """
    # 半順序集合の例
    # 要素: a < b < c < d, かつ e は a,b,c,d と比較不能だが、d の上にある
    G = nx.DiGraph()
    G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])
    # e は d の上にあるが、順序関係はない（図では上に配置）

    pos = {
        'a': (0, 0),
        'b': (0, 1),
        'c': (0, 2),
        'd': (0, 3),
        'e': (1, 4)
    }

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)

    # 鎖 a < b < c < d を強調
    chain_nodes = ['a', 'b', 'c', 'd']
    nx.draw_networkx_nodes(G, pos, nodelist=chain_nodes, node_color='orange', node_size=800)
    # 上界 e を強調
    nx.draw_networkx_nodes(G, pos, nodelist=['e'], node_color='green', node_size=800)

    plt.title("鎖（オレンジ）とその上界（緑）のイメージ")
    plt.axis('off')
    plt.show()

visualize_chain_and_upper_bound()

def visualize_chain_without_upper_bound():
    """
    「上に伸び続ける鎖」に上界がない半順序集合のイメージを可視化する。
    実際には有限個の点で表現する。
    """
    # 自然数の鎖 0 < 1 < 2 < ... を有限個で表現
    n_nodes = 6
    G = nx.DiGraph()
    for i in range(n_nodes - 1):
        G.add_edge(str(i), str(i+1))

    pos = {str(i): (0, i) for i in range(n_nodes)}

    plt.figure(figsize=(4, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)

    # 鎖を強調
    nx.draw_networkx_nodes(G, pos, nodelist=[str(i) for i in range(n_nodes)],
                           node_color='orange', node_size=800)
    nx.draw_networkx_edges(G, pos, edgelist=[(str(i), str(i+1)) for i in range(n_nodes-1)],
                           edge_color='orange', arrows=True, arrowsize=20, width=2)

    # 「上に続く」ことを示す矢印
    plt.arrow(0, n_nodes-0.5, 0, 0.5, head_width=0.1, head_length=0.1, fc='red', ec='red')
    plt.text(0.2, n_nodes-0.2, "...", fontsize=14)

    plt.title("上に伸び続ける鎖（上界がない）のイメージ\n（Zornの補題の条件を満たさない例）")
    plt.axis('off')
    plt.show()

visualize_chain_without_upper_bound()