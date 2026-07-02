import matplotlib.pyplot as plt
import numpy as np

def visualize_finite_ordinals(n=8):
    """
    有限順序数 0,1,2,...,n-1 を数直線上に可視化する。
    順序数は「整列した番号」としてのイメージ。
    """
    ordinals = np.arange(n)
    y = np.zeros_like(ordinals)

    plt.figure(figsize=(10, 2))
    plt.scatter(ordinals, y, s=100, color='red', zorder=3)
    for i in ordinals:
        plt.text(i, 0.1, str(i), ha='center', va='bottom', fontsize=12)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.yticks([])
    plt.xlabel('順序数')
    plt.title(f'有限順序数 0,1,...,{n-1} のイメージ\n（整列した番号の列）')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# 例：0〜7までの順序数
visualize_finite_ordinals(8)

def visualize_omega_ordinal(n_finite=6):
    """
    有限順序数 0,1,...,n-1 と、最初の無限順序数 ω の関係を可視化する。
    """
    finite_ordinals = np.arange(n_finite)
    y_finite = np.zeros_like(finite_ordinals)

    # ω の位置（有限の右側）
    omega_pos = n_finite + 1
    y_omega = 0

    plt.figure(figsize=(10, 2))
    # 有限順序数
    plt.scatter(finite_ordinals, y_finite, s=100, color='red', zorder=3, label='有限順序数')
    for i in finite_ordinals:
        plt.text(i, 0.1, str(i), ha='center', va='bottom', fontsize=12)

    # ω
    plt.scatter([omega_pos], [y_omega], s=100, color='blue', zorder=3, label='ω')
    plt.text(omega_pos, 0.1, 'ω', ha='center', va='bottom', fontsize=12)

    # 「...」で続きを表現
    plt.text(n_finite-0.5, -0.2, "...", fontsize=14, ha='center', va='top')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.yticks([])
    plt.xlabel('順序数')
    plt.title('有限順序数と最初の無限順序数 ω のイメージ')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

visualize_omega_ordinal()

import networkx as nx

def visualize_ordinal_as_order_type():
    """
    同じ順序数を持つ2つの整列集合が「同じ形」であることを可視化する。
    ここでは順序数 3 を持つ整列集合の例。
    """
    # 整列集合 A: {a < b < c}
    G1 = nx.DiGraph()
    G1.add_edges_from([('a', 'b'), ('b', 'c')])
    pos1 = {'a': (0,0), 'b': (0,1), 'c': (0,2)}

    # 整列集合 B: {x < y < z}（A と順序同型）
    G2 = nx.DiGraph()
    G2.add_edges_from([('x', 'y'), ('y', 'z')])
    pos2 = {'x': (2,0), 'y': (2,1), 'z': (2,2)}

    plt.figure(figsize=(8, 4))
    # 集合 A
    nx.draw_networkx_nodes(G1, pos1, node_color='lightblue', node_size=800)
    nx.draw_networkx_labels(G1, pos1, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G1, pos1, edge_color='blue', arrows=True, arrowsize=20)
    plt.text(0, 2.5, "整列集合 A", ha='center', va='bottom', fontsize=12)

    # 集合 B
    nx.draw_networkx_nodes(G2, pos2, node_color='lightgreen', node_size=800)
    nx.draw_networkx_labels(G2, pos2, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G2, pos2, edge_color='green', arrows=True, arrowsize=20)
    plt.text(2, 2.5, "整列集合 B", ha='center', va='bottom', fontsize=12)

    # 順序同型の対応を示す矢印
    plt.arrow(0.5, 1, 1, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
    plt.text(1, 1.2, "順序同型", ha='center', va='bottom', fontsize=10, color='red')

    plt.title("順序数は「整列順序の型」\n（同じ順序数を持つ集合は順序同型）")
    plt.axis('off')
    plt.show()

visualize_ordinal_as_order_type()

def visualize_ordinal_as_order_type():
    """
    Visualize that two well-ordered sets with the same ordinal number have "the same shape".
    Here we use an example of sets with ordinal number 3.
    """
    # Well-ordered set A: {a < b < c}
    G1 = nx.DiGraph()
    G1.add_edges_from([('a', 'b'), ('b', 'c')])
    pos1 = {'a': (0,0), 'b': (0,1), 'c': (0,2)}

    # Well-ordered set B: {x < y < z} (order-isomorphic to A)
    G2 = nx.DiGraph()
    G2.add_edges_from([('x', 'y'), ('y', 'z')])
    pos2 = {'x': (2,0), 'y': (2,1), 'z': (2,2)}

    plt.figure(figsize=(8, 4))
    # Set A
    nx.draw_networkx_nodes(G1, pos1, node_color='lightblue', node_size=800)
    nx.draw_networkx_labels(G1, pos1, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G1, pos1, edge_color='blue', arrows=True, arrowsize=20)
    plt.text(0, 2.5, "Well-ordered set A", ha='center', va='bottom', fontsize=12)

    # Set B
    nx.draw_networkx_nodes(G2, pos2, node_color='lightgreen', node_size=800)
    nx.draw_networkx_labels(G2, pos2, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G2, pos2, edge_color='green', arrows=True, arrowsize=20)
    plt.text(2, 2.5, "Well-ordered set B", ha='center', va='bottom', fontsize=12)

    # Arrow indicating order isomorphism
    plt.arrow(0.5, 1, 1, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
    plt.text(1, 1.2, "Order isomorphism", ha='center', va='bottom', fontsize=10, color='red')

    plt.title("Ordinal numbers are 'order types'\n(Sets with the same ordinal are order-isomorphic)")
    plt.axis('off')
    plt.show()

visualize_ordinal_as_order_type()

import matplotlib.pyplot as plt
import networkx as nx

def visualize_comparable_sets():
    """
    集合の包含関係による半順序集合で、
    比較可能なペアと比較不能なペアを可視化する。
    """
    # 集合族: ∅, {1}, {2}, {1,2}
    sets = {
        '∅': set(),
        '{1}': {1},
        '{2}': {2},
        '{1,2}': {1,2}
    }

    # 包含関係による半順序
    G = nx.DiGraph()
    G.add_edges_from([
        ('∅', '{1}'),
        ('∅', '{2}'),
        ('{1}', '{1,2}'),
        ('{2}', '{1,2}')
    ])

    pos = {
        '∅': (0, 0),
        '{1}': (-1, 1),
        '{2}': (1, 1),
        '{1,2}': (0, 2)
    }

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)

    # 比較可能なペア（包含関係がある）を強調
    comparable_pairs = [('∅', '{1}'), ('∅', '{2}'), ('{1}', '{1,2}'), ('{2}', '{1,2}')]
    nx.draw_networkx_edges(G, pos, edgelist=comparable_pairs,
                           edge_color='green', arrows=True, arrowsize=20, width=2)

    # 比較不能なペア（{1} と {2}）を強調
    nx.draw_networkx_nodes(G, pos, nodelist=['{1}', '{2}'],
                            node_color='orange', node_size=800)

    plt.title("集合の包含関係による半順序\n（緑：比較可能、オレンジ：比較不能なペア）")
    plt.axis('off')
    plt.show()

visualize_comparable_sets()

def visualize_totally_ordered_sets():
    """
    数直線上に元を並べ、すべての元が比較可能（全順序）であることを可視化する。
    """
    values = [1, 3, 5, 7]
    y = np.zeros_like(values)

    plt.figure(figsize=(10, 2))
    plt.scatter(values, y, s=100, color='red', zorder=3)
    for x in values:
        plt.text(x, 0.1, str(x), ha='center', va='bottom', fontsize=12)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.yticks([])
    plt.xlabel('値')
    plt.title("全順序集合のイメージ\n（すべての元が比較可能）")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

visualize_totally_ordered_sets()