#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BFS（幅優先探索）の矢印付き可視化
networkx + matplotlib を使用

実行方法:
    python bfs_visualization.py
"""

from collections import deque
import networkx as nx
import matplotlib.pyplot as plt


def bfs_with_arrow_visualization(graph_dict, start):
    """
    BFSを実行し、探索順を矢印付きで可視化する

    Parameters:
        graph_dict: dict
            隣接リスト表現のグラフ
            例: {'A': ['B', 'C'], 'B': ['A', 'D'], ...}
        start: str
            始点のノード名

    Returns:
        visited: list
            BFSの訪問順
        bfs_edges: list
            BFSの探索経路（親→子のエッジリスト）
    """

    # ============================================
    # 1. 有向グラフの作成（矢印を表示するた）
    # ============================================
    G = nx.DiGraph()
    for node, neighbors in graph_dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    print("=" * 60)
    print("BFS（幅優先探索）矢印付き可視化")
    print("=" * 60)
    print(f"\nグラフのノード数: {G.number_of_nodes()}")
    print(f"グラフの辺数: {G.number_of_edges()}")
    print(f"ノード一覧: {list(G.nodes())}")
    print(f"辺一覧: {list(G.edges())}")
    print()

    # ============================================
    # 2. BFSの実行
    # ============================================
    visited = []           # 訪問順を記録
    queue = deque([start]) # 探索用キュー（FIFO）
    seen = set([start])    # 訪問済み管理
    parent = {}            # 経路復元用: child -> parent

    print("【BFS探索プロセス】")
    print("-" * 60)

    step = 0
    while queue:
        step += 1
        node = queue.popleft()
        visited.append(node)

        print(f"ステップ{step}: '{node}' を訪問")

        for neighbor in G.neighbors(node):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = node  # neighborはnodeから発見された
                print(f"  -> '{neighbor}' をキューに追加（'{node}'から発見）")

        print(f"  キューの状態: {list(queue)}")

    # BFSの探索経路エッジを抽出
    bfs_edges = []
    for child, par in parent.items():
        bfs_edges.append((par, child))

    print("-" * 60)
    print(f"訪問順: {' -> '.join(visited)}")
    print(f"探索経路（親->子）: {bfs_edges}")
    print()

    # ============================================
    # 3. 可視化（矢印付き）
    # ============================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ノードのレイアウト位置計算（固定シードで再現性を確保）
    pos = nx.spring_layout(G, seed=42)

    # --- 左図: グラフ全体（全エッジ） ---
    ax1 = axes[0]

    # 全エッジを描画（薄いグレーの矢印）
    nx.draw_networkx_edges(
        G, pos, ax=ax1,
        edge_color='lightgray',
        arrows=True,
        arrowsize=20,
        width=1.5,
        connectionstyle='arc3,rad=0.1'
    )

    # 全ノードを描画（水色）
    nx.draw_networkx_nodes(
        G, pos, ax=ax1,
        node_color='lightblue',
        node_size=1500,
        edgecolors='black'
    )

    # ノードラベルを描画
    nx.draw_networkx_labels(
        G, pos, ax=ax1,
        font_size=12,
        font_weight='bold'
    )

    ax1.set_title('Graph (All Edges)', fontsize=14)
    ax1.axis('off')

    # --- 右図: BFS探索順の経路（強調表示） ---
    ax2 = axes[1]

    # 全エッジを薄い色で描画（背景として）
    nx.draw_networkx_edges(
        G, pos, ax=ax2,
        edge_color='lightgray',
        arrows=True,
        arrowsize=15,
        width=1,
        connectionstyle='arc3,rad=0.1'
    )

    # BFS経路のエッジを赤色・太線で強調
    nx.draw_networkx_edges(
        G, pos, ax=ax2,
        edgelist=bfs_edges,
        edge_color='red',
        arrows=True,
        arrowsize=25,
        width=3,
        connectionstyle='arc3,rad=0.1'
    )

    # ノードの色を設定（始点=緑、それ以外=オレンジ）
    node_colors = [
        'green' if n == start else 'orange'
        for n in G.nodes()
    ]

    nx.draw_networkx_nodes(
        G, pos, ax=ax2,
        node_color=node_colors,
        node_size=1500,
        edgecolors='black'
    )

    # 訪問順のラベルを追加
    visit_labels = {}
    for i, node in enumerate(visited):
        visit_labels[node] = f"{node}\n({i+1})"

    nx.draw_networkx_labels(
        G, pos, visit_labels, ax=ax2,
        font_size=10,
        font_weight='bold'
    )

    ax2.set_title(
        f"BFS Traversal Order\n{' -> '.join(visited)}",
        fontsize=14
    )
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('bfs_arrows_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("画像を保存しました: bfs_arrows_visualization.png")

    return visited, bfs_edges

import heapq


def astar(grid, start, goal):
    """
    A*アルゴリズムで迷路の最短経路を探索する。

    Parameters:
        grid: 2次元リスト (0=通路, 1=壁)
        start: (行, 列) スタート位置
        goal:  (行, 列) ゴール位置

    Returns:
        path: 最短経路座標リスト
        cost: 総コスト
        explored_count: 探索したセル数
    """
    rows = len(grid)
    cols = len(grid[0])

    def heuristic(a, b):
        """マンハッタン距離（ヒューリスティック）"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # 優先度付きキュー: (f値, g値, 位置)
    heap = []
    start_h = heuristic(start, goal)
    heapq.heappush(heap, (start_h, 0, start))

    g = {start: 0}        # 始点からの実コスト
    came_from = {}         # 経路復元用
    visited = set()        # 確定済みセル
    explored = []          # 探索順記録

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while heap:
        f, g_val, current = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)
        explored.append(current)

        if current == goal:
            # 経路復元
            path = []
            node = goal
            while node != start:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, g_val, len(explored)

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            # 範囲外または壁ならスキップ
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr][nc] == 1:
                continue

            new_g = g_val + 1
            if neighbor not in g or new_g < g[neighbor]:
                g[neighbor] = new_g
                f_val = new_g + heuristic(neighbor, goal)
                heapq.heappush(heap, (f_val, new_g, neighbor))
                came_from[neighbor] = current

    return None, None, len(explored)


def print_grid(grid, path=None, explored=None, start=None, goal=None):
    """迷路を可視化して表示する"""
    rows = len(grid)
    cols = len(grid[0])

    print("   ", end="")
    for c in range(cols):
        print(f"{c} ", end="")
    print()

    for r in range(rows):
        print(f"{r}  ", end="")
        for c in range(cols):
            pos = (r, c)
            if start and pos == start:
                print("S ", end="")
            elif goal and pos == goal:
                print("G ", end="")
            elif path and pos in path:
                print("* ", end="")
            elif explored and pos in explored:
                print("+ ", end="")
            elif grid[r][c] == 1:
                print("# ", end="")
            else:
                print(". ", end="")
        print()


# ========== メイン処理 ==========
if __name__ == "__main__":
    # 迷路の定義 (0=通路, 1=壁)
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]

    start = (0, 0)
    goal = (4, 4)

    print("=" * 50)
    print("A* アルゴリズム 最短経路探索")
    print("=" * 50)
    print()
    print("【迷路】")
    print("   S = スタート, G = ゴール, # = 壁, . = 通路")
    print()
    print_grid(grid, start=start, goal=goal)
    print()

    # A*で最短経路を探索
    path, cost, explored_count = astar(grid, start, goal)

    print("【結果】")
    print(f"最短経路: {path}")
    print(f"経路の長さ（総コスト）: {cost}")
    print(f"探索したセル数: {explored_count}")
    print()

    print("【最短経路の可視化】")
    print("   * = 最短経路")
    print_grid(grid, path=path, start=start, goal=goal)

# ============================================
# メイン処理
# ============================================
if __name__ == "__main__":

    # グラフの定義（隣接リスト）
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D'],
        'C': ['A', 'D'],
        'D': ['B', 'C']
    }

    # BFS実行と可視化
    visited, bfs_edges = bfs_with_arrow_visualization(graph, start='A')
