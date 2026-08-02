import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class MazeEnv:
    """
    5x5迷路環境クラス（動画保存機能付き）
    """
    def __init__(self, maze_file="maze.txt", rows=5, cols=5):
        if os.path.exists(maze_file):
            self.maze = self._load_maze(maze_file)
            self.start, self.goal = self._find_start_goal()
        else:
            self.generate_random_maze(rows=rows, cols=cols)

        self.state = self.start
        self.done = False
        self.history = []
        self.rows = rows
        self.cols = cols

        # --- 変更点: 報酬パラメータを一箇所にまとめて調整しやすくする ---
        self.step_penalty = -0.05        # 通常移動の基本ペナルティ
        self.wall_penalty = -0.6         # 壁衝突（遠回りより重くする）
        self.progress_reward = 0.3       # ゴールに近づいた場合
        self.regress_penalty = -0.15     # ゴールから遠ざかった場合（壁衝突より軽く）
        self.goal_reward = 20.0          # ゴール到達報酬（固定値でスケールを安定させる）
        self.stop_penalty = -0.4

    def _load_maze(self, file_path):
        maze = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ' ' in line:
                    row = line.split()
                else:
                    row = list(line)
                maze.append(row)
        return maze

    def _find_start_goal(self):
        start = None
        goal = None
        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                if cell == 'S':
                    start = (i, j)
                elif cell == 'G':
                    goal = (i, j)
        return start, goal

    def _is_valid_move(self, pos):
        x, y = pos
        if x < 0 or y < 0:
            return False
        if x >= len(self.maze) or y >= len(self.maze[0]):
            return False
        if self.maze[x][y] == 'W':
            return False
        return True

    # --- 追加: ゴールからの距離マップをBFSで一括計算 ---
    def _bfs_distance_map(self, maze, goal):
        """
        goalを起点に全マスまでの最短距離を計算する。
        到達不可能なマスはNoneとする。
        """
        rows = len(maze)
        cols = len(maze[0])
        dist = [[None] * cols for _ in range(rows)]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        dist[goal[0]][goal[1]] = 0
        queue = deque([goal])

        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if maze[nx][ny] != 'W' and dist[nx][ny] is None:
                        dist[nx][ny] = dist[x][y] + 1
                        queue.append((nx, ny))
        return dist

    def reset(self, maze_change=True):
        if maze_change:
            self.generate_random_maze()
        # --- 変更点: max_goal_reward を迷路の長さに依存させず固定化 ---
        # (以前は length_min * 3.0 でエピソードごとにスケールが変動していた)
        self.state = self.start
        self.done = False
        self.history = [self.state]

        # --- 追加: ゴールまでの距離マップをエピソード開始時に1回だけ計算 ---
        self.dist_map = self._bfs_distance_map(self.maze, self.goal)

        # --- 追加: 正規化用に到達可能マスの最大距離を保持 ---
        # （observationの距離チャンネルをTransformerに渡すために使う）
        finite_dists = [d for row in self.dist_map for d in row if d is not None]
        self._max_dist = max(finite_dists) if finite_dists else 1
        if self._max_dist <= 0:
            self._max_dist = 1

        return self.state

    def step(self, action):
        if self.done:
            raise ValueError("Episode is already done. Call reset() first.")

        x, y = self.state
        if action == 0:   # 上
            next_state = (x - 1, y)
        elif action == 1: # 下
            next_state = (x + 1, y)
        elif action == 2: # 左
            next_state = (x, y - 1)
        elif action == 3: # 右
            next_state = (x, y + 1)
        else:
            raise ValueError("Invalid action")

        # 1. 移動の成否判定
        hit_wall = False
        if not self._is_valid_move(next_state):
            next_state = self.state
            hit_wall = True
            step_reward = self.wall_penalty
        else:
            step_reward = self.step_penalty

        # 【修正】停止判定
        # 従来は hit_wall=True でも常に stop_penalty(-0.4) に上書きされてしまい、
        # wall_penalty(-0.6) が実質使われていなかった（壁の方が停止より軽く罰される逆転バグ）。
        # 壁衝突と、それ以外の理由で座標が変わらなかったケースを区別する。
        if next_state == self.state and not hit_wall:
            step_reward = self.stop_penalty
        # hit_wall=True の場合は既に self.wall_penalty が設定済みなのでそのまま使う

        # 2. 距離マップ上で近づいたかの判定（実際に動いた場合のみ計算）
        if next_state != self.state:
            cur_dist = self.dist_map[x][y]
            next_dist = self.dist_map[next_state[0]][next_state[1]]

            if cur_dist is not None and next_dist is not None:
                if next_dist < cur_dist:
                    step_reward += self.progress_reward   # ゴールに近づいた
                elif next_dist > cur_dist:
                    step_reward += self.regress_penalty    # ゴールから遠ざかった

        # 3. ゴール到達判定
        if self.maze[next_state[0]][next_state[1]] == 'G':
            reward = self.goal_reward + step_reward
            self.done = True
        else:
            reward = step_reward
            self.done = False

        self.state = next_state
        self.history.append(self.state)
        return self.state, reward, self.done

    def render(self):
        maze_copy = [row[:] for row in self.maze]
        x, y = self.state
        if not self.done:
            maze_copy[x][y] = 'A'
        for row in maze_copy:
            print(' '.join(row))
        print()

    def get_image_observation(self):
        """
        観測は5チャンネル構成:
          ch0: 壁マップ (1=壁, 0=通行可)
          ch1: スタート位置
          ch2: ゴール位置
          ch3: エージェントの現在位置
          ch4: ゴールまでのBFS距離（正規化・近いほど値が大きい）
               壁・到達不可能マスは0
        ch4を追加した狙い:
          Transformer側のattentionだけに経路探索を任せると、
          層数×1ホップ分の受容野しかゴール方向の情報が伝わらず、
          数マス先の迂回が必要な壁配置を学習しづらい。
          BFS距離を直接与えることで、各セルが「ゴールに近づく方向」を
          受容野に関係なく参照できるようにする。
        """
        rows = len(self.maze)
        cols = len(self.maze[0])
        obs = np.zeros((5, rows, cols), dtype=np.float32)

        dist_map = getattr(self, 'dist_map', None)
        max_dist = getattr(self, '_max_dist', 1) or 1

        for i in range(rows):
            for j in range(cols):
                cell = self.maze[i][j]
                if cell == 'W':
                    obs[0, i, j] = 1.0
                elif cell == 'S':
                    obs[1, i, j] = 1.0
                elif cell == 'G':
                    obs[2, i, j] = 1.0
                elif cell == '.':
                    pass

                if dist_map is not None:
                    d = dist_map[i][j]
                    if d is not None:
                        obs[4, i, j] = (max_dist - d) / max_dist

        x, y = self.state
        obs[3, x, y] = 1.0
        return obs

    def save_video(self, output_path="maze_animation.mp4", fps=2):
        rows = len(self.maze)
        cols = len(self.maze[0])

        cell_colors = {
            'S': 'lightblue',
            'G': 'lightgreen',
            'W': 'black',
            '.': 'white',
            'A': 'red'
        }

        fig, ax = plt.subplots(figsize=(cols, rows))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True)

        for i in range(rows):
            for j in range(cols):
                cell = self.maze[i][j]
                color = cell_colors.get(cell, 'white')
                rect = plt.Rectangle((j - 0.5, rows - i - 1.5), 1, 1,
                                     facecolor=color, edgecolor='gray')
                ax.add_patch(rect)
                if cell in ['S', 'G', 'W', '.']:
                    ax.text(j, rows - i - 1, cell,
                            ha='center', va='center', fontsize=12)

        agent_marker, = ax.plot([], [], 'o', markersize=20, color='red')

        def init():
            agent_marker.set_data([], [])
            return agent_marker,

        def update(frame):
            if frame >= len(self.history):
                return agent_marker,
            x, y = self.history[frame]
            plot_y = rows - x - 1
            agent_marker.set_data([y], [plot_y])
            return agent_marker,

        anim = FuncAnimation(fig, update, frames=len(self.history),
                            init_func=init, blit=True, interval=1000/fps)
        anim.save(output_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"Animation saved to {output_path}")

    def _is_reachable(self, start, goal, maze):
        rows = len(maze)
        cols = len(maze[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        queue = deque([start])
        visited = set([start])

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if maze[nx][ny] != 'W' and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def generate_random_maze(self, rows=5, cols=5):
        while True:
            maze = [['.' for _ in range(cols)] for _ in range(rows)]

            while True:
                start = (random.randint(0, rows-1), random.randint(0, cols-1))
                goal = (random.randint(0, rows-1), random.randint(0, cols-1))
                if start != goal:
                    break

            maze[start[0]][start[1]] = 'S'
            maze[goal[0]][goal[1]] = 'G'

            path = self._find_shortest_path(maze, start, goal)
            if not path:
                continue

            self._add_walls_safely(maze, path, start, goal)

            if self._is_reachable(start, goal, maze):
                self.maze = maze
                self.start = start
                self.goal = goal
                break
        return len(path)

    def _find_shortest_path(self, maze, start, goal):
        rows = len(maze)
        cols = len(maze[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        queue = deque()
        queue.append((start, [start]))
        visited = set([start])

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                return path
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return None

    def print_distance_map(self):
        """
        現在のエピソードにおけるゴールからの距離マップをターミナルに美しく表示する。
        """
        if not hasattr(self, 'dist_map') or self.dist_map is None:
            print("Distance map has not been calculated yet. Call reset() first.")
            return

        print("--- Goal Distance Map ---")
        for i, row in enumerate(self.maze):
            row_str = []
            for j, cell in enumerate(row):
                if cell == 'W':
                    # 壁は視覚的に分かりやすく表現
                    row_str.append(" [W] ")
                elif cell == 'G':
                    row_str.append(" [G] ")
                else:
                    dist = self.dist_map[i][j]
                    if dist is None:
                        row_str.append(" [X] ") # 到達不可能（孤立空間など）
                    else:
                        # 数値の桁揃え（2桁まで対応）をして表示
                        row_str.append(f" {dist:2d}  ")
            print("".join(row_str))
        print("-------------------------\n")

    def _add_walls_safely(self, maze, path, start, goal):
        rows = len(maze)
        cols = len(maze[0])
        path_set = set(path)

        candidate_cells = []
        for i in range(rows):
            for j in range(cols):
                if (i, j) not in path_set and (i, j) != start and (i, j) != goal:
                    candidate_cells.append((i, j))

        wall_density = 0.3
        num_walls = int(len(candidate_cells) * wall_density)
        random.shuffle(candidate_cells)

        walls_added = 0
        for x, y in candidate_cells:
            if walls_added >= num_walls:
                break
            maze[x][y] = 'W'
            if self._is_reachable(start, goal, maze):
                walls_added += 1
            else:
                maze[x][y] = '.'
