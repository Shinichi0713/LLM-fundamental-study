import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import unittest
import io

class MazeEnv:
    """
    5x5迷路環境クラス（動画保存機能付き）
    """
    def __init__(self, maze_file="maze.txt", rows=5, cols=5):
        if os.path.exists(maze_file):
            self.maze = self._load_maze(maze_file)
            self.start, self.goal = self._find_start_goal()
        else:
            # 自動生成メソッドを呼び出す
            self.generate_random_maze(rows=rows, cols=cols)

        self.state = self.start
        self.done = False
        self.history = []
        self.rows = rows
        self.cols = cols
        self.reward = 10
        self.reward_deteriorate = -0.001

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

    def reset(self, maze_change=True):
        if maze_change:
            length_min = self.generate_random_maze()
            self.reward = length_min * 3
        else:
            self.reward = 10
        self.state = self.start
        self.done = False
        self.history = [self.state]  # 履歴もリセット
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

        if not self._is_valid_move(next_state):
            next_state = self.state
            reward = -1
        else:
            reward = 0

        # 時間経過で減衰
        self.reward -= self.reward_deteriorate

        if self.maze[next_state[0]][next_state[1]] == 'G':
            reward = self.reward
            self.done = True
        else:
            self.done = False

        self.state = next_state
        self.history.append(self.state)  # 履歴に追加
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
        迷路を画像（NumPy配列）として返す
        チャネル0: 壁 (W) = 1, それ以外 = 0
        チャネル1: スタート (S) = 1, それ以外 = 0
        チャネル2: ゴール (G) = 1, それ以外 = 0
        チャネル3: エージェント位置 (A) = 1, それ以外 = 0
        """
        rows = len(self.maze)
        cols = len(self.maze[0])
        obs = np.zeros((4, rows, cols), dtype=np.float32)

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
                    pass  # 何も立てない
        # エージェント位置
        x, y = self.state
        obs[3, x, y] = 1.0
        return obs

    def save_video(self, output_path="maze_animation.mp4", fps=2):
        """
        エージェントの動きを動画として保存
        """
        # 迷路のサイズ
        rows = len(self.maze)
        cols = len(self.maze[0])

        # カラーマップの定義
        cell_colors = {
            'S': 'lightblue',  # スタート
            'G': 'lightgreen', # ゴール
            'W': 'black',      # 壁
            '.': 'white',      # 通路
            'A': 'red'         # エージェント（描画時に上書き）
        }

        fig, ax = plt.subplots(figsize=(cols, rows))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True)

        # 背景（迷路）を描画
        for i in range(rows):
            for j in range(cols):
                cell = self.maze[i][j]
                color = cell_colors.get(cell, 'white')
                rect = plt.Rectangle((j - 0.5, rows - i - 1.5), 1, 1,
                                     facecolor=color, edgecolor='gray')
                ax.add_patch(rect)
                # ラベル（S, G, W, .）を表示
                if cell in ['S', 'G', 'W', '.']:
                    ax.text(j, rows - i - 1, cell,
                            ha='center', va='center', fontsize=12)

        # エージェントの位置を示すマーカー
        agent_marker, = ax.plot([], [], 'o', markersize=20, color='red')

        def init():
            agent_marker.set_data([], [])
            return agent_marker,

        def update(frame):
            if frame >= len(self.history):
                return agent_marker,
            x, y = self.history[frame]
            # 座標変換（matplotlibはy軸が下向きなので反転）
            plot_y = rows - x - 1
            agent_marker.set_data([y], [plot_y])
            return agent_marker,

        anim = FuncAnimation(fig, update, frames=len(self.history),
                            init_func=init, blit=True, interval=1000/fps)

        # MP4として保存（ffmpegが必要）
        anim.save(output_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"Animation saved to {output_path}")

    def _is_reachable(self, start, goal, maze):
        """
        BFSでスタートからゴールに到達可能かチェックする。
        """
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
        """
        ランダムにスタートとゴールが設定され、必ず解ける迷路を生成する。
        """
        while True:
            # 1. すべてを通路('.')で初期化
            maze = [['.' for _ in range(cols)] for _ in range(rows)]

            # 2. スタートとゴールをランダムに配置（同一マスは避ける）
            while True:
                start = (random.randint(0, rows-1), random.randint(0, cols-1))
                goal = (random.randint(0, rows-1), random.randint(0, cols-1))
                if start != goal:
                    break

            maze[start[0]][start[1]] = 'S'
            maze[goal[0]][goal[1]] = 'G'

            # 3. スタートからゴールまでの最短経路を1本確保（BFS）
            #    この経路上のマスは壁にしない（連結性の保証）。
            path = self._find_shortest_path(maze, start, goal)
            # print(path)
            if not path:
                # 万が一経路がなければ再試行（空迷路なので通常は起こらない）
                continue

            # 4. 経路を壊さない範囲でランダムに壁を追加
            #    迷路全体の連結性を維持するようにする。
            self._add_walls_safely(maze, path, start, goal)

            # 5. 最終的な到達可能性チェック（念のため）
            if self._is_reachable(start, goal, maze):
                self.maze = maze
                self.start = start
                self.goal = goal
                break
        return len(path)

    def _find_shortest_path(self, maze, start, goal):
        """
        BFSでスタートからゴールまでの最短経路を1本見つける。
        ここでは迷路はすべて通路なので、単純なグリッドBFSでOK。
        """
        rows = len(maze)
        cols = len(maze[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右, 下, 左, 上

        queue = deque()
        queue.append((start, [start]))  # (現在位置, 経路)
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

        return None  # 経路なし（空迷路なので通常は起こらない）

    def _add_walls_safely(self, maze, path, start, goal):
        """
        経路を壊さない範囲でランダムに壁を追加する。
        迷路全体の連結性を維持するようにする。
        """
        rows = len(maze)
        cols = len(maze[0])
        path_set = set(path)  # 経路上のマスは壁にしない

        # 壁候補のマスを列挙（経路上とスタート/ゴールは除外）
        candidate_cells = []
        for i in range(rows):
            for j in range(cols):
                if (i, j) not in path_set and (i, j) != start and (i, j) != goal:
                    candidate_cells.append((i, j))

        # ランダムに壁を追加（連結性チェック付き）
        # ここでは簡易的に「壁の密度」を調整（例：候補の30%を壁にする）
        wall_density = 0.3
        num_walls = int(len(candidate_cells) * wall_density)
        random.shuffle(candidate_cells)

        walls_added = 0
        for x, y in candidate_cells:
            if walls_added >= num_walls:
                break

            # 仮に壁を置いてみる
            maze[x][y] = 'W'

            # 連結性チェック（BFSでスタートからゴールに到達可能か）
            if self._is_reachable(start, goal, maze):
                walls_added += 1
            else:
                # 連結性が失われる場合は壁を戻す
                maze[x][y] = '.'

    def render_image(self, show=True, save_path=None):
        """
        迷路を画像（matplotlib）で表示する
        - show=True: 画面に表示
        - save_path: 指定があれば画像を保存
        """
        rows = len(self.maze)
        cols = len(self.maze[0])

        # カラーマップの定義
        cell_colors = {
            'S': 'lightblue',  # スタート
            'G': 'lightgreen', # ゴール
            'W': 'black',      # 壁
            '.': 'white',      # 通路
            'A': 'red'         # エージェント（描画時に上書き）
        }

        fig, ax = plt.subplots(figsize=(cols, rows))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True)

        # 迷路の背景を描画
        for i in range(rows):
            for j in range(cols):
                cell = self.maze[i][j]
                color = cell_colors.get(cell, 'white')
                # matplotlib は y 軸が下向きなので反転
                rect = plt.Rectangle((j - 0.5, rows - i - 1.5), 1, 1,
                                     facecolor=color, edgecolor='gray')
                ax.add_patch(rect)
                # ラベル（S, G, W, .）を表示
                if cell in ['S', 'G', 'W', '.']:
                    ax.text(j, rows - i - 1, cell,
                            ha='center', va='center', fontsize=12)

        # エージェント位置を描画
        x, y = self.state
        plot_y = rows - x - 1
        ax.plot(y, plot_y, 'o', markersize=20, color='red', label='Agent')

        ax.set_title("Maze (Agent = Red dot)")
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Image saved to {save_path}")
        plt.close(fig)

    def print_maze(self):
        """
        迷路をテキストで表示（確認用）
        """
        for row in self.maze:
            print(' '.join(row))
        print()


    def render_maze_image(self, show=True, save_path="random_maze.png"):
        """
        生成されたランダム迷路を画像として可視化する。
        - show=True: 画面に表示
        - save_path: 画像保存先パス
        """
        rows = self.rows
        cols = self.cols
        maze = self.maze

        # カラーマップの定義
        cell_colors = {
            'S': 'lightblue',  # スタート
            'G': 'lightgreen', # ゴール
            'W': 'black',      # 壁
            '.': 'white',      # 通路
        }

        fig, ax = plt.subplots(figsize=(cols, rows))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True)

        # 迷路の背景を描画
        for i in range(rows):
            for j in range(cols):
                cell = maze[i][j]
                color = cell_colors.get(cell, 'white')
                # matplotlib は y 軸が下向きなので反転
                rect = plt.Rectangle((j - 0.5, rows - i - 1.5), 1, 1,
                                     facecolor=color, edgecolor='gray')
                ax.add_patch(rect)
                # ラベル（S, G, W, .）を表示
                if cell in ['S', 'G', 'W', '.']:
                    ax.text(j, rows - i - 1, cell,
                            ha='center', va='center', fontsize=12)

        ax.set_title("Random Maze (S=Start, G=Goal, W=Wall)")
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Maze image saved to {save_path}")
        plt.close(fig)


class TestMaze(unittest.TestCase):
    def setUp(self):
        self.env = MazeEnv()

    def test_reward(self):
        self.env.reset()
        _, reward, _ = self.env.step(2)  # 左
        self.assertEqual(reward, -1, "壁にぶつかったときの報酬は -1 であるべき")

    def test_reward_goal_rearch(self):
        self.env.reset()
        self.env.state = (self.env.goal[0] - 1, self.env.goal[1])
        self.env.done = False
        _, reward, done = self.env.step(1)  # 右 → ゴールへ
        self.assertEqual(reward, 10, "ゴール到達時の報酬は +10 であるべき")
        self.assertTrue(done, "ゴール到達時は done=True であるべき")

    def save_video(self, output_path="maze_animation.mp4", fps=2):
        """
        エージェントの動きを動画として保存
        """
        # 迷路のサイズ
        rows = len(self.maze)
        cols = len(self.maze[0])

        # カラーマップの定義
        cell_colors = {
            'S': 'lightblue',  # スタート
            'G': 'lightgreen', # ゴール
            'W': 'black',      # 壁
            '.': 'white',      # 通路
            'A': 'red'         # エージェント（描画時に上書き）
        }

        fig, ax = plt.subplots(figsize=(cols, rows))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True)

        # 背景（迷路）を描画
        for i in range(rows):
            for j in range(cols):
                cell = self.maze[i][j]
                color = cell_colors.get(cell, 'white')
                rect = plt.Rectangle((j - 0.5, rows - i - 1.5), 1, 1,
                                     facecolor=color, edgecolor='gray')
                ax.add_patch(rect)
                # ラベル（S, G, W, .）を表示
                if cell in ['S', 'G', 'W', '.']:
                    ax.text(j, rows - i - 1, cell,
                            ha='center', va='center', fontsize=12)

        # エージェントの位置を示すマーカー
        agent_marker, = ax.plot([], [], 'o', markersize=20, color='red')

        def init():
            agent_marker.set_data([], [])
            return agent_marker,

        def update(frame):
            if frame >= len(self.history):
                return agent_marker,
            x, y = self.history[frame]
            # 座標変換（matplotlibはy軸が下向きなので反転）
            plot_y = rows - x - 1
            agent_marker.set_data([y], [plot_y])
            return agent_marker,

        anim = FuncAnimation(fig, update, frames=len(self.history),
                            init_func=init, blit=True, interval=1000/fps)

        # MP4として保存（ffmpegが必要）
        anim.save(output_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"Animation saved to {output_path}")

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMaze)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

