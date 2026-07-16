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

"""
「ddeque」という名前の独立したライブラリは、一般的にはあまり知られておらず、Python 標準ライブラリの **`collections.deque`**（両端キュー）を指している可能性が高いです。

---

## 1. `collections.deque` とは

`deque`（発音は「デック」）は、Python の標準ライブラリ `collections` モジュールに含まれる**両端キュー（double-ended queue）**です。

- **両端から要素の追加・削除が高速にできる**データ構造です。
- 内部的には**双方向連結リスト**で実装されており、先頭・末尾の追加・削除が O(1) で行えます。
- リストと似たインターフェースを持ちますが、**先頭側の操作が高速**な点が大きな違いです。

```python
from collections import deque

dq = deque([1, 2, 3])
dq.append(4)        # 末尾に追加 → [1, 2, 3, 4]
dq.appendleft(0)    # 先頭に追加 → [0, 1, 2, 3, 4]
x = dq.pop()        # 末尾から取り出し → x = 4
y = dq.popleft()    # 先頭から取り出し → y = 0
```

---

## 2. `deque` の主な特徴・用途

### 2.1 両端からの高速な操作

- `append(x)` / `appendleft(x)`  
- `pop()` / `popleft()`

これらがすべて O(1) で動作するため、  
- **キュー（FIFO）**  
- **スタック（LIFO）**  
- **両端キュー（deque）**  

として効率的に使えます[Pythonのdequeでキュー、スタック、デック（両端キュー）を ...](https://note.nkmk.me/python-collections-deque)。

### 2.2 最大長を指定できる（リングバッファ）

`maxlen` を指定すると、**固定長のリングバッファ**として使えます。

```python
dq = deque(maxlen=3)
dq.append(1)  # [1]
dq.append(2)  # [1, 2]
dq.append(3)  # [1, 2, 3]
dq.append(4)  # [2, 3, 4]  （先頭の 1 が押し出される）
```

直近 N 件の履歴を保持するバッファなどに便利です。

### 2.3 回転（rotate）など便利なメソッド

- `rotate(n)`：要素を n ステップ分回転させる  
  - `rotate(1)` で右に1つずらす  
  - `rotate(-1)` で左に1つずらす

---

## 3. リストとの比較

| 操作 | リスト（list） | deque |
|------|----------------|-------|
| 末尾に追加 | O(1) | O(1) |
| 先頭に追加 | O(n) | O(1) |
| 末尾から削除 | O(1) | O(1) |
| 先頭から削除 | O(n) | O(1) |
| 中央への挿入・削除 | O(n) | O(n) |

**先頭側の操作が多い場合（キューなど）は `deque` の方が高速**です[Python's deque: Implement Efficient Queues and Stacks – Real Python](https://realpython.com/python-deque)。

---

## 4. 「ddeque」という名前について

「ddeque」という名前の独立したライブラリは、一般的な Python エコシステムではあまり見かけません。  
おそらく以下のいずれかの可能性があります。

1. **`collections.deque` の typo または略記**  
   - `ddeque` → `deque`（d が重なっているだけ）  
   - あるいは「double deque」のような意味で使われている

2. **特定のプロジェクトや独自ライブラリ内でのクラス名**  
   - 例えば「MyDoubleDeque」のようなクラスを `ddeque` と略しているケース

もし「○○というライブラリの中の `ddeque` クラス」など、具体的な文脈があれば、そのライブラリ名を教えていただければ、より正確に説明できます。

---

## 5. まとめ

- 一般的に「ddeque」という名前のライブラリはなく、Python 標準の **`collections.deque`** を指している可能性が高いです。
- `deque` は**両端キュー**で、先頭・末尾の追加・削除が高速に行えるデータ構造です。
- キュー・スタック・リングバッファなど、**先頭側の操作が多い用途**でリストより有利です。

もし「このライブラリの `ddeque` について知りたい」という形で、具体的なライブラリ名やコード断片があれば、それに合わせてより詳しく説明できます。


「ライブラリ」と「パッケージ」は、文脈によって意味が少し変わりますが、**ざっくり言うと「ライブラリ」は再利用可能なコードの集まり、「パッケージ」はその配布単位や構造単位**と考えると分かりやすいです。

---

## 1. 一般的な違い（概念レベル）

### ライブラリ（Library）
- **再利用可能なコードの集まり**（関数・クラス・モジュールなど）
- 目的は「特定の機能を提供すること」
- 例：数値計算ライブラリ、画像処理ライブラリ、HTTP クライアントライブラリなど

### パッケージ（Package）
- **コードをまとめて配布・管理する単位**
- ファイル構造や配布形式（ディレクトリ構成、メタデータなど）を含む
- 例：Python のパッケージ（`setup.py` / `pyproject.toml` 付き）、npm パッケージ、Java の JAR ファイルなど

**イメージ**：  
- ライブラリ ＝ 「本の内容（機能）」  
- パッケージ ＝ 「本＋表紙＋ISBN＋出版社情報（配布単位）」

---

## 2. Python での具体的な違い

### 2.1 モジュール（Module）
- 1つの `.py` ファイル  
- 例：`math.py`（Python 標準の math モジュール）

### 2.2 パッケージ（Package）
- `__init__.py` を含むディレクトリ  
- 複数のモジュールをまとめたもの  
- 例：
  ```
  mypkg/
    __init__.py
    module1.py
    module2.py
    subpkg/
      __init__.py
      module3.py
  ```
  この `mypkg` がパッケージです。

### 2.3 ライブラリ（Library）
- Python では**厳密な定義はありません**が、一般的には：
  - 標準ライブラリ（`os`, `sys`, `json` など）
  - サードパーティライブラリ（`requests`, `numpy`, `pandas` など）
- これらは「パッケージ」として配布されていることが多いです。

**Python での関係**：  
モジュール（.py）  
↓ まとめる  
パッケージ（ディレクトリ＋`__init__.py`）  
↓ 配布・利用  
ライブラリ（機能提供の単位として呼ばれる）

---

## 3. 他の言語・エコシステムでの違い（参考）

### npm（Node.js）
- **パッケージ**：`package.json` を持つディレクトリ（モジュールの配布単位）
- **ライブラリ**：再利用可能なコードの集まり（React ライブラリ、Lodash ライブラリなど）
- ほぼ「パッケージ ＝ 配布単位」「ライブラリ ＝ 機能の集まり」という関係です。

### Java
- **ライブラリ**：`.class` ファイルやソースコードの集まり
- **パッケージ**：名前空間を分けるためのディレクトリ構造（`com.example.mylib` など）
- **JAR**：Java Archive（パッケージをまとめた配布ファイル）

---

## 4. まとめ

- **ライブラリ**：  
  - 再利用可能なコードの集まり  
  - 「何ができるか」に焦点

- **パッケージ**：  
  - コードをまとめて配布・管理する単位  
  - 「どう配布・構造化されているか」に焦点

Python では、  
- 1ファイル → モジュール  
- 複数ファイルをディレクトリにまとめる → パッケージ  
- それらをまとめて機能提供する単位として → ライブラリ  

と考えると分かりやすいです。

もし「この言語ではどうか？」「このライブラリはパッケージと呼ぶべきか？」など、具体的な例があれば、その文脈に合わせて説明できます。

強化学習の**経験再生（Experience Replay）**用メモリバッファで `deque` がよく使われるのは、主に次の理由があります。

---

## 1. 経験再生バッファの典型的な使い方

DQN などで使われる経験再生では、タプル  
`(状態 s, 行動 a, 報酬 r, 次状態 s', 終了フラグ done)`  
をバッファに貯め、学習時にランダムにサンプリングします。

典型的な操作は：

- **末尾への追加（append）**：新しい経験を追加
- **先頭からの削除（popleft）**：バッファが一杯になったとき、古い経験を捨てる
- **ランダムアクセス**：ミニバッチ学習でランダムにサンプリング

このとき、**「末尾への追加」と「先頭からの削除」が頻繁に発生**します。

---

## 2. deque が向いている理由

### 2.1 先頭・末尾の操作が O(1)

`deque` は双方向連結リストとして実装されているため：

- `append(x)`：末尾への追加が O(1)
- `popleft()`：先頭からの削除が O(1)

一方、リスト（`list`）で同じことをすると：

- `append(x)`：O(1)（償却）
- `pop(0)`：先頭からの削除は O(n)（後続要素をすべてずらす必要がある）

**経験再生バッファは FIFO（先入れ先出し）的に使われることが多く、先頭側の削除が頻発するため、deque の方が効率的**です。

### 2.2 リングバッファとして使える

`deque(maxlen=N)` とすると、**固定長のリングバッファ**として使えます。

```python
from collections import deque

buffer = deque(maxlen=10000)

# 追加するだけで、自動的に古い経験が押し出される
buffer.append(experience)
```

- バッファサイズを超えたとき、**自動的に古い経験が捨てられる**  
- 自分で `if len(buffer) > maxlen: buffer.popleft()` と書く必要がなく、実装がシンプル

---

## 3. リストを使う場合の問題点

もしリストで同じことをしようとすると：

```python
buffer = []

buffer.append(experience)        # 末尾追加は O(1)
if len(buffer) > maxlen:
    buffer.pop(0)                # 先頭削除は O(n)
```

- バッファサイズが大きいと、`pop(0)` のコストが無視できなくなる  
- 特に DQN ではバッファサイズが数万〜数十万になることもあり、**先頭削除の O(n) がボトルネック**になり得ます

---

## 4. ランダムアクセスについて

経験再生では、ミニバッチ学習のためにランダムに経験をサンプリングします。

- `deque` はランダムアクセスが O(n)（最悪）ですが、実際には  
  - バッファはメモリ上でブロック単位にまとまっている  
  - Python 実装も効率化されている  
  ため、**実用上は十分高速**です。
- もしランダムアクセスが極端に多い場合は、`list` や `numpy` 配列を使う選択肢もありますが、多くの DQN 実装では `deque` で十分です。

---

## 5. まとめ

強化学習のメモリバッファで `deque` がよく使われる理由は：

1. **経験再生バッファが FIFO 的に使われる**（末尾追加・先頭削除）  
2. **deque は先頭・末尾の操作が O(1) で高速**  
3. **`maxlen` 指定でリングバッファとして簡単に使える**  
4. **リストの `pop(0)` は O(n) で、バッファが大きいと重くなる**

という特性のマッチングによるものです。

実装例としては、多くの DQN のチュートリアルや書籍で `collections.deque` が使われています（PyTorch の公式チュートリアルなども同様です）。
"""