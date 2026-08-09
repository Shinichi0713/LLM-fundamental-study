"""
苦手な迷路を優先的に出題するための迷路プール。

考え方（Prioritized Level Replayの簡易版）:
  - 固定サイズの迷路プールを保持する
  - 各迷路ごとに直近の成功/失敗履歴を記録する
  - 「苦手度スコア = 1 - 直近成功率」に比例した確率でプールから出題する
    -> 正解率が低い迷路ほど頻繁に出題される
  - 一定確率でプール外の完全新規ランダム迷路も混ぜる
    -> プール内の固定パターンだけに過適合するのを防ぎ、
       「その迷路の形」ではなく「壁の避け方」自体を汎化して学習させる
  - プールが満杯なら、最も得意（成功率が高い）迷路を追い出して入れ替える
"""

import random
from collections import deque


class MazePool:
    def __init__(
        self,
        pool_size=100,
        history_len=10,
        new_maze_prob=0.2,
        min_score_floor=0.05,
    ):
        """
        pool_size: プールに保持する迷路の最大数
        history_len: 1迷路あたり直近何エピソード分の成功/失敗を見るか
        new_maze_prob: プールが満杯後も、この確率で完全新規のランダム迷路を注入する
        min_score_floor: 成功率100%の迷路でも出題確率が完全に0にならないようにする下駄
        """
        self.pool_size = pool_size
        self.history_len = history_len
        self.new_maze_prob = new_maze_prob
        self.min_score_floor = min_score_floor
        self.entries = []
        self._next_id = 0

    def _make_entry(self, maze, start, goal):
        entry = {
            "id": self._next_id,
            "maze": [row[:] for row in maze],
            "start": start,
            "goal": goal,
            "history": deque(maxlen=self.history_len),
            "attempts": 0,
        }
        self._next_id += 1
        return entry

    def add(self, maze, start, goal):
        """新しい迷路をプールに追加する。満杯なら最も得意な迷路と入れ替える。"""
        entry = self._make_entry(maze, start, goal)
        if len(self.entries) < self.pool_size:
            self.entries.append(entry)
        else:
            def easiness(e):
                # 未試行のものは追い出さない（-1で最下位扱い）
                if e["attempts"] == 0 or len(e["history"]) == 0:
                    return -1.0
                return sum(e["history"]) / len(e["history"])

            idx = max(range(len(self.entries)), key=lambda i: easiness(self.entries[i]))
            self.entries[idx] = entry
        return entry

    def _score(self, entry):
        """苦手度スコア。未試行の迷路は最優先(1.0)で試す。"""
        if entry["attempts"] == 0 or len(entry["history"]) == 0:
            return 1.0
        success_rate = sum(entry["history"]) / len(entry["history"])
        return max(self.min_score_floor, 1.0 - success_rate)

    def sample(self):
        """苦手度スコアに比例した確率でプールから1つ選ぶ"""
        if not self.entries:
            return None
        scores = [self._score(e) for e in self.entries]
        total = sum(scores)
        weights = [s / total for s in scores]
        return random.choices(self.entries, weights=weights, k=1)[0]

    def record_result(self, entry, success):
        entry["attempts"] += 1
        entry["history"].append(1 if success else 0)

    def is_full(self):
        return len(self.entries) >= self.pool_size

    def should_inject_new(self):
        """新規ランダム迷路をこのタイミングで注入すべきか"""
        if not self.is_full():
            return True  # プールが埋まるまでは常に新規追加
        return random.random() < self.new_maze_prob

    def stats(self, top_k=5):
        """
        診断用の要約。全体の平均成功率と、最も苦手な上位top_k件を返す。
        """
        tried = [e for e in self.entries if e["attempts"] > 0]
        if not tried:
            return {"num_tried": 0, "overall_success_rate": None, "hardest": []}

        overall = sum(sum(e["history"]) for e in tried) / sum(len(e["history"]) for e in tried)
        hardest = sorted(
            tried,
            key=lambda e: sum(e["history"]) / len(e["history"]),
        )[:top_k]
        hardest_summary = [
            {
                "id": e["id"],
                "attempts": e["attempts"],
                "recent_success_rate": sum(e["history"]) / len(e["history"]),
            }
            for e in hardest
        ]
        return {
            "num_tried": len(tried),
            "pool_size": len(self.entries),
            "overall_success_rate": overall,
            "hardest": hardest_summary,
        }
