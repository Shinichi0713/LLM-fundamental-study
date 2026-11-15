"""
============================================================
Simple AI Agent in Pure Python (Standard Library)
============================================================
ReAct (Reasoning + Acting) パターンを用いた最小構成のAIエージェントです。
外部ライブラリ不要で動作します。

使い方:
    python simple_agent.py

機能:
    - ユーザーから目標（タスク）を受け取る
    - 思考 → 行動 → 観察 のループで自律的にタスクを進める
    - 電卓、ファイル読み込み、現在時刻取得 の3つのツールを使用可能
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Callable, Dict, List


# ============================================================
# ツール定義
# ============================================================

def tool_calculator(expression: str) -> str:
    """数式を計算して結果を返すツール"""
    try:
        # 安全のため、evalに渡す式を制限
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "エラー: 使用可能な文字は 0-9 + - * / . ( ) スペース のみです"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"計算エラー: {e}"


def tool_read_file(filepath: str) -> str:
    """指定されたファイルの内容を読み込むツール"""
    try:
        if not os.path.exists(filepath):
            return f"エラー: ファイル '{filepath}' が見つかりまん"
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # 長すぎる場合は先頭500文字のみ返す
        if len(content) > 500:
            content = content[:500] + "\n... (以下省略)"
        return content
    except Exception as e:
        return f"読み込みエラー: {e}"


def tool_get_time(_: str = "") -> str:
    """現在の日時を返すツール"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ツールレジストリ
TOOLS: Dict[str, Callable[[str], str]] = {
    "calculator": tool_calculator,
    "read_file": tool_read_file,
    "get_time": tool_get_time,
}

TOOL_DESCRIPTIONS = {
    "calculator": "数式を計算します。引数: expression (例: '2 + 3 * 4')",
    "read_file": "ファイルの内容を読み込みます。引数: filepath (例: 'data.txt')",
    "get_time": "現在の日時を取得します。引数: 不要",
}


# ============================================================
# エージェント本体
# ============================================================

class SimpleAgent:
    """
    ReActパターンを模したシンプルな自律エージェント。
    実際のLLMの代わりに、ルールベースのパーサーで思考と行動をシミュレートします。
    """

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.memory: List[str] = []

    def think(self, goal: str) -> str:
        """
        目標に対して次の行動を決定する（シミュレート）。
        実際のLLMではなく、キーワードに基づいて行動を選択します。
        """
        goal_lower = goal.lower()
        memory_text = "\n".join(self.memory)

        # ステップ1: ファイル読み込みが必要か？
        if any(k in goal_lower for k in ["ファイル", "読む", "read", "内容", "中身"]) and "read_file" not in memory_text:
            # ファイル名を推測
            match = re.search(r"[\w\-\.]+\.(txt|md|csv|json|py)", goal)
            filename = match.group(0) if match else "sample.txt"
            return f'{{"thought": "ファイルの内容を確認する必要があります", "action": "read_file", "action_input": "{filename}"}}'

        # ステップ2: 計算が必要か？
        if any(k in goal_lower for k in ["計算", "足す", "引く", "掛ける", "割る", "+", "-", "*", "/", "合計", "平"]) and "calculator" not in memory_text:
            # 数式を推測
            numbers = re.findall(r"\d+", goal)
            if len(numbers) >= 2:
                expr = f"{numbers[0]} + {numbers[1]}"
                return f'{{"thought": "計算が必要です", "action": "calculator", "action_input": "{expr}"}}'

        # ステップ3: 時刻が必要か？
        if any(k in goal_lower for k in ["時間", "時刻", "日時", "今何時", "today", "now", "time"]) and "get_time" not in memory_text:
            return '{"thought": "現在時刻を確認します", "action": "get_time", "action_input": ""}'

        # ステップ4: 終了判定
        return '{"thought": "必要な情報が揃いました。タスクを完了します", "action": "finish", "action_input": ""}'

    def parse_action(self, thought_json: str) -> tuple:
        """JSON形式の思考文字列から行動を抽出"""
        try:
            data = json.loads(thought_json)
            return data.get("action", ""), data.get("action_input", ""), data.get("thought", "")
        except json.JSONDecodeError:
            return "", "", thought_json

    def run(self, goal: str) -> str:
        """
        目標に向かってステップを繰り返し、最終回答を返す。
        """
        print("=" * 60)
        print(f"🎯 目標: {goal}")
        print("=" * 60)

        for step in range(1, self.max_steps + 1):
            print(f"\n--- ステップ {step} ---")

            # 思考
            thought_json = self.think(goal)
            action, action_input, thought = self.parse_action(thought_json)

            print(f"💭 思考: {thought}")
            print(f"🔧 行動: {action}({action_input})")

            if action == "finish":
                print("\n✅ タスク完了")
                break

            # 行動実行
            if action in TOOLS:
                observation = TOOLS[action](action_input)
            else:
                observation = f"未知のツール: {action}"

            print(f"📊 観察: {observation}")
            self.memory.append(f"Step{step}: {action}({action_input}) -> {observation}")

        final_memory = "\n".join(self.memory)
        print("\n" + "=" * 60)
        print("📋 実行ログ:")
        print(final_memory if final_memory else "(ログなし)")
        print("=" * 60)
        return final_memory


# ============================================================
# メイン処理
# ============================================================

if __name__ == "__main__":
    agent = SimpleAgent(max_steps=10)

    # サンプルタスク
    sample_goals = [
        "sample.txt のファイル内容を読んで、現在の時刻も教えて",
        "23 + 47 を計算して",
        "今の日時を教えて",
    ]

    # サンプルファイルを作成
    with open("sample.txt", "w", encoding="utf-8") as f:
        f.write("これはサンプルファイルです。\nAIエージェントのテスト用データ。\n")

    for goal in sample_goals:
        agent.memory.clear()
        agent.run(goal)
        print("\n")
