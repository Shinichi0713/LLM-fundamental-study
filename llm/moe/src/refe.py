import openai
import json
from typing import Dict, List, Optional

class QualityManagementAgent:
    """
    LLMを利用した品質管理エージェント（QMA）の簡易実装例
    """
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        # 品質チェックのルール（例）
        self.rules = {
            "security": ["パスワードは平文で保存しない", "SQLインジェクション対策を実施"],
            "performance": ["N+1クエリを避ける", "インデックスを適切に設定"],
            "maintainability": ["関数は単一責任", "コメントは必要最小限かつ明確"],
        }

    def check_quality(self, content: str, content_type: str = "code") -> Dict:
        """
        コンテンツ（コードやドキュメント）の品質をチェックする
        """
        # LLMへのプロンプト構築
        prompt = self._build_prompt(content, content_type)

        # LLM呼び出し
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "あなたはソフトウェア品質管理の専門家です。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )

        llm_output = response.choices[0].message.content

        # ルールベースチェック（補助）
        rule_violations = self._check_rules(content)

        # 結果の整形
        result = {
            "content_type": content_type,
            "llm_feedback": llm_output,
            "rule_violations": rule_violations,
            "overall_score": self._score_quality(llm_output, rule_violations),
        }
        return result

    def _build_prompt(self, content: str, content_type: str) -> str:
        """
        LLMへのプロンプトを構築
        """
        if content_type == "code":
            base_prompt = """
以下のコードの品質を評価し、改善提案をしてください。
評価観点：
- セキュリティ
- パフォーマンス
- 保守性
- 可読性

コード：