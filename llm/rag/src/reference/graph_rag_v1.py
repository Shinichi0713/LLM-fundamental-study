import os
from neo4j import GraphDatabase
import openai

# 環境変数から設定を取得（各自設定してください）
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

class GraphRAG:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def extract_entities(self, question: str) -> list[str]:
        """
        簡易的なエンティティ抽出（ここではLLMを使う例）
        実際にはNERモデルやルールベースでも可
        """
        prompt = f"""
        以下の質問から、固有名詞や重要な概念（エンティティ）を抽出し、カンマ区切りで列挙してください。
        例: 「AとBの関係は？」 → A,B

        質問: {question}
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )
        entities_text = response.choices[0].message.content.strip()
        return [e.strip() for e in entities_text.split(",")]

    def query_graph(self, entities: list[str]) -> list[dict]:
        """
        GraphDBから関係パスを取得する
        """
        if len(entities) < 2:
            return []

        # 簡易的なCypherクエリ（1〜3ホップのパスを取得）
        query = """
        MATCH path = (a:Entity)-[r*1..3]-(b:Entity)
        WHERE a.name IN $entities AND b.name IN $entities AND a <> b
        RETURN nodes(path) AS nodes, relationships(path) AS rels
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, entities=entities)
            paths = []
            for record in result:
                nodes = [node["name"] for node in record["nodes"]]
                rels = []
                for rel in record["rels"]:
                    rels.append({
                        "from": rel.start_node["name"],
                        "to": rel.end_node["name"],
                        "type": rel.type,
                        "desc": rel.get("desc", ""),
                    })
                paths.append({"nodes": nodes, "relationships": rels})
            return paths

    def build_prompt(self, question: str, graph_info: list[dict]) -> str:
        """
        Graph情報をLLMプロンプトに埋め込む
        """
        graph_text = ""
        for i, path in enumerate(graph_info, 1):
            graph_text += f"【パス{i}】\n"
            graph_text += f"- ノード: {', '.join(path['nodes'])}\n"
            for rel in path["relationships"]:
                graph_text += f"  * {rel['from']} → {rel['to']} ({rel['type']}): {rel['desc']}\n"
            graph_text += "\n"

        prompt = f"""
あなたは専門家アシスタントです。
以下のグラフ情報をもとに、ユーザーの質問に答えてください。

【グラフ情報】
{graph_text if graph_text else "（グラフ情報はありません）"}

【ユーザー質問】
{question}

【指示】
- グラフ情報に基づいて、関係を正確に説明してください。
- グラフにない関係は推測せず、「情報がありません」と答えてください。
- 回答は日本語で、簡潔に。
"""
        return prompt

    def ask_llm(self, prompt: str) -> str:
        """
        LLMに質問を投げる
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response.choices[0].message.content

    def run(self, question: str) -> str:
        """
        メインフロー: エンティティ抽出 → Graphクエリ → LLM回答
        """
        # 1. エンティティ抽出
        entities = self.extract_entities(question)
        print(f"抽出エンティティ: {entities}")

        # 2. GraphDBから関係パスを取得
        graph_info = self.query_graph(entities)
        print(f"取得パス数: {len(graph_info)}")

        # 3. プロンプト構築
        prompt = self.build_prompt(question, graph_info)

        # 4. LLMに投げる
        answer = self.ask_llm(prompt)
        return answer

# 実行例
if __name__ == "__main__":
    rag = GraphRAG()
    try:
        question = "AとBの関係を説明してください"
        answer = rag.run(question)
        print("=== 回答 ===")
        print(answer)
    finally:
        rag.close()