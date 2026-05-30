import openai
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt

openai.api_key = os.getenv("OPENAI_API_KEY")

class SimpleGraphRAG:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def extract_entities(self, question: str) -> list[str]:
        """
        簡易的なエンティティ抽出（LLM使用）
        Colabで無料モデルを使いたい場合は、後述の代替案を参照
        """
        prompt = f"""
        以下の質問から、病気や治療法などの固有名詞・重要な概念（エンティティ）を抽出し、カンマ区切りで列挙してください。
        例: 「糖尿病と高血圧の関係は？」 → Diabetes,Hypertension

        質問: {question}
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            entities_text = response.choices[0].message.content.strip()
            return [e.strip() for e in entities_text.split(",")]
        except Exception as e:
            # APIキーがない場合のフォールバック（手動入力）
            print("OpenAI APIが利用できないため、手動でエンティティを入力してください")
            manual = input("エンティティをカンマ区切りで入力（例: Diabetes,Hypertension）: ")
            return [e.strip() for e in manual.split(",")]

    def query_relation_paths(self, entities: list[str]) -> list[dict]:
        """
        関係パスを取得（GraphDBクエリ）
        """
        if len(entities) < 2:
            return []

        query = """
        MATCH path = (a)-[r*1..3]-(b)
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
あなたは医療専門家アシスタントです。
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
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLMエラー] プロンプト内容:\n{prompt}"

    def run(self, question: str) -> str:
        """
        メインフロー: エンティティ抽出 → Graphクエリ → LLM回答
        """
        # 1. エンティティ抽出
        entities = self.extract_entities(question)
        print(f"抽出エンティティ: {entities}")

        # 2. GraphDBから関係パスを取得
        graph_info = self.query_relation_paths(entities)
        print(f"取得パス数: {len(graph_info)}")

        # 3. プロンプト構築
        prompt = self.build_prompt(question, graph_info)

        # 4. LLMに投げる
        answer = self.ask_llm(prompt)
        return answer

    def visualize_community(self):
        """
        簡易的なコミュニティ可視化（NetworkXで描画）
        """
        query = """
        MATCH (n)-[r]-(m)
        RETURN n.name AS from, m.name AS to, r.type AS type
        """
        with self.driver.session() as session:
            result = session.run(query)
            G = nx.Graph()
            for record in result:
                G.add_edge(record["from"], record["to"], label=record["type"])

        # 簡易コミュニティ色分け（ノード名の先頭文字で色分け）
        colors = []
        for node in G.nodes():
            if node.startswith("D"):
                colors.append("lightblue")  # Disease
            elif node.startswith("T"):
                colors.append("lightgreen") # Treatment
            else:
                colors.append("yellow")

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2000, font_size=10)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("GraphRAG PoC: 医療知識グラフ（簡易コミュニティ可視化）")
        plt.show()