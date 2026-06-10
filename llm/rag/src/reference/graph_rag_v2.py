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



import re
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =====================================================================
# 1. 文章のチャンキング（簡易版）
# =====================================================================
def chunk_text(text, max_chars=200):
    """
    文章を一定文字数でチャンクに分割する（簡易実装）
    実際には文境界や意味単位で分割するのが望ましい。
    """
    # 文分割（. ! ? で区切る）
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        # 現在のチャンクに文を足したときの長さを確認
        if len(current_chunk) + len(sent) <= max_chars:
            current_chunk += " " + sent if current_chunk else sent
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# =====================================================================
# 2. チャンクからトリプレットを抽出（ルールベース or LLM）
# =====================================================================
def extract_triplets_from_chunk_rule_based(chunk):
    """
    ルールベースで簡易的にトリプレットを抽出する例
    実際にはLLMや依存関係解析を使うのが望ましい。
    """
    # 例: "A is a B", "A uses B", "A is created by B" などのパターン
    patterns = [
        # "A is a B"
        (r'([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\s+is\s+a\s+([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)', "is a"),
        # "A uses B"
        (r'([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\s+uses\s+([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)', "uses"),
        # "A is created by B"
        (r'([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\s+is\s+created\s+by\s+([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)', "created by"),
        # "A provides B"
        (r'([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\s+provides\s+([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)', "provides"),
    ]
    
    triplets = []
    for pattern, relation in patterns:
        matches = re.finditer(pattern, chunk)
        for match in matches:
            subject = match.group(1).strip()
            obj = match.group(2).strip()
            triplets.append((subject, obj, relation))
    
    return triplets


def extract_triplets_from_chunk_llm(chunk, llm):
    """
    LLMを使ってチャンクからトリプレットを抽出する例
    """
    prompt = f"""
[Instruction]
You are a triple extractor. Given a sentence, extract all subject-predicate-object triplets in the form:
(subject, object, relation)

[Example]
Input: "Google Colab is a cloud-based Python development environment created by Google."
Output:
- ("Google Colab", "cloud-based Python development environment", "is a")
- ("Google Colab", "Google", "created by")

[Input Sentence]
{chunk}

[Output]
"""
    
    outputs = llm(prompt, max_new_tokens=150, temperature=0.0)
    generated_text = outputs[0]["generated_text"]
    
    # 生成されたテキストからトリプレットをパース（簡易実装）
    # 実際には正規表現やJSON出力など、より堅牢なパースを推奨
    triplets = []
    lines = generated_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("- (") or line.startswith("("):
            # 例: - ("A", "B", "relation") または ("A", "B", "relation")
            parts = line.strip(" -").strip("()").split(",")
            if len(parts) >= 3:
                subject = parts[0].strip().strip('"')
                obj = parts[1].strip().strip('"')
                relation = parts[2].strip().strip('"')
                triplets.append((subject, obj, relation))
    
    return triplets


# =====================================================================
# 3. トリプレットからGraphRAG用DBを構築
# =====================================================================
def build_graph_from_triplets(triplets, embed_model):
    """
    トリプレットのリストからGraphRAG用のナレッジグラフDBを構築
    """
    graph_db = nx.DiGraph()
    
    for subject, obj, relation in triplets:
        # ノードが未登録なら追加し、埋め込みを計算
        if not graph_db.has_node(subject):
            graph_db.add_node(subject, embedding=embed_model.encode(subject))
        if not graph_db.has_node(obj):
            graph_db.add_node(obj, embedding=embed_model.encode(obj))
        
        # エッジ（関係性）を追加
        graph_db.add_edge(subject, obj, relation=relation, embedding=embed_model.encode(relation))
    
    return graph_db


# =====================================================================
# 4. 全体パイプライン（文章 → チャンク → トリプレット → Graph DB）
# =====================================================================
def build_graphrag_db_from_texts(texts, embed_model, use_llm=False, llm=None):
    """
    複数の文章からGraphRAG用のナレッジグラフDBを構築するパイプライン
    """
    all_triplets = []
    
    for text in texts:
        # 1. チャンキング
        chunks = chunk_text(text, max_chars=200)
        
        for chunk in chunks:
            # 2. トリプレット抽出
            if use_llm and llm is not None:
                triplets = extract_triplets_from_chunk_llm(chunk, llm)
            else:
                triplets = extract_triplets_from_chunk_rule_based(chunk)
            
            all_triplets.extend(triplets)
    
    # 3. グラフDB構築
    graph_db = build_graph_from_triplets(all_triplets, embed_model)
    
    return graph_db


# =====================================================================
# 使用例
# =====================================================================
if __name__ == "__main__":
    # 埋め込みモデルのロード
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 軽量LLMのロード（LLM版を使う場合）
    llm = pipeline(
        "text-generation",
        model="FilippoMedia/TinyLlama-1.1B-Chat-v1.0-miniguanaco",
        max_new_tokens=150,
        temperature=0.0
    )
    
    # サンプル文章
    sample_texts = [
        "Google Colab is a cloud-based Python development environment created by Google.",
        "Google Colab supports powerful GPUs like NVIDIA T4 for deep learning.",
        "NetworkX is a Python library used for studying graphs and networks.",
        "GraphRAG combines Knowledge Graphs with LLM retrieval to improve RAG systems.",
        "GraphRAG heavily utilizes NetworkX or Neo4j to store structural data."
    ]
    
    # GraphRAG用DBの構築（ルールベース版）
    print("Building GraphRAG DB from texts (rule-based)...")
    graph_db_rule = build_graphrag_db_from_texts(
        sample_texts, embed_model, use_llm=False, llm=None
    )
    print(f"Rule-based DB: {graph_db_rule.number_of_nodes()} nodes, {graph_db_rule.number_of_edges()} edges")
    
    # GraphRAG用DBの構築（LLM版）
    print("\nBuilding GraphRAG DB from texts (LLM-based)...")
    graph_db_llm = build_graphrag_db_from_texts(
        sample_texts, embed_model, use_llm=True, llm=llm
    )
    print(f"LLM-based DB: {graph_db_llm.number_of_nodes()} nodes, {graph_db_llm.number_of_edges()} edges")
    
    # 可視化（任意）
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph_db_rule, k=1.0, seed=42)
    nx.draw(graph_db_rule, pos, with_labels=True, node_color="lightblue", node_size=2500, font_size=9, font_weight="bold", arrowsize=15)
    edge_labels = nx.get_edge_attributes(graph_db_rule, "relation")
    nx.draw_networkx_edge_labels(graph_db_rule, pos, edge_labels=edge_labels, font_size=8, font_color="red")
    plt.title("GraphRAG DB Built from Texts (Rule-based)")
    plt.show()

import os
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.embeddings import resolve_embed_model
from llama_index.postprocessor.huggingface import HuggingFaceRerank

# ---------------------------------------------------------
# 1. テストデータの準備
# ---------------------------------------------------------
documents_text = [
    "株式会社TECH-AIの福利厚生では、1年に1回、上限10万円までの旅行費補助が出る。申請は社内ポータルから行う。",
    "スマートフォンの次世代通信規格「6G」は、2030年頃の商用化を目指して世界中で開発が進んでいる。",
    "日本の新元号「未来（みらい）」は、2028年から施行される予定であるとの誤報が流れた。",
    "TECH-AI社では、リモートワーク手当として毎月5,000円が全社員に支給される。",
    "旅行費の補助金（出張旅費規程）に関する問い合わせは、総務部の佐藤さんまで連絡してください。"
]

# LlamaIndexのDocumentオブジェクトのリストに変換
documents = [Document(text=text) for text in documents_text]

# ---------------------------------------------------------
# 2. 埋め込みモデルとインデックス（データベース）の構築
# ---------------------------------------------------------
print("1. 埋め込みモデルを初期化し、インデックスを構築中...")

# 日本語に対応したオープンソースの埋め込みモデルを指定
embed_model = resolve_embed_model("local:bzk/ja-sentence-transformer-v1")

# ドキュメントをベクトル化してインデックスを構築
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# ---------------------------------------------------------
# 3. 高性能リランカー（NodePostprocessor）の準備
# ---------------------------------------------------------
print("2. リランキングモデル（BGE-Reranker）を読み込み中...")

# LlamaIndexが提供する HuggingFaceRerank クラスを使用
reranker = HuggingFaceRerank(
    model="BAAI/bge-reranker-v2-m3",
    top_n=2  # リランキング後に最終的に残す件数
)

print("準備完了。検索を実行します。\n" + "="*50)

# ---------------------------------------------------------
# 4. クエリエンジンの構築と実行
# ---------------------------------------------------------
# ユーザーの質問
user_query = "旅行の補助金について、申請はどこから行えがいいですか？"
print(f"【ユーザーの質問】: {user_query}\n")

# クエリエンジン（検索・生成を行う窓口）を作成
# ※今回は生成（LLM回答）の手前である「検索（Retrieval）」の結果を確認するため、
#   retrieverオブジェクトを直接操作するか、query_engineの段階でリランカーを仕込みます。
query_engine = index.as_query_engine(
    similarity_top_k=4,            # 1次検索（ベクトル検索）で多めに取得する件数
    node_postprocessors=[reranker] # 2次検索としてリランカーを適用
)

# 検索とコンテキスト抽出の実行（応答オブジェクトを取得）
# ※LLMをセットアップしていない場合は、以下のように retriever からノードを直接取得して確認できます。
retriever = index.as_retriever(similarity_top_k=4)

# 1. 1次検索（ベクトル検索のみ）の結果
initial_nodes = retriever.retrieve(user_query)
print("【1次検索：ベクトル検索（FAISS等）のみの結果】")
for i, node in enumerate(initial_nodes):
    print(f"   候補 {i+1} (Score: {node.score:.4f}): {node.text}")

print("\n" + "-"*50)

# 2. 2次検索（リランキング後）の結果
# 1次検索の結果をリランカーに投入
final_nodes = reranker.postprocess_nodes(initial_nodes, query_str=user_query)

print("【2次検索：リランキング（BGE-Reranker）適用後の結果】")
for i, node in enumerate(final_nodes):
    print(f"★ 順位 {i+1} (Score: {node.score:.4f}): {node.text}")