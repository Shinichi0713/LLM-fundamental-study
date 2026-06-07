import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================================================================
# 1. サンプルテキストデータ（ナレッジベースの元ネタ）
# =====================================================================
documents = [
    "Google Colab is a cloud-based Python development environment created by Google.",
    "Google Colab supports powerful GPUs like NVIDIA T4 for deep learning.",
    "NetworkX is a Python library used for studying graphs and networks.",
    "GraphRAG combines Knowledge Graphs with LLM retrieval to improve RAG systems.",
    "GraphRAG heavily utilizes NetworkX or Neo4j to store structural data."
]

# =====================================================================
# 2. LLMの代わりにトリプレット（主語-述語-目的語）を定義
# (※本来はLLMを使って自動抽出しますが、今回は確実なDB構築のため定義します)
# =====================================================================
# 形式: (主語, 目的語, 関係性)
triplets = [
    ("Google Colab", "Google", "created by"),
    ("Google Colab", "Python", "supports environment for"),
    ("Google Colab", "NVIDIA T4 GPU", "provides access to"),
    ("NVIDIA T4 GPU", "Deep Learning", "used for"),
    ("NetworkX", "Python", "is a library of"),
    ("NetworkX", "Graphs and Networks", "used for studying"),
    ("GraphRAG", "Knowledge Graphs", "combines"),
    ("GraphRAG", "LLM Retrieval", "integrates"),
    ("GraphRAG", "RAG Systems", "improves"),
    ("GraphRAG", "NetworkX", "utilizes"),
    ("GraphRAG", "Neo4j", "utilizes"),
]

# =====================================================================
# 3. グラフデータベースの構築 (NetworkX)
# =====================================================================
# Graph RAGのキモである有向グラフを作成
graph_db = nx.DiGraph()

# 埋め込みモデル（Embedding）のロード
# ColabのCPUでも一瞬で動く軽量・高性能モデルを使用
print("Loading Embedding Model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Building Graph RAG Database...")
for subject, obj, relation in triplets:
    # ノードが未登録なら追加し、テキスト表現のベクトル（埋め込み）を計算して保存
    # (これがGraph RAGにおける「エンティティ・ベクトル検索」の基盤になります)
    if not graph_db.has_node(subject):
        graph_db.add_node(subject, embedding=embed_model.encode(subject))
    if not graph_db.has_node(obj):
        graph_db.add_node(obj, embedding=embed_model.encode(obj))
        
    # エッジ（関係性）を追加。関係性のテキストもベクトル化して保持可能
    graph_db.add_edge(subject, obj, relation=relation, embedding=embed_model.encode(relation))

print(f"データベース構築完了! ノード数: {graph_db.number_of_nodes()}, エッジ数: {graph_db.number_of_edges()}")

# =====================================================================
# 4. 構築したデータベースの検索テスト（Vector Search on Graph）
# =====================================================================
def search_graph_node(query, graph, model, top_k=2):
    """クエリに最も意味が近いノードをグラフから検索する"""
    query_vector = model.encode(query)
    node_scores = []
    
    for node, data in graph.nodes(data=True):
        node_vector = data["embedding"]
        # コサイン類似度の計算
        similarity = np.dot(query_vector, node_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(node_vector))
        node_scores.append((node, similarity))
        
    # スコア順にソート
    node_scores.sort(key=lambda x: x[1], reverse=True)
    return node_scores[:top_k]

# テスト検索
query = "What runs on cloud hardware?"
print(f"\n検索クエリ: '{query}'")
matched_nodes = search_graph_node(query, graph_db, embed_model)

for rank, (node, score) in enumerate(matched_nodes, 1):
    print(f"{rank}位のヒットノード: {node} (類似度: {score:.4f})")
    # ヒットしたノードから伸びている「関係性（エッジ）」を抽出
    neighbors = graph_db.successors(node)
    for nbr in neighbors:
        rel = graph_db[node][nbr]["relation"]
        print(f"  $\rightarrow$ 関連知識: [{node}] --({rel})--> [{nbr}]")

# =====================================================================
# 5. 構築したデータベースの可視化
# =====================================================================
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(graph_db, k=1.0, seed=42)

# ノードと線の描画
nx.draw(graph_db, pos, with_labels=True, node_color="lightblue", node_size=2500, font_size=9, font_weight="bold", arrowsize=15, connectionstyle="arc3,rad=0.1")

# エッジのラベル（関係性）を描画
edge_labels = nx.get_edge_attributes(graph_db, "relation")
nx.draw_networkx_edge_labels(graph_db, pos, edge_labels=edge_labels, font_size=8, font_color="red")

plt.title("Graph RAG Sample Database Structure in Colab")
plt.show()


import numpy as np
import torch
from transformers import pipeline

# =====================================================================
# 1. グラフ検索（Retrieval）関数の定義
# =====================================================================
def retrieve_knowledge_graph(query, graph, model, top_k=2):
    query_vector = model.encode(query)
    node_scores = []
    
    for node, data in graph.nodes(data=True):
        node_vector = data["embedding"]
        similarity = np.dot(query_vector, node_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(node_vector))
        node_scores.append((node, similarity))
        
    node_scores.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, score in node_scores[:top_k]]
    
    extracted_facts = []
    for node in top_nodes:
        # パターンA: 自分が主語の関係
        for successor in graph.successors(node):
            relation = graph[node][successor]["relation"]
            extracted_facts.append(f"- {node} -> ({relation}) -> {successor}")
        # パターンB: 自分が目的語の関係
        for predecessor in graph.predecessors(node):
            relation = graph[predecessor][node]["relation"]
            extracted_facts.append(f"- {predecessor} -> ({relation}) -> {node}")
            
    return list(set(extracted_facts))

# =====================================================================
# 2. Graph RAG 実行パイプラインの定義
# =====================================================================
class GraphRAGSystem:
    def __init__(self, graph_db, embed_model, use_mock_llm=False):
        self.graph_db = graph_db
        self.embed_model = embed_model
        self.use_mock_llm = use_mock_llm
        
        if not self.use_mock_llm:
            print("Loading local TinyLlama LLM for Generation...")
            # 誰でもアクセスできる公開のリポジトリに修正しました
            self.llm = pipeline(
                "text-generation", 
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                max_new_tokens=150,
                temperature=0.1,
                device=0 if torch.cuda.is_available() else -1  # GPUがあれば使う
            )
        else:
            print("Using Mock LLM Mode (Fast & Stable)")

    def _mock_llm_generate(self, facts, question):
        """LLMのダウンロードが遅い・動かない場合のバックアップ用ロジック"""
        ans = "Based on the Knowledge Graph:\n"
        ans += "\n".join(facts)
        ans += f"\n\nTherefore, we can reason the answer for: '{question}'"
        return ans

    def ask(self, question):
        # 1. グラフデータベースから関連知識を検索
        facts = retrieve_knowledge_graph(question, self.graph_db, self.embed_model, top_k=2)
        context_str = "\n".join(facts)
        
        if self.use_mock_llm:
            return self._mock_llm_generate(facts, question)
            
        # 2. プロンプトの構築
        prompt = f"""<|system|>
Answer the question based ONLY on the provided Knowledge Graph facts.
If the facts do not contain the answer, say "I don't know".

[Knowledge Graph Facts]
{context_str}
</s>
<|user|>
{question}</s>
<|assistant|>
"""
        
        print("\n--- [Internal] Constructed Graph-Augmented Prompt ---")
        print(prompt)
        print("----------------------------------------------------\n")
        
        # 3. LLMによるテキスト生成
        outputs = self.llm(prompt)
        generated_text = outputs[0]["generated_text"]
        answer = generated_text.split("<|assistant|>")[-1].strip()
        
        return answer

# =====================================================================
# 3. 実際にGraph RAGを実行
# =====================================================================
# ※ もしLLMのロードで再度エラーが出る・または遅い場合は、以下を use_mock_llm=True にしてください
rag_sys = GraphRAGSystem(graph_db, embed_model, use_mock_llm=False)

# 質問例1: グラフの繋がりを辿る質問
question_1 = "Explain the relationship between GraphRAG and Python."
response_1 = rag_sys.ask(question_1)
print(f"=== Final GraphRAG Answer ===\n{response_1}\n")

print("=" * 60)

# 質問例2: ハードウェアに関する質問
question_2 = "What kind of GPU can I use on Google Colab for deep learning?"
response_2 = rag_sys.ask(question_2)
print(f"=== Final GraphRAG Answer ===\n{response_2}\n")

