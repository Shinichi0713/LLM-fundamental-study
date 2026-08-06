import os
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import VectorContextRetriever

# 1. 既存のGraphDB（Neo4j）への接続設定
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="your_password",
    url="bolt://localhost:7687",
)

# 2. 登録済みGraphStoreからIndexをロード設定
index = PropertyGraphIndex.from_existing(
    graph_store=graph_store
)

# -------------------------------------------------------------
# 方法A: クエリエンジンによる「自然言語の回答」出力
# -------------------------------------------------------------
query_engine = index.as_query_engine(include_text=True)

response = query_engine.query("RegionCLIPとSAMの関係性やそれぞれの特徴について教えてください。")
print("=== LLMによる検索・生成回答 ===")
print(response)

# -------------------------------------------------------------
# 方法B: Retrieverによる「ヒットしたグラフデータ（トリプル）」の直接出力
# -------------------------------------------------------------
retriever = index.as_retriever(
    sub_retrievers=[
        # キーワード/エンティティベースのグラフ探索
        index.as_retriever(sub_retriever_type="llm"), 
    ]
)

# 検索の実行
nodes = retriever.retrieve("RegionCLIP")

print("\n=== GraphDBから取得されたサブグラフデータ（ノード・トリプル） ===")
for node in nodes:
    print(f"スコア: {node.score}")
    print(f"テキスト/トリプル情報:\n{node.text}")
    print("-" * 40)