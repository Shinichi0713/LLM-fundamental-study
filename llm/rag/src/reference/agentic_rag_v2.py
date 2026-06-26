# Colab で実行する前提のコード例

# 必要なパッケージをインストール
!pip install llama-index-core llama-index-llms-openai llama-index-graph-stores-neo4j neo4j openai

import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.query_engine import GraphRAGQueryEngine
from llama_index.llms.openai import OpenAI

# OpenAI API キーを設定（Colab のシークレットなどから読み込む想定）
os.environ["OPENAI_API_KEY"] = "sk-..."  # 実際のキーに置き換え

# LLM の設定（gpt-4 など）
llm = OpenAI(model="gpt-4")

# サンプルドキュメントを読み込み
documents = SimpleDirectoryReader("sample_docs").load_data()

# グラフストア（ここでは簡易なインメモリ版）
graph_store = SimpleGraphStore()

# ストレージコンテキスト
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# PropertyGraphIndex の構築（GraphRAG のベース）
property_graph_index = PropertyGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    llm=llm,
    # 必要に応じて entity_extraction_prompt_templates などをカスタマイズ
)

# GraphRAG クエリエンジンを作成
graph_rag_query_engine = GraphRAGQueryEngine(
    index=property_graph_index,
    llm=llm,
    # コミュニティ検出や要約のパラメータを調整可能
)

from langchain.agents import Tool
from langchain_openai import ChatOpenAI

# LangChain 用の LLM
langchain_llm = ChatOpenAI(model="gpt-4", temperature=0)

# GraphRAG を呼び出す関数
def graph_rag_tool_func(query: str) -> str:
    response = graph_rag_query_engine.query(query)
    return str(response)

# LangChain の Tool として登録
graph_rag_tool = Tool(
    name="GraphRAG_Knowledge_Base",
    description=(
        "内部ドキュメントから、グラフ構造に基づいて高度な推論を行うためのツール。"
        "複雑な関係性やマルチホップな質問に適している。"
    ),
    func=graph_rag_tool_func,
)

from langchain.agents import initialize_agent, AgentType

# 使うツールのリスト（ここでは GraphRAG のみ）
tools = [graph_rag_tool]

# エージェントの初期化
agent = initialize_agent(
    tools=tools,
    llm=langchain_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ReAct 型エージェント
    verbose=True,
)

# 実行例
result = agent.run(
    "サンプルドキュメント群に基づいて、関連するエンティティ間の関係を整理し、"
    "その中で最も中心的な役割を果たしているエンティティはどれか説明してください。"
)

print(result)