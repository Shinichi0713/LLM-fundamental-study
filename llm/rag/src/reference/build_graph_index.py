"""
LlamaIndex を用いて PDF / テキストファイルを読み込み、
グラフDB (Neo4j) にナレッジグラフとして登録するサンプルスクリプト。

事前準備:
1. Neo4j を起動しておく (Docker例):
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password123 \
     -e NEO4J_PLUGINS='["apoc"]' \
     neo4j:5

2. 必要なパッケージをインストール:
   pip install llama-index llama-index-graph-stores-neo4j \
               llama-index-llms-openai llama-index-embeddings-openai \
               pypdf

3. 環境変数に OPENAI_API_KEY を設定しておく
   export OPENAI_API_KEY="sk-xxxx"
"""

import os
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


# ---------------------------------------------------------
# 1. LLM / Embedding モデルの設定
# ---------------------------------------------------------
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


# ---------------------------------------------------------
# 2. Neo4j グラフストアへの接続
# ---------------------------------------------------------
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")

graph_store = Neo4jPropertyGraphStore(
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    url=NEO4J_URI,
    # database="neo4j",  # 必要に応じてDB名を指定
)


# ---------------------------------------------------------
# 3. PDF / テキストファイルの読み込み
#    input_dir 配下の .pdf, .txt, .md などをまとめて読み込む
# ---------------------------------------------------------
INPUT_DIR = "./data"  # ここにPDFやテキストを置く

documents = SimpleDirectoryReader(
    input_dir=INPUT_DIR,
    recursive=True,
    required_exts=[".pdf", ".txt", ".md"],
).load_data()

print(f"読み込んだドキュメント数: {len(documents)}")


# ---------------------------------------------------------
# 4. グラフ抽出の設定
#    SimpleLLMPathExtractor: LLMに (subject, relation, object) の
#    トリプルを自由生成させるシンプルな抽出器。
#    スキーマを固定したい場合は SchemaLLMPathExtractor を使用する。
# ---------------------------------------------------------
kg_extractor = SimpleLLMPathExtractor(
    llm=Settings.llm,
    max_paths_per_chunk=10,  # 1チャンクあたり抽出するトリプル数の上限
    num_workers=4,
)


# ---------------------------------------------------------
# 5. PropertyGraphIndex を構築しつつ Neo4j に書き込む
# ---------------------------------------------------------
index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[kg_extractor],
    property_graph_store=graph_store,
    embed_kg_nodes=True,   # ノードにも埋め込みを付与してベクトル検索可能にする
    show_progress=True,
)

print("グラフDBへの登録が完了しました。")


from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from typing import Literal

# エンティティの種別を定義
entities = Literal["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT"]

# 関係の種別を定義
relations = Literal["WORKS_FOR", "FOUNDED", "LOCATED_IN", "PARTICIPATED_IN", "PRODUCES"]

# (エンティティ, 関係, エンティティ) の許可される組み合わせ(トリプル)を定義
validation_schema = {
    "PERSON": ["WORKS_FOR", "FOUNDED", "PARTICIPATED_IN"],
    "ORGANIZATION": ["LOCATED_IN", "PRODUCES"],
    "EVENT": ["LOCATED_IN"],
}

kg_extractor = SchemaLLMPathExtractor(
    llm=Settings.llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True,  # True: スキーマ外の抽出は除外 / False: 参考程度に留める
)


import os
from pypdf import PdfReader
from llama_index.core import Document, PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
# スキーマ制御に必要なモジュールをインポート
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from enum import Enum

# 1. APIキーとLLM/Embeddingの設定
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
Settings.llm = OpenAI(model="gpt-4o", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

# 2. グラフデータベース（GraphDB）の接続設定
graph_store = Neo4jPGStore(
    username="neo4j",
    password="your-password",
    url="bolt://localhost:7687",
)

# ==========================================
# 3. グラフスキーマの定義
# ==========================================
# ノードのラベル（種類）を定義
class NodeLabel(str, Enum):
    PERSON = "PERSON"        # 人物
    COMPANY = "COMPANY"      # 企業・組織
    PROJECT = "PROJECT"      # プロジェクト・製品
    TECHNOLOGY = "TECHNOLOGY" # 技術・スキル

# 関係性（エッジの種類）を定義
class RelationType(str, Enum):
    WORKS_AT = "WORKS_AT"      # 〜に勤務している
    MANAGES = "MANAGES"        # 〜を管理・推進している
    USES = "USES"              # 〜（技術など）を使用している
    PARTNER_WITH = "PARTNER_WITH" # 〜と提携している

# スキーマをバリデーション構造として定義
# (主語のラベル, 関係性, 目的語のラベル) の組み合わせを厳密に指定します
validation_schema = {
    NodeLabel.PERSON: [RelationType.WORKS_AT, RelationType.MANAGES, RelationType.USES],
    NodeLabel.COMPANY: [RelationType.PARTNER_WITH, RelationType.USES],
    NodeLabel.PROJECT: [RelationType.USES],
}

# 定義したスキーマをベースに抽出器（Extractor）を作成
schema_extractor = SchemaLLMPathExtractor(
    llm=Settings.llm,
    possible_entities=NodeLabel,
    possible_relations=RelationType,
    kg_validation_schema=validation_schema,
    strict=True # Trueにすると、スキーマに合致しない自由な抽出を禁止します
)
# ==========================================

# 4. 自前でのファイル読み込み処理（前回のコードと同様）
def load_custom_documents(data_dir: str) -> list[Document]:
    documents = []
    if not os.path.exists(data_dir):
        return documents
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(Document(text=f.read(), metadata={"file_name": filename}))
        elif filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append(Document(text=text, metadata={"file_name": filename, "page": page_num + 1}))
    return documents

custom_docs = load_custom_documents("./data")

# 5. 設計したスキーマ（kg_extractors）を指定してインデックスを作成
print("🧠 定義されたスキーマに従って、GraphDBに厳密にデータを登録中...")
index = PropertyGraphIndex.from_documents(
    custom_docs,
    property_graph_store=graph_store,
    kg_extractors=[schema_extractor], # ここで自作したスキーマ抽出器を渡す
    show_progress=True
)

print("✨ スキーマに沿った綺麗なグラフの登録が完了しました！")

from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)

query_engine = index.as_query_engine(
    sub_retrievers=[
        LLMSynonymRetriever(index.property_graph_store, llm=Settings.llm),
        VectorContextRetriever(index.property_graph_store, embed_model=Settings.embed_model),
    ],
    include_text=True,
)

# ---------------------------------------------------------
# 6. (任意) 登録したグラフに対してクエリを実行する例
# ---------------------------------------------------------
if __name__ == "__main__":
    query_engine = index.as_query_engine(
        include_text=True,   # 元テキストの内容も回答に利用する
        similarity_top_k=5,
    )

    question = "このドキュメントに登場する主要な登場人物と関係性を教えてください。"
    response = query_engine.query(question)

    print("\n=== 質問 ===")
    print(question)
    print("\n=== 回答 ===")
    print(response)

    # 抽出されたトリプルを確認したい場合
    # retriever = index.as_retriever(include_text=False)
    # nodes = retriever.retrieve(question)
    # for n in nodes:
    #     print(n.text)
