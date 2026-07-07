import os
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# 1. APIキーとLLM/Embeddingの設定
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# デフォルトのモデルを指定（必要に応じてモデル名を変更してください）
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

# 2. グラフデータベース（GraphDB）の接続設定
# ※ローカルまたはクラウド(Neo4j Aura等)で起動しているNeo4jの情報を入力します
graph_store = Neo4jPGStore(
    username="neo4j",
    password="your-password",
    url="bolt://localhost:7687",
)

# 3. テキストデータの読み込み
# 「data」ディレクトリの中に解析したいテキストファイル（.txtなど）を配置しておきます
documents = SimpleDirectoryReader("./data").load_data()

# 4. プロパティグラフインデックスの作成（データ投入とグラフ構築）
# テキストから自動で三つ組（主語-述語-目的語）を抽出し、GraphDBへ格納します
index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    show_progress=True
)

# 💡【すでにデータがGraphDBにある場合】は、上記 from_documents の代わりに以下でロードできます
# index = PropertyGraphIndex.from_existing(property_graph_store=graph_store)

# 5. 検索エンジン（Query Engine）の構築とクエリ実行
# デフォルトでは、ベクトル検索とキーワード検索、およびグラフのパス探索がハイブリッドで行われます
query_engine = index.as_query_engine(include_text=True)

# 自然言語で質問
response = query_engine.query("主人公とA社の関係性について教えてください。")

print("\n--- 検索結果 ---")
print(response)

from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)

# 明示的に検索手法（例: ベクトルとシノニム）を指定
sub_retrievers = [
    VectorContextRetriever(index),
    LLMSynonymRetriever(index),
]

query_engine = index.as_query_engine(
    sub_retrievers=sub_retrievers, 
    include_text=True
)

### GraphDBの中身

import os
from neo4j import GraphDatabase

# 1. 接続情報の設定（先ほどと同じ情報を使用）
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "your-password")

def check_graph_database():
    print("GraphDBのデータ格納状況を確認中...\n")
    
    # ドライバーの初期化
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        # セッションを開いてCypherクエリを実行
        with driver.session() as session:
            
            # ① 総ノード数のカウント
            node_query = "MATCH (n) RETURN count(n) AS node_count"
            node_result = session.run(node_query).single()
            node_count = node_result["node_count"] if node_result else 0
            
            # ② 総エッジ（関係性）数のカウント
            edge_query = "MATCH ()-[r]->() RETURN count(r) AS edge_count"
            edge_result = session.run(edge_query).single()
            edge_count = edge_result["edge_count"] if edge_result else 0
            
            # 画面に出力
            print("====================================")
            print(f"- 登録されている総ノード数 (Node): {node_count} 個")
            print(f"- 登録されている総エッジ数 (Edge): {edge_count} 本")
            print("====================================")
            
            # 3. データの状態に応じたアドバイス
            if node_count == 0:
                print("\n[判定] ⚠️ グラフDBが完全に空っぽです！")
                print("LlamaIndexからのデータ投入（インデックス作成）が失敗しているか、")
                print("コミットされる前に処理が落ちている可能性があります。")
            elif edge_count == 0:
                print("\n[判定] ⚠️ ノードはありますが、矢印（関係性）が1本もありません！")
                print("LLMがテキストからエンティティ同士の繋がりをうまく抽出できなかった可能性があります。")
            else:
                print("\n[判定] ✨ データは正常に格納されています！")
                print("これで検索できない場合、原因は「LlamaIndex側のリトリーバー（検索アルゴリズム）の設定」や")
                print("「質問文とグラフ内データのミスマッチ」に絞られます。")

            # （オマケ）具体的なノードの種類や関係性の種類も内訳を表示
            if node_count > 0:
                print("\n--- 登録されているラベルの内訳 ---")
                labels_query = "MATCH (n) RETURN labels(n) AS labels, count(n) AS count"
                for record in session.run(labels_query):
                    print(f"  - ラベル {record['labels']}: {record['count']} 個")
                    
            if edge_count > 0:
                print("\n--- 登録されている関係性の内訳 ---")
                types_query = "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count"
                for record in session.run(types_query):
                    print(f"  - 関係性 [{record['type']}]: {record['count']} 本")

if __name__ == "__main__":
    # 接続に必要な追加ライブラリ `pip install neo4j` が必要です
    try:
        check_graph_database()
    except Exception as e:
        print(f"エラー: データベースへの接続自体に失敗しました。\n{e}")


# データ登録

import os
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# 1. APIキーとLLM/Embeddingの設定
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# グラフ構造の抽出をより正確にするため、少し賢いモデル（gpt-4oなど）を推奨します
Settings.llm = OpenAI(model="gpt-4o", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

# 2. グラフデータベース（GraphDB）の接続設定
graph_store = Neo4jPGStore(
    username="neo4j",
    password="your-password",
    url="bolt://localhost:7687",
)

# 3. PDFやテキストファイルが置かれたディレクトリの指定
# 指定したフォルダ以下の「.txt」「.md」「.pdf」などを自動で読み込みます
data_dir = "./data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"📁 '{data_dir}' フォルダを作成しました。ここにPDFやテキストファイルを配置してください。")
    exit()

print("📄 データを読み込んでいます...")
# SimpleDirectoryReaderは標準でpypdfを使ったPDFパースに対応しています
reader = SimpleDirectoryReader(input_dir=data_dir)
documents = reader.load_data()

if not documents:
    print("⚠️ 読み込むファイルが見つかりませんでした。ファイルが正しく配置されているか確認してください。")
    exit()

print(f"Successfully loaded {len(documents)} pages/documents.")

# 4. プロパティグラフインデックスの作成とデータ登録
# LLMがドキュメントを読み、自動でエンティティとリレーションを抽出してNeo4jに書き込みます
print("🧠 グラフ構造を抽出してGraphDBに登録中... (少し時間がかかります)")
index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    show_progress=True
)

print("✨ 登録が完了しました！データベースを確認するか、検索を実行してください。")

import os
from pypdf import PdfReader
from llama_index.core import Document, PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

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

# 3. 自前でファイルを読み込んで Document オブジェクトのリストを作る関数
def load_custom_documents(data_dir: str) -> list[Document]:
    documents = []
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        # テキストファイルの読み込み
        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            # Documentを作成し、メタデータ（ファイル名など）を付与
            doc = Document(
                text=text,
                metadata={"file_name": filename, "file_type": "text"}
            )
            documents.append(doc)
            print(f"Loaded text: {filename}")
            
        # PDFファイルの読み込み（pypdfを直接使用）
        elif filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            # ページごとにテキストを抽出
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():  # 空白ページはスキップ
                    doc = Document(
                        text=text,
                        metadata={
                            "file_name": filename,
                            "file_type": "pdf",
                            "page_number": page_num + 1
                        }
                    )
                    documents.append(doc)
            print(f"Loaded PDF: {filename} ({len(reader.pages)} pages)")
            
    return documents

# 4. 自作関数でデータを読み込み
data_dir = "./data"
custom_docs = load_custom_documents(data_dir)

if not custom_docs:
    print("⚠️ 読み込むデータがありませんでした。")
    exit()

# 5. プロパティグラフインデックスの作成とデータ登録
print(f"\n🧠 {len(custom_docs)}個のセグメントからグラフ構造を抽出してGraphDBに登録中...")
index = PropertyGraphIndex.from_documents(
    custom_docs,
    property_graph_store=graph_store,
    show_progress=True
)

print("✨ 登録が完了しました！")

# 検索用スキーマ

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

