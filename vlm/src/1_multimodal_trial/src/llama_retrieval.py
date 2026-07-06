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