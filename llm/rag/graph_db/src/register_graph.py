import networkx as nx
import matplotlib.pyplot as plt

# 1. グラフ（データベース）の初期化
G = nx.Graph()

# 2. ノード（データ）の追加
G.add_node("User_A", label="Person", name="Alice")
G.add_node("User_B", label="Person", name="Bob")
G.add_node("Item_X", label="Product", name="Smartphone")

# 3. エッジ（関係性）の追加
G.add_edge("User_A", "User_B", relation="FRIEND")
G.add_edge("User_A", "Item_X", relation="PURCHASED")

# 4. 可視化
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()


import os
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 1. APIキーの設定 (OpenAIを使用する例)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 2. サンプルデータの準備
documents = [
    Document(
        text=(
            "アラン・チューリングはイギリスの数学者であり、現代コンピュータ科学の父と呼ばれている。"
            "彼はチューリングマシンという概念を提唱し、第二次世界大戦中にはエニグマ暗号の解読に貢献した。"
        ),
        metadata={"doc_id": "doc_001", "source": "history_article"},
    ),
    Document(
        text=(
            "チューリング賞は、コンピュータ科学分野における最高の賞であり、"
            "「計算機科学のノーベル賞」とも称される。"
        ),
        metadata={"doc_id": "doc_002", "source": "award_info"},
    ),
]

# 3. LLM と Embedding モデルの設定
llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 4. テキストのチャンク分割設定
# 文脈を保持した適切なサイズのテキストノード (Text Node) に分割
node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

# 5. インメモリ型 Property Graph Store の作成 (永続化する場合は他のGraphStoreに変更可能)
graph_store = SimplePropertyGraphStore()

# 6. GraphRAG 用 PropertyGraphIndex の構築
# PropertyGraphIndex を使うことで、抽出された Entity/Relation と
# その元となった Text Node (ソース文章) の紐付けが自動的にインデックス化されます。
index = PropertyGraphIndex.from_nodes(
    nodes,
    llm=llm,
    embed_model=embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)

# 7. インデックスおよびグラフデータの保存（永続化）
# ストレージディレクトリに保存することで次回以降再利用可能
index.storage_context.persist(persist_dir="./graph_rag_storage")

print("========== GraphRAG データベースの構築・保存が完了しました ==========")

# -------------------------------------------------------------------
# 【確認用】抽出されたエンティティとソース文章の紐付け確認
# -------------------------------------------------------------------

# グラフストアから抽出されたトリプル（主語 - 述語 - 目的語）やノード情報を確認
for triplet in graph_store.get_triplets():
    # 各エンティティ（Node）に関連付けられたソーステキスト情報を確認
    source_node_id = getattr(triplet.subject, "source_id", None)
    print(f"\n[Triple]: {triplet.subject.name} --({triplet.label})--> {triplet.object.name}")

# トレサビリティの確認: インデックスのDocstoreから元文章を取得
docstore = index.docstore
print("\n========== 登録されているベース文章（Text Chunks） ==========")
for node_id, node_obj in docstore.docs.items():
    print(f"\n[Node ID]: {node_id}")
    print(f"[Text Chunk]: {node_obj.get_content()}")
    print(f"[Source Metadata]: {node_obj.metadata}")


import os
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 1. APIキーの設定
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 2. ディレクトリからドキュメントを一括読み込み
# ./data ディレクトリ配下のファイル (.txt, .pdf, .md, .docx など) を自動で読み込みます
data_dir = "./data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    # テスト用のサンプルファイルを作成
    with open(os.path.join(data_dir, "sample1.txt"), "w", encoding="utf-8") as f:
        f.write(
            "アラン・チューリングはイギリスの数学者であり、現代コンピュータ科学の父と呼ばれている。"
            "彼はチューリングマシンという概念を提唱し、第二次世界大戦中にはエニグマ暗号の解読に貢献した。"
        )
    with open(os.path.join(data_dir, "sample2.txt"), "w", encoding="utf-8") as f:
        f.write(
            "チューリング賞は、コンピュータ科学分野における最高の賞であり、"
            "「計算機科学のノーベル賞」とも称される。"
        )

reader = SimpleDirectoryReader(
    input_dir=data_dir,
    recursive=True,  # サブディレクトリも再帰的に読み込む場合は True
)
documents = reader.load_data()

print(f"読み込んだファイル数/ページ数: {len(documents)}")

# 3. LLM・Embedding・チャンク分割の設定
llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# チャンク分割 (ファイル内の文章を適切なサイズに切断)
node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

# 4. Property Graph Store と Index の構築
graph_store = SimplePropertyGraphStore()

index = PropertyGraphIndex.from_nodes(
    nodes,
    llm=llm,
    embed_model=embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)

# 5. 保存（永続化）
index.storage_context.persist(persist_dir="./graph_rag_storage")

print("\n========== データベースの構築・保存が完了しました ==========")

# -------------------------------------------------------------------
# 【確認用】読み込まれた各ノード（テキストチャンク）とファイル情報の出力
# -------------------------------------------------------------------
print("\n========== 登録されたベース文章とファイル情報 ==========")
for node in nodes:
    print(f"\n[Node ID]: {node.node_id}")
    print(f[元ファイル名]: {node.metadata.get('file_name')})
    print(f"[ファイルパス]: {node.metadata.get('file_path')}")
    print(f"[該当チャンクのテキスト]:\n{node.get_content()}")




import os
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 1. APIキーの設定
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 2. ディレクトリからドキュメントを読み込み
data_dir = "./data"
reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
documents = reader.load_data()

# 3. LLM・Embedding・チャンク分割の設定
llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

node_parser = SentenceSplitter(chunk_size=300, chunk_overlap=30)
nodes = node_parser.get_nodes_from_documents(documents)

# 4. メタデータ（ソース文章）をノードプロパティに埋め込む抽出器（KG Extractor）の設定
# SchemaLLMPathExtractor を使用し、抽出されるノードにソーステキストをプロパティとして付与します
kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    max_triplets_per_chunk=10,
    num_workers=4,
)

# 5. Graph Store の初期化
graph_store = SimplePropertyGraphStore()

# 6. PropertyGraphIndex の構築
index = PropertyGraphIndex.from_nodes(
    nodes,
    llm=llm,
    embed_model=embed_model,
    kg_extractors=[kg_extractor],  # カスタム抽出器を指定
    property_graph_store=graph_store,
    show_progress=True,
)

# 7. インデックスの保存
index.storage_context.persist(persist_dir="./graph_rag_storage")

print("\n========== データベースの構築・保存が完了しました ==========")

# -------------------------------------------------------------------
# 【確認用】抽出されたエンティティノードと、保持されている元文章メタデータ
# -------------------------------------------------------------------
print("\n========== グラフノードに保持された元文章プロパティの確認 ==========")

# グラフストアからすべてのノード（エンティティ）を取得して確認
all_nodes = graph_store.get_nodes()
for node in all_nodes[:5]:  # 先頭5件を表示
    print(f"\n[Entity Name]: {node.name}")
    print(f"[Properties / Metadata]: {node.properties}")

print("\n========== トリプル（エッジ）と対応するソースノードID ==========")
for triplet in graph_store.get_triplets()[:5]:  # 先頭5件を表示
    # 各トリプルには生成元の TextNode ID (source_id) が紐付いています
    print(
        f"[Triple]: {triplet.subject.name} --({triplet.label})--> {triplet.object.name}"
    )
    print(f"  └─ [Source Text Node ID]: {triplet.source_id}")


