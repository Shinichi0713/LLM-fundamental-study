import os
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgentWorker

# 1. 環境変数とモデルの初期化
load_dotenv()

# エージェントの「思考・判断」の脳となるLLM（Tool Callに強いGPT-4o系を推奨）
llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
# ベクトル検索用の埋め込みモデル
embed_model = resolve_embed_model("local:bzk/ja-sentence-transformer-v1")

# ---------------------------------------------------------
# 2. 異なる2つのデータソース（RAG）の準備
# ---------------------------------------------------------
print("1. データソースのインデックス（RAG）を構築中...")

# データソース A: 社内規定
hr_docs = [
    Document(text="株式会社TECH-AIの福利厚生では、1年に1回、上限10万円までの旅行費補助が出る。申請は社内ポータルから行う。"),
    Document(text="TECH-AI社では、リモートワーク手当として毎月5,000円が全社員に支給される。")
]
hr_index = VectorStoreIndex.from_documents(hr_docs, embed_model=embed_model)
hr_engine = hr_index.as_query_engine(llm=llm)

# データソース B: 技術マニュアル
tech_docs = [
    Document(text="AI基盤『NeuronFlow』のバージョン2.0では、新機能としてマルチモーダルリランキング（Vision対応）が追加された。"),
    Document(text="NeuronFlowのAPI認証エラー（コード501）が発生した場合は、環境変数のAPIキーの有効期限を確認してください。")
]
tech_index = VectorStoreIndex.from_documents(tech_docs, embed_model=embed_model)
tech_engine = tech_index.as_query_engine(llm=llm)

# ---------------------------------------------------------
# 3. RAGをエージェント用の「ツール」としてパッケージング
# ---------------------------------------------------------
# エージェントは、この「name」と「description」を見て、どちらのツールを使うべきか判断します。
# そのため、説明文（description）を具体的に書くことが精度向上の最大のコツです。
query_engine_tools = [
    QueryEngineTool(
        query_engine=hr_engine,
        metadata=ToolMetadata(
            name="hr_rules_search",
            description="TECH-AI社の福利厚生、手当、社内ルール、申請方法に関する質問に答えるための検索ツール。"
        )
    ),
    QueryEngineTool(
        query_engine=tech_engine,
        metadata=ToolMetadata(
            name="tech_manual_search",
            description="AI基盤『NeuronFlow』の仕様、機能、エラーコード、技術的なトラブルシューティングに関する質問に答えるための検索ツール。"
        )
    )
]

# ---------------------------------------------------------
# 4. Agentic RAG の構築
# ---------------------------------------------------------
print("2. Agentic RAG を起動中...")
agent = FunctionCallingAgentWorker.from_tools(
    tools=query_engine_tools,
    llm=llm,
    verbose=True # エージェントの「思考プロセス（どのツールを選んだか）」をターミナルに表示する
).as_agent()

# ---------------------------------------------------------
# 5. 実行テスト
# ---------------------------------------------------------
print("\n" + "="*50 + "\nエージェントへの質問を開始します。\n" + "="*50)

# テスト1: 社内規定ツールを選ぶべき質問
print("\n--- テスト1 ---")
response_1 = agent.chat("リモートワークの手当はいくら出ますか？")
print(f"【最終回答】: {response_1}\n")

# テスト2: 技術マニュアルツールを選ぶべき質問
print("\n--- テスト2 ---")
response_2 = agent.chat("NeuronFlowで501のエラーが出た時の対処法は？")
print(f"【最終回答】: {response_2}\n")

# テスト3: 複雑な質問（エージェントが自分で判断して、両方のツールを順番に叩く、または検索が不要と判断する）
print("\n--- テスト3 ---")
response_3 = agent.chat("NeuronFlowの最新機能と、当社のリモートワーク手当の金額をまとめて教えてください。")
print(f"【最終回答】: {response_3}\n")

import os
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgentWorker

# 環境変数とモデルの初期化
load_dotenv()

# 判断力とマルチステップの計画性に優れたモデル（gpt-4oなど）を推奨
llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
embed_model = resolve_embed_model("local:bzk/ja-sentence-transformer-v1")

# ---------------------------------------------------------
# 2. あえて情報をバラバラに配置したデータソースの準備
# ---------------------------------------------------------
print("1. データベース（ナレッジベース）を構築中...")

# データベース1: プロジェクトの基本情報（誰が責任者かだけが書いてある）
project_docs = [
    Document(text="2026年度の最重要機密プロジェクトである『Aegis（イージス）』の開発責任者は、シニアエンジニアの『マイケル・スコット』である。"),
    Document(text="新規事業プロジェクト『Horizons』の責任者は『キャロル・べス』である。")
]
project_index = VectorStoreIndex.from_documents(project_docs, embed_model=embed_model)
project_engine = project_index.as_query_engine(llm=llm)

# データベース2: 社員の名簿情報（誰がどこに所属しているかだけが書いてある）
employee_docs = [
    Document(text="マイケル・スコット（シニアエンジニア）の現在の所属部署は『次世代AI開発部・研究開発チーム』である。オフィスは5階。"),
    Document(text="キャロル・べス（マネージャー）の現在の所属部署は『グローバルマーケティング部』である。")
]
employee_index = VectorStoreIndex.from_documents(employee_docs, embed_model=embed_model)
employee_engine = employee_index.as_query_engine(llm=llm)

# ---------------------------------------------------------
# 3. エージェント用ツールの定義
# ---------------------------------------------------------
query_engine_tools = [
    QueryEngineTool(
        query_engine=project_engine,
        metadata=ToolMetadata(
            name="project_info_search",
            description="各プロジェクトの概要、目的、およびその『開発責任者・担当者の名前』を調べるためのツール。"
        )
    ),
    QueryEngineTool(
        query_engine=employee_engine,
        metadata=ToolMetadata(
            name="employee_directory_search",
            description="社員の名前から、その社員の『現在の所属部署』、職位、座席位置などを調べるためのツール。"
        )
    )
]

# ---------------------------------------------------------
# 4. Agentic RAG（マルチホップ対応）の構築
# ---------------------------------------------------------
print("2. マルチホップ対応 Agentic RAG を起動中...")
agent = FunctionCallingAgentWorker.from_tools(
    tools=query_engine_tools,
    llm=llm,
    verbose=True # エージェントの「思考のステップ」を可視化するために必須
).as_agent()

# ---------------------------------------------------------
# 5. 実行テスト（マルチホップ検索の検証）
# ---------------------------------------------------------
print("\n" + "="*50)
question = "プロジェクト『Aegis』の責任者は、どこの部署に所属していますか？"
print(f"【ユーザーからの質問】: {question}")
print("="*50 + "\n")

# エージェントが自律的に2回検索を行うか確認
response = agent.chat(question)

print("\n" + "="*50)
print(f"【最終回答】:\n{response}")
print("="*50)