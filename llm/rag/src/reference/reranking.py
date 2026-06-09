from sentence_transformers import CrossEncoder
import numpy as np

# 1. モデル読み込み
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 2. 初期検索結果（例：ベクトル検索で取ってきた上位50件）
query = "RAGにおけるリランキングの実装方法"
documents = [
    "RAGではまずベクトル検索で候補を取得し、その後クロスエンコーダで再ランク付けします。",
    "リランキングはRAGの精度を大きく向上させる重要なステップです。",
    "LLMに渡すコンテキストは、リランキングで質の高い少数に絞るのが良いです。",
    # ... 他47件
]

# 3. クエリ×文書ペアを作成
pairs = [(query, doc) for doc in documents]

# 4. スコア計算（バッチ推奨）
scores = model.predict(pairs, batch_size=16)

# 5. スコア順にソート
ranked_indices = np.argsort(scores)[::-1]  # 降順
reranked_docs = [documents[i] for i in ranked_indices]

# 6. 上位k件をLLMに渡す
top_k = 5
context_for_llm = "\n\n".join(reranked_docs[:top_k])

import cohere

# APIキーは環境変数などから取得
co = cohere.Client("YOUR_COHERE_API_KEY")

# 初期検索結果（例：ベクトル検索でtop 50）
query = "RAGにおけるリランキングの実装方法"
documents = [
    "RAGではまずベクトル検索で候補を取得し、その後クロスエンコーダで再ランク付けします。",
    "リランキングはRAGの精度を大きく向上させる重要なステップです。",
    # ... 他48件
]

# Rerank API呼び出し
response = co.rerank(
    model="rerank-english-v3.0",  # モデル名は最新のものを指定
    query=query,
    documents=documents,
    top_n=10,  # 上位何件を返すか
)

# スコア順に並んだ結果を取得
reranked_docs = [r.document for r in response.results]

# LLMに渡すコンテキスト
context_for_llm = "\n\n".join(reranked_docs)

from openai import OpenAI

client = OpenAI()

query = "RAGにおけるリランキングの実装方法"
docs = {
    "doc1": "RAGではまずベクトル検索で候補を取得し、その後クロスエンコーダで再ランク付けします。",
    "doc2": "リランキングはRAGの精度を大きく向上させる重要なステップです。",
    # ...
}

prompt = f"""
あなたは検索結果のリランキング専門モデルです。

クエリ: {query}

検索結果（文書IDと内容）:
""" + "\n".join([f"{i+1}. [{id}] {text}" for i, (id, text) in enumerate(docs.items())]) + """

上記の文書のうち、クエリに最も関連性が高いものから順に、
文書IDをカンマ区切りで並べてください。
例: doc3, doc1, doc7, ...

出力:
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
)

# 出力例: "doc2, doc1, doc3, ..."
ranked_ids = [id.strip() for id in response.choices[0].message.content.split(",")]
reranked_docs = [docs[id] for id in ranked_ids]

# 1. 初期検索（ベクトルDBなど）
query = "ユーザーの質問"
initial_results = vector_db.similarity_search(query, k=50)

# 2. リランキング（例：Cross-Encoder）
pairs = [(query, doc.page_content) for doc in initial_results]
scores = cross_encoder_model.predict(pairs)
ranked_indices = np.argsort(scores)[::-1]
reranked_docs = [initial_results[i] for i in ranked_indices]

# 3. 上位k件をLLMに渡す
top_k_docs = reranked_docs[:10]
context = "\n\n".join([doc.page_content for doc in top_k_docs])

# 4. LLMで回答生成
prompt = f"""
以下の文書を参考に、ユーザーの質問に答えてください。

文書:
{context}

質問:
{query}
"""
response = llm.generate(prompt)


import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------
# 1. テストデータと各種モデルの準備
# ---------------------------------------------------------
# RAGの知識ベースとなるドキュメント
documents = [
    "株式会社TECH-AIの福利厚生では、1年に1回、上限10万円までの旅行費補助が出る。申請は社内ポータルから行う。",
    "スマートフォンの次世代通信規格「6G」は、2030年頃の商用化を目指して世界中で開発が進んでいる。",
    "日本の新元号「未来（みらい）」は、2028年から施行される予定である。",
    "TECH-AI社では、リモートワーク手当として毎月5,000円が全社員に支給される。",
    "旅行費の補助金（出張旅費規程）に関する問い合わせは、総務部の佐藤さんまで連絡してください。"
]

print("1. モデルとインデックスを初期化中...")

# A. 1次検索用（Bi-Encoder）モデルとFAISSインデックスの作成
bi_encoder = SentenceTransformer('bzk/ja-sentence-transformer-v1')
embeddings = bi_encoder.encode(documents)
embeddings = np.array(embeddings).astype('float32')

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# B. 2次検索用（Cross-Encoder / Reranker）モデルの読み込み
# 日本語に対応した非常に強力なオープンソースリランカー
reranker_name = "BAAI/bge-reranker-v2-m3"
rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)
rerank_model.eval()  # 推論モード

print("準備完了。検索を実行します。\n" + "="*50)

# ---------------------------------------------------------
# 2. 検索 ＆ リランキング関数
# ---------------------------------------------------------
def retrieve_and_rerank(query, top_k_initial=4, top_k_final=2):
    """
    query: ユーザーの質問
    top_k_initial: 1次検索（FAISS）で多めに取得する件数
    top_k_final: リランキング後に最終的にLLMへ渡す件数
    """
    print(f"【ユーザーの質問】: {query}\n")
    
    # --- Step 1: 1次検索 (FAISS によるベクトル検索) ---
    query_vector = bi_encoder.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, min(top_k_initial, len(documents)))
    
    # 検索されたドキュメントの抽出
    retrieved_docs = [documents[idx] for idx in indices[0] if idx != -1]
    print(f"-> 1次検索（FAISS）で {len(retrieved_docs)} 件の候補を抽出しました。")
    for i, doc in enumerate(retrieved_docs):
        print(f"   候補 {i+1}: {doc}")

    if not retrieved_docs:
        return []

    # --- Step 2: 2次検索 (Cross-Encoder によるリランキング) ---
    # 「質問」と「各候補ドキュメント」のペアを作成
    pairs = [[query, doc] for doc in retrieved_docs]
    
    # モデルへの入力テンソルを作成
    inputs = rerank_tokenizer(
        pairs, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        max_length=512
    )
    
    # 各ペアの関連度スコアを計算
    with torch.no_grad():
        outputs = rerank_model(**inputs)
        # BGE-Rerankerはlogits（生の出力値）が高いほど関連度が高いと判定します
        scores = outputs.logits.view(-1).tolist()
    
    # ドキュメントとスコアを紐付け、スコアが高い順（降順）にソート
    reranked_results = sorted(
        zip(retrieved_docs, scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # --- Step 3: 結果の表示と返却 ---
    print("\n【2次検索：リランキング結果（スコア順）】")
    for i, (doc, score) in enumerate(reranked_results):
        # 最終的にLLMに渡す上位ドキュメントには★マークをつける
        marker = "★" if i < top_k_final else "  "
        print(f"{marker} 順位 {i+1} (Score: {score:6.4f}): {doc}")
        
    # 最終的にLLMに引き渡す上位 N 件のテキストを返す
    return [doc for doc, score in reranked_results[:top_k_final]]

# ---------------------------------------------------------
# 3. テスト実行
# ---------------------------------------------------------
if __name__ == "__main__":
    # 質問: 「旅行の補助の具体的な『申請方法』」を知りたいという意図
    user_query = "旅行の補助金について、申請はどこから行えばいいですか？"
    
    # 1次検索で4件抽出し、リランキングして最終的に最も適切な2件に絞り込む
    final_context = retrieve_and_rerank(user_query, top_k_initial=4, top_k_final=2)


    import os
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import MeCab

# 分かち書き関数の定義
wakati = MeCab.Tagger("-Owbakati")
def tokenize_japanese(text):
    return wakati.parse(text).strip().split()

class LoadedHybridSearcher:
    def __init__(self, db_dir="rag_db"):
        print("データベースをロード中...")
        # 1. 埋め込みモデルのロード
        self.bi_encoder = SentenceTransformer('bzk/ja-sentence-transformer-v1')
        
        # 2. FAISS インデックスのロード
        faiss_path = os.path.join(db_dir, "image_vectors.faiss")
        self.faiss_index = faiss.read_index(faiss_path)
        
        # 3. メタデータストア（パスとテキスト）のロード
        store_path = os.path.join(db_dir, "image_store.pkl")
        with open(store_path, "rb") as f:
            store_data = pickle.load(f)
            
        self.image_paths = store_data["image_paths"]
        self.metadatas = store_data["metadatas"]
        
        # 4. ロードしたテキストからBM25インデックスをオンメモリで再構築
        # （テキストデータ自体が軽量なため、起動時に構築するのが最もシンプルで柔軟です）
        tokenized_docs = [tokenize_japanese(text) for text in self.metadatas]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"ロード完了: {len(self.image_paths)} 件の画像が検索可能です。")

    def query(self, search_text, top_k=1):
        # A. ベクトル検索 (Dense)
        query_vector = self.bi_encoder.encode([search_text]).astype('float32')
        _, dense_indices = self.faiss_index.search(query_vector, min(top_k + 2, len(self.image_paths)))
        dense_hits = [idx for idx in dense_indices[0] if idx != -1]
        
        # B. キーワード検索 (BM25)
        tokenized_query = tokenize_japanese(search_text)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_hits = np.argsort(bm25_scores)[::-1][:top_k + 2].tolist()
        sparse_hits = [idx for idx in sparse_hits if bm25_scores[idx] > 0]
        
        # C. ハイブリッド統合（重複排除）
        combined_hits = list(set(dense_hits + sparse_hits))
        
        # 検索結果の整形
        results = []
        for idx in combined_hits[:top_k]:
            results.append({
                "image_path": self.image_paths[idx],
                "metadata": self.metadatas[idx]
            })
        return results

# ---------------------------------------------------------
# 検索のテスト実行
# ---------------------------------------------------------
if __name__ == "__main__":
    # 事前に build_vector_db.py を実行して "rag_db" がある状態で実行します
    if not os.path.exists("rag_db"):
        print("エラー: 先にデータベースを構築（build_vector_db.pyを実行）してください。")
    else:
        searcher = LoadedHybridSearcher(db_dir="rag_db")
        
        # 検索クエリの実行
        user_query = "2025年の売上推移が書かれた青いグラフの画像を探して"
        hits = searcher.query(user_query, top_k=1)
        
        print("\n=== 検索結果 ===")
        for hit in hits:
            print(f"【該当画像】: {hit['image_path']}")
            print(f"【AIによる画像説明文】:\n{hit['metadata']}")

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

class GraphRAGChunker:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        """
        chunk_size: 1つのチャンクの最大文字数。
                    GraphRAGではエンティティ抽出の密度を保つため、600〜800文字程度が推奨されます。
        chunk_overlap: チャンク間の重複文字数。
                       文をまたいだ関係性（リレーション）の途切れを防ぐために設定します。
        """
        # RecursiveCharacterTextSplitterは、セパレータの優先順位（改行 -> 句点など）に従って、
        # 意味の切れ目を保ちながら指定した文字数に収まるよう賢く分割してくれます。
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )

    def chunk_file(self, file_path):
        """ファイルを読み込んでチャンクに分割する"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # チャンキングの実行
        chunks = self.splitter.split_text(text)
        return chunks

# ---------------------------------------------------------
# 実行テスト
# ---------------------------------------------------------
if __name__ == "__main__":
    # テスト用の長文ドキュメントを作成
    sample_doc_path = "company_history.txt"
    sample_text = """
    株式会社TECH-AIは、2022年に代表取締役の山田太郎によって東京で設立された。同社は設立当初からAI技術を活用した業務効率化ツールの開発に注力しており、2024年には独自のLLMファインチューニング基盤「NeuronFlow」を発表した。この「NeuronFlow」の開発には、CTOである佐藤次郎率いる開発チームが2年間の歳月を費やした。
    
    2025年、TECH-AI社はグローバル展開の第一歩として、アメリカのシリコンバレーに子会社「TECH-AI Global Inc.」を設立。現地でのCEOには、元大手テック企業のアレックス・スミス氏が就任した。さらに、同年10月には国内投資ファンドから総額5億円の資金調達を実施している。この資金は、主に次世代通信規格「6G」を活用したリアルタイムAI解析システムのR&D（研究開発）に投資される予定である。
    
    現在、TECH-AI社は総勢100名の従業員を抱え、そのうち約6割がエンジニアおよびデータサイエンティストで構成されている。社内では完全フルリモートワーク制度が導入されており、メンバーは日本全国、および海外から開発に参加している。
    """
    
    with open(sample_doc_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    # チャンカーの初期化（分かりやすくするために少し小さめのサイズでテスト）
    # 実際のGraphRAG（Microsoft推奨値など）では chunk_size=600〜800, overlap=100〜150 あたりがベストです。
    chunker = GraphRAGChunker(chunk_size=200, chunk_overlap=40)
    
    try:
        final_chunks = chunker.chunk_file(sample_doc_path)
        
        print(f"元のテキストを {len(final_chunks)} 個のチャンクに分割しました。\n")
        
        for i, chunk in enumerate(final_chunks):
            print(f"--- [チャンク {i+1}] (文字数: {len(chunk)}) ---")
            print(chunk)
            print()
            
    finally:
        # テストファイルのクリーンアップ
        if os.path.exists(sample_doc_path):
            os.remove(sample_doc_path)


import openai  # または AzureOpenAI, anthropic, google.generativeai など
from typing import List, Dict

def generate_community_topic(
    chunks: List[str],
    model_name: str = "gpt-4o",  # または "gpt-4", "claude-3-sonnet", "gemini-1.5-flash" など
    max_tokens: int = 500,
) -> Dict[str, str]:
    """
    コミュニティ内のチャンク群から、トピックタイトルと説明文を生成する。
    
    Args:
        chunks: コミュニティに属するチャンクのテキストリスト
        model_name: 使用するLLMのモデル名
        max_tokens: 生成トークン数の上限
    
    Returns:
        {"title": str, "description": str}
    """
    
    # チャンクを結合（コンテキスト長に注意）
    # 実際には、コンテキスト長を超えないようにトリミングが必要
    combined_text = "\n\n".join([f"- {chunk}" for chunk in chunks])
    
    # プロンプト設計
    prompt = f"""
以下の文書群は、あるトピックに関連するチャンクです。
これらの内容を要約し、このトピックを表すタイトルと、
200字程度の説明文を生成してください。

チャンク:
{combined_text}

出力形式は以下のJSON形式でお願いします。
{{
  "title": "トピックのタイトル",
  "description": "トピックの説明（200字程度）"
}}
"""
    
    # LLM呼び出し（OpenAIの場合）
    client = openai.OpenAI(api_key="your-api-key")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "あなたは専門的な文書を要約するアシスタントです。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.1  # 事実ベースの要約なので低め
    )
    
    # レスポンスからJSONをパース（簡易実装）
    # 実際には、LLMがJSONを返すように設定するか、正規表現で抽出する
    content = response.choices[0].message.content.strip()
    
    # ここでは簡易的に文字列を返す（実際にはJSONパース推奨）
    return {
        "title": "生成されたタイトル（パース処理が必要）",
        "description": content  # 実際にはtitleとdescriptionを分離
    }


def generate_all_community_topics(
    community_chunks: Dict[int, List[str]],
    model_name: str = "gpt-4o",
) -> Dict[int, Dict[str, str]]:
    """
    すべてのコミュニティについてトピック要約を生成する。
    
    Args:
        community_chunks: {community_id: [chunk_texts]} の辞書
        model_name: 使用するLLMのモデル名
    
    Returns:
        {community_id: {"title": str, "description": str}}
    """
    community_topics = {}
    
    for community_id, chunks in community_chunks.items():
        print(f"Processing community {community_id}...")
        
        # コンテキスト長の制限に対応（簡易版）
        # 実際には、トークナイザで長さを計測し、長すぎる場合は要約を段階的に行うなどの工夫が必要
        if len(chunks) > 50:  # 例: 50チャンク以上はサンプリング
            chunks = chunks[:50]  # またはランダムサンプリング
        
        topic = generate_community_topic(chunks, model_name=model_name)
        community_topics[community_id] = topic
    
    return community_topics


# 例: コミュニティ0は「GraphRAGの概要」に関するチャンク群
community_chunks = {
    0: [
        "GraphRAGは、文書をチャンクに分割し、埋め込み類似度からグラフを構築する。",
        "グラフをクラスタリングしてコミュニティを抽出し、各コミュニティの要約を生成する。",
        "コミュニティ要約は、トピック単位での検索や推論に活用される。",
    ],
    1: [
        "埋め込みモデルは、テキストをベクトル空間に写像する。",
        "コサイン類似度を用いて、チャンク間の類似度を計算する。",
        "類似度が高いチャンク同士をエッジで結び、グラフを構築する。",
    ],
    # ... 他のコミュニティ
}

topics = generate_all_community_topics(community_chunks, model_name="gpt-4o")

for cid, topic in topics.items():
    print(f"Community {cid}:")
    print(f"  Title: {topic['title']}")
    print(f"  Description: {topic['description']}")
    print()


import os
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

class LlamaIndexGraphChunker:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        """
        LlamaIndexのSentenceSplitterを使用してチャンキングを行います。
        
        chunk_size: チャンクの最大トークン（または文字）数。
                    GraphRAGのエンティティ抽出効率を最大化するため600〜800が推奨されます。
        chunk_overlap: チャンク間の重複度。
        """
        # SentenceSplitterはデフォルトで文字数ではなく「トークン数」でカウントしますが、
        # 日本語環境で厳密に文字数ベースで制御したい場合は、length_functionに len を指定します。
        self.parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def process_text_to_nodes(self, text, doc_id="company_doc_001"):
        """
        テキストをLlamaIndexのドキュメントに変換し、
        GraphRAGに最適な関係性情報を持つ『Node（ノード）』のリストに分割します。
        """
        # 1. プレーンテキストをLlamaIndexのDocumentオブジェクトにラップ
        # （ここでファイル名や日付などのメタデータを付与することも可能です）
        document = Document(
            text=text, 
            doc_id=doc_id,
            metadata={"category": "company_history"}
        )
        
        # 2. パサーを実行してNode（チャンク）に分割
        # LlamaIndexのNodeは、自身のテキストだけでなく「前後のノードがどれか」という
        # リレーション情報（next_node / prev_node）を自動で保持します。
        nodes = self.parser.get_nodes_from_documents([document])
        return nodes

# ---------------------------------------------------------
# 実行テスト
# ---------------------------------------------------------
if __name__ == "__main__":
    # テスト用の長文ドキュメント
    sample_text = """
    株式会社TECH-AIは、2022年に代表取締役の山田太郎によって東京で設立された。同社は設立当初からAI技術を活用した業務効率化ツールの開発に注力しており、2024年には独自のLLMファインチューニング基盤「NeuronFlow」を発表した。この「NeuronFlow」の開発には、CTOである佐藤次郎率いる開発チームが2年間の歳月を費やした。
    
    2025年、TECH-AI社はグローバル展開の第一歩として、アメリカのシリコンバレーに子会社「TECH-AI Global Inc.」を設立。現地でのCEOには、元大手テック企業のアレックス・スミス氏が就任した。さらに、同年10月には国内投資ファンドから総額5億円の資金調達を実施している。この資金は、主に次世代通信規格「6G」を活用したリアルタイムAI解析システムのR&D（研究開発）に投資される予定である。
    """

    # 挙動が分かりやすいように小さめのサイズ（200文字）で設定
    chunker = LlamaIndexGraphChunker(chunk_size=200, chunk_overlap=40)
    
    # チャンキングの実行
    nodes = chunker.process_text_to_nodes(sample_text)
    
    print(f"LlamaIndexにより、テキストが {len(nodes)} 個のノードに分割されました。\n")
    
    for i, node in enumerate(nodes):
        print(f"--- [Node {i+1}] (Node ID: {node.node_id} / 文字数: {len(node.text)}) ---")
        print(f"メタデータ: {node.metadata}")
        print(f"本文:\n{node.text}")
        
        # LlamaIndex特有の機能：前後のノードとの関係性を保持しているか確認
        if node.next_node:
            print(f"-> 次のノードIDが存在します: {node.next_node.node_id}")
        print()


import numpy as np
from sentence_transformers import SentenceTransformer

def search_graph_nodes(query, graph, embed_model, top_k=5):
    """
    Graph RAGにおける「グラフ上のノード（エンティティ）検索」の実装
    
    クエリのベクトルに最も近いノードを、グラフ内の全ノードから検索して返します。
    
    Args:
        query (str): ユーザーの質問文（例: "What runs on cloud hardware?"）
        graph (nx.DiGraph): NetworkXの有向グラフ（ノードに embedding 属性を持つ）
        embed_model: SentenceTransformer などの埋め込みモデル
        top_k (int): 返す上位ノード数
    
    Returns:
        list: [(node_name, similarity_score), ...] のリスト（類似度降順）
    """
    # 1. クエリをベクトル化
    query_vector = embed_model.encode(query)
    
    # 2. 全ノードとの類似度を計算
    node_scores = []
    for node, data in graph.nodes(data=True):
        # ノードの埋め込みベクトルを取得
        node_vector = data["embedding"]
        
        # コサイン類似度の計算
        similarity = np.dot(query_vector, node_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(node_vector)
        )
        node_scores.append((node, similarity))
    
    # 3. 類似度で降順ソートし、上位 top_k 件を返す
    node_scores.sort(key=lambda x: x[1], reverse=True)
    return node_scores[:top_k]


# ==================================================
# 使用例（前回のコードと組み合わせる場合）
# ==================================================
if __name__ == "__main__":
    # 埋め込みモデルのロード（例）
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 既存の graph_db を利用（前回のセルで構築したもの）
    # graph_db = nx.DiGraph() ...（省略）
    
    # クエリ例
    query = "What runs on cloud hardware?"
    
    # ノード検索の実行
    top_nodes = search_graph_nodes(query, graph_db, embed_model, top_k=3)
    
    print(f"クエリ: '{query}'")
    for rank, (node, score) in enumerate(top_nodes, 1):
        print(f"{rank}位: {node} (類似度: {score:.4f})")

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =====================================================================
# 1. ノード検索（ベクトル類似度）
# =====================================================================
def search_graph_nodes(query, graph, embed_model, top_k=10):
    """
    クエリに最も近いノードをグラフから検索（ベクトル類似度ベース）
    """
    query_vector = embed_model.encode(query)
    node_scores = []
    
    for node, data in graph.nodes(data=True):
        node_vector = data["embedding"]
        similarity = np.dot(query_vector, node_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(node_vector)
        )
        node_scores.append((node, similarity))
    
    node_scores.sort(key=lambda x: x[1], reverse=True)
    return node_scores[:top_k]


# =====================================================================
# 2. LLMによるノードリランキング
# =====================================================================
def rerank_nodes_with_llm(query, candidate_nodes, llm, top_k=3):
    """
    LLMを使って、クエリとの関連度でノードをリランキングする
    """
    # 候補ノードをテキストに変換
    nodes_text = "\n".join([f"- {node}" for node, _ in candidate_nodes])
    
    prompt = f"""
[Instruction]
You are a reranking model. Given a user query and a list of candidate nodes from a knowledge graph,
output the top {top_k} nodes that are most relevant to the query, in order of relevance.

[Query]
{query}

[Candidate Nodes]
{nodes_text}

[Output Format]
Return only a numbered list of node names, like:
1. NodeName1
2. NodeName2
...
"""

    outputs = llm(prompt, max_new_tokens=200, temperature=0.0)
    generated_text = outputs[0]["generated_text"]
    
    # 生成されたテキストからノード名を抽出（簡易実装）
    # 実際には正規表現やJSON出力など、より堅牢なパースを推奨
    lines = generated_text.strip().split("\n")
    reranked_nodes = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            # "1. NodeName" のような形式を想定
            parts = line.split(".", 1)
            if len(parts) == 2:
                node_name = parts[1].strip()
                reranked_nodes.append(node_name)
    
    # 上位 top_k に絞る
    return reranked_nodes[:top_k]


# =====================================================================
# 3. リランキング済みノードからサブグラフを抽出
# =====================================================================
def retrieve_subgraph_from_nodes(nodes, graph, max_hops=1):
    """
    リランキング済みのノードリストから、周辺のサブグラフ（トリプレット）を抽出する
    max_hops: 中心ノードから何ホップ先まで辿るか（1なら直接の隣接ノードまで）
    """
    extracted_facts = set()
    
    for node in nodes:
        # BFS的にエッジを辿る（簡易版）
        visited = set([node])
        queue = [(node, 0)]  # (current_node, hop_count)
        
        while queue:
            current, hop = queue.pop(0)
            if hop > max_hops:
                continue
            
            # 自分が主語の関係（自分 → 他ノード）
            for successor in graph.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    relation = graph[current][successor]["relation"]
                    extracted_facts.add(f"- {current} -> ({relation}) -> {successor}")
                    queue.append((successor, hop + 1))
            
            # 自分が目的語の関係（他ノード → 自分）
            for predecessor in graph.predecessors(current):
                if predecessor not in visited:
                    visited.add(predecessor)
                    relation = graph[predecessor][current]["relation"]
                    extracted_facts.add(f"- {predecessor} -> ({relation}) -> {current}")
                    queue.append((predecessor, hop + 1))
    
    return list(extracted_facts)


# =====================================================================
# 4. Graph RAG + リランキング システム
# =====================================================================
class GraphRAGWithRerankingSystem:
    def __init__(self, graph_db, embed_model):
        self.graph_db = graph_db
        self.embed_model = embed_model
        
        # リランキング用の軽量LLM（例）
        self.rerank_llm = pipeline(
            "text-generation",
            model="FilippoMedia/TinyLlama-1.1B-Chat-v1.0-miniguanaco",
            max_new_tokens=200,
            temperature=0.0  # リランキングは温度0推奨
        )
        
        # 回答生成用のLLM（例）
        self.answer_llm = pipeline(
            "text-generation",
            model="FilippoMedia/TinyLlama-1.1B-Chat-v1.0-miniguanaco",
            max_new_tokens=150,
            temperature=0.1
        )

    def ask(self, question, top_k_search=10, top_k_rerank=3, max_hops=1):
        """
        Graph RAG + リランキングの一連の流れ
        """
        # 1. ノード検索（ベクトル類似度）
        candidate_nodes = search_graph_nodes(question, self.graph_db, self.embed_model, top_k=top_k_search)
        print(f"[1. Node Search] 候補ノード（類似度順）:")
        for node, score in candidate_nodes:
            print(f"  - {node} (score: {score:.4f})")
        
        # 2. LLMによるリランキング
        reranked_nodes = rerank_nodes_with_llm(
            question,
            candidate_nodes,
            self.rerank_llm,
            top_k=top_k_rerank
        )
        print(f"\n[2. Reranking] リランキング結果（上位{top_k_rerank}件）:")
        for i, node in enumerate(reranked_nodes, 1):
            print(f"  {i}. {node}")
        
        # 3. リランキング済みノードからサブグラフ抽出
        facts = retrieve_subgraph_from_nodes(reranked_nodes, self.graph_db, max_hops=max_hops)
        print(f"\n[3. Subgraph Retrieval] 抽出されたトリプレット数: {len(facts)}")
        for fact in facts:
            print(f"  {fact}")
        
        # 4. LLMによる回答生成
        context_str = "\n".join(facts)
        prompt = f"""[INST] You are a helpful assistant. Answer the question based ONLY on the provided Knowledge Graph facts.
If the facts do not contain the answer, say "I don't know".

[Knowledge Graph Facts]
{context_str}

[Question]
{question} [/INST]
[Answer]"""
        
        outputs = self.answer_llm(prompt)
        generated_text = outputs[0]["generated_text"]
        answer = generated_text.split("[Answer]")[-1].strip()
        
        return answer


# =====================================================================
# 使用例
# =====================================================================
if __name__ == "__main__":
    # 事前に graph_db, embed_model を構築済みと仮定
    # graph_db = nx.DiGraph() ...（前回のコードを参照）
    # embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    rag_sys = GraphRAGWithRerankingSystem(graph_db, embed_model)
    
    question = "Explain the relationship between GraphRAG and Python."
    answer = rag_sys.ask(
        question,
        top_k_search=10,
        top_k_rerank=3,
        max_hops=1  # 中心ノードから1ホップ先まで探索
    )
    
    print("\n=== Final Answer ===")
    print(answer)