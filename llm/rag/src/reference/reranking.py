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
