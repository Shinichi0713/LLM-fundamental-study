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