import os
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# .envファイルの読み込み
load_dotenv()

# APIキーの確認
if not os.getenv("OPENAI_AI_KEY") and not os.getenv("OPENAI_API_KEY"):
    print("Warning: OpenAI API KEY が設定されていません。.env ファイルを確認してください。")
else:
    print("Environment setup environment: OK")

# 埋め込みモデルの初期化（日本語に強い軽量モデル）
print("Loading embedding model...")
embed_model = SentenceTransformer('bzk/ja-sentence-transformer-v1') 
print("Model loaded successfully!")

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 埋め込みモデルの初期化（日本語に対応したモデル）
print("埋め込みモデルを読み込んでいます...")
model = SentenceTransformer('bzk/ja-sentence-transformer-v1')

# 2. ドキュメントの読み込み
# ※事前に documents.txt を作成し、1行に1つの知識（文章）を書き込んでおいてください。
document_path = "documents.txt"

if not os.path.exists(document_path):
    # テスト用の簡易ファイル作成（ファイルがない場合）
    with open(document_path, "w", encoding="utf-8") as f:
        f.write("スマートフォンの次世代通信規格「6G」は、2030年頃の商用化を目指して開発が進んでいる。\n")
        f.write("株式会社TECH-AIの福利厚生では、1年に1回、上限10万円までの旅行費補助が出る。\n")
        f.write("日本の新元号「未来（みらい）」は、2028年から施行される予定である。\n")

print(f"'{document_path}' からドキュメントを読み込んでいます...")
with open(document_path, "r", encoding="utf-8") as f:
    # 空行を除外して、行ごとにリスト化（これが最もシンプルなチャンク分割になります）
    documents = [line.strip() for line in f if line.strip()]

print(f"読み込み完了: {len(documents)} 件のテキスト")

# 3. テキストのベクトル化（Embedding）
print("テキストをベクトルに変換中...")
embeddings = model.encode(documents)
# FAISSで扱うために、float32型のNumPy配列に変換
embeddings = np.array(embeddings).astype('float32')

# ベクトルの次元数を確認（モデルによって異なります。例：384次元や768次元など）
vector_dimension = embeddings.shape[1]
print(f"ベクトルの次元数: {vector_dimension}")

# 4. ベクトルデータベース（FAISSインデックス）の構築
# 最もシンプルな「L2距離（ユークリッド距離）」による全探索インデックスを作成
index = faiss.IndexFlatL2(vector_dimension)

# ベクトルデータをデータベースに登録
index.add(embeddings)
print(f"FAISSインデックスに {index.ntotal} 件のベクトルを登録しました。")

# 5. インデックスとテキストデータの保存
# 検索時に元の文章を取り出せるよう、テキストのリストも一緒に保存するか、メモリ上に保持します。
faiss.write_index(index, "my_rag_index.faiss")

# 元のテキストデータも紐付け用に保存
with open("indexed_documents.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc + "\n")

print("データベースの構築と保存が完了しました！ ('my_rag_index.faiss' が生成されました)")

import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------
# 1. 各種モデルとデータの準備
# ---------------------------------------------------------
print("モデルとデータベースを読み込んでいます...")

# ベクトル検索用（Bi-Encoder）モデルの読み込み
bi_encoder = SentenceTransformer('bzk/ja-sentence-transformer-v1')

# リランキング用（Cross-Encoder）モデルの読み込み
# 日本語に対応した軽量・高性能なリランカーモデルを指定
reranker_model_name = "hotamsh/koheidb-modernbert-japanese-reranker-base"
rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
rerank_model.eval()  # 推論モードに設定

# FAISSインデックスとテキストデータの読み込み
index = faiss.read_index("my_rag_index.faiss")

with open("indexed_documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

# ---------------------------------------------------------
# 2. 検索 ＆ リランキング関数
# ---------------------------------------------------------
def retrieve_and_rerank(query, top_k_initial=10, top_k_final=2):
    """
    query: ユーザーの質問
    top_k_initial: FAISSで最初に多めに取得する件数
    top_k_final: リランキング後に最終的にLLMに渡す件数
    """
    print(f"\n質問: {query}")
    
    # --- Step 2-1: 1次検索 (FAISSによるベクトル検索) ---
    query_vector = bi_encoder.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    
    # 登録件数を超えないように調整
    k = min(top_k_initial, len(documents))
    distances, indices = index.search(query_vector, k)
    
    # 検索結果のテキストを抽出
    retrieved_docs = [documents[idx] for idx in indices[0] if idx != -1]
    
    if not retrieved_docs:
        print("該当するドキュメントが見つかりませんでした。")
        return []
        
    print(f"-> 1次検索（FAISS）で {len(retrieved_docs)} 件の候補を抽出しました。")

    # --- Step 2-2: 2次検索 (Cross-Encoderによるリランキング) ---
    # 質問と各候補ドキュメントのペアを作成
    pairs = [[query, doc] for doc in retrieved_docs]
    
    # モデルの入力形式にトークナイズ
    inputs = rerank_tokenizer(
        pairs, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    
    # スコア（類似度・関連度）の計算
    with torch.no_grad():
        outputs = rerank_model(**inputs)
        # 通常、モデルの最初の出力（logits）の1次元目が関連度スコアになります
        # モデルの実装によっては、二値分類(0か1か)の確率をスコアとして使う場合もあります
        scores = outputs.logits.view(-1).tolist()
    
    # ドキュメントとスコアを紐付けて、スコアが高い順にソート
    reranked_results = sorted(
        zip(retrieved_docs, scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # --- Step 2-3: 結果の出力 ---
    print("\n--- リランキング結果 ---")
    for i, (doc, score) in enumerate(reranked_results):
        # スコアを表示（値が高いほど質問に対する関連度が高い）
        print(f"順位 {i+1} (Score: {score:.4f}): {doc}")
        
    # 最終的にLLMへ渡す上位 N 件を返す
    return [doc for doc, score in reranked_results[:top_k_final]]

# ---------------------------------------------------------
# 3. テスト実行
# ---------------------------------------------------------
if __name__ == "__main__":
    # テスト用の質問（福利厚生について）
    user_query = "旅行の補助金はいくらまで出ますか？"
    
    # 1次検索で3件取り出し、リランキングして最終的に上位1件に絞り込む
    final_context = retrieve_and_rerank(user_query, top_k_initial=3, top_k_final=1)
    
    print("\n--- LLMに渡される最終コンテキスト ---")
    print(final_context)

import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 日本語の分かち書き（BM25用）にMeCabを使用
import MeCab
wakati = MeCab.Tagger("-Owavati")

def tokenize_japanese(text):
    """日本語のテキストを単語リストに分割する（BM25用）"""
    return wakati.parse(text).strip().split()

# ---------------------------------------------------------
# 1. テスト用ドキュメントデータの準備
# ---------------------------------------------------------
documents = [
    "株式会社TECH-AIの福利厚生では、1年に1回、上限10万円までの旅行費補助が出る。申請は社内ポータルから行う。",
    "スマートフォンの次世代通信規格「6G」は、2030年頃の商用化を目指して世界中で開発が進んでいる。",
    "日本の新元号「未来（みらい）」は、2028年から施行される予定であるとの誤報が流れた。",
    "TECH-AI社では、リモートワーク手当として毎月5,000円が全社員に支給される。",
    "旅行費の補助金（出張旅費規程）に関する問い合わせは、総務部の佐藤さんまで連絡してください。"
]

print(f"データベースに {len(documents)} 件のドキュメントを登録中...")

# ---------------------------------------------------------
# 2. 検索インデックスの構築 (Dense & Sparse)
# ---------------------------------------------------------
# A. ベクトル検索 (Dense) の準備
bi_encoder = SentenceTransformer('bzk/ja-sentence-transformer-v1')
embeddings = bi_encoder.encode(documents)
embeddings = np.array(embeddings).astype('float32')
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# B. キーワード検索 (Sparse / BM25) の準備
tokenized_docs = [tokenize_japanese(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# C. 最先端リランカー (BGE-Reranker-v2-m3) の準備
# ※非常に強力な多言語・日本語対応リランカーです
reranker_name = "BAAI/bge-reranker-v2-m3"
rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)
rerank_model.eval()

print("全ての検索インデックスとリランキングモデルの準備が完了しました。\n" + "="*50)

# ---------------------------------------------------------
# 3. 高精度ハイブリッド検索 ＆ リランキング関数
# ---------------------------------------------------------
def advanced_retrieve_and_rerank(query, top_k_pool=4, top_k_final=2):
    """
    query: ユーザーの質問
    top_k_pool: 各検索手法から集める候補数（多めに取る）
    top_k_final: リランキング後に最終的に残す件数
    """
    print(f"\n【ユーザーの質問】: {query}\n")
    
    # --- Step 3-1: ベクトル検索 (Dense Retrieval) ---
    query_vector = bi_encoder.encode([query]).astype('float32')
    _, dense_indices = index.search(query_vector, min(top_k_pool, len(documents)))
    dense_candidates = [documents[idx] for idx in dense_indices[0] if idx != -1]
    
    # --- Step 3-2: キーワード検索 (BM25 / Sparse Retrieval) ---
    tokenized_query = tokenize_japanese(query)
    # BM25のスコアが高い順にドキュメントを取得
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_pool]
    sparse_candidates = [documents[idx] for idx in bm25_indices if bm25_scores[idx] > 0]
    
    # --- Step 3-3: 候補の重複排除（マージ） ---
    # 両方の検索結果を合わせ、重複のないユニークなプールを作る
    candidate_pool = list(set(dense_candidates + sparse_candidates))
    print(f"-> ベクトル検索候補: {len(dense_candidates)}件 / キーワード検索候補: {len(sparse_candidates)}件")
    print(f"-> 重複排除後のリランク対象プール: {len(candidate_pool)}件")
    
    if not candidate_pool:
        print("候補ドキュメントが1件も見つかりませんでした。")
        return []

    # --- Step 3-4: BGE-Rerankerによる厳密なリランキング ---
    # 質問と候補文のペアを作成
    pairs = [[query, candidate] for candidate in candidate_pool]
    
    inputs = rerank_tokenizer(
        pairs, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        max_length=512
    )
    
    with torch.no_grad():
        outputs = rerank_model(**inputs)
        # BGE-Rerankerはlogitsの直値、またはシグモイドをかけた値がスコアになります
        # ここでは順位比較のため、そのままの数値を採用
        scores = outputs.logits.view(-1).tolist()
    
    # スコアで降順ソート
    reranked_results = sorted(
        zip(candidate_pool, scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # --- Step 3-5: 結果の可視化 ---
    print("\n【リランキング結果（スコア順）】")
    for i, (doc, score) in enumerate(reranked_results):
        marker = "★" if i < top_k_final else "  "
        print(f"{marker} 順位 {i+1} (Score: {score:6.4f}): {doc}")
        
    return [doc for doc, score in reranked_results[:top_k_final]]

# ---------------------------------------------------------
# 4. 実行テスト
# ---------------------------------------------------------
if __name__ == "__main__":
    # テスト1: 「旅行費の補助」という文脈（意味）とキーワードの両方が絡む質問
    # 「旅行費補助」のある福利厚生(1つ目)と、佐藤さんへの連絡(5つ目)のどちらが上位に来るか？
    query_1 = "旅行の補助金について、申請はどこから行えばいいですか？"
    final_docs_1 = advanced_retrieve_and_rerank(query_1, top_k_pool=3, top_k_final=1)
    
    print("\n" + "="*50)
    
    # テスト2: 「TECH-AI」という特定の固有名詞キーワードが重要な質問
    query_2 = "TECH-AIのリモートワークの手当について教えて"
    final_docs_2 = advanced_retrieve_and_rerank(query_2, top_k_pool=3, top_k_final=1)