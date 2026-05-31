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