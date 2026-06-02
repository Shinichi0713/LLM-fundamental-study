import os
import base64
from PIL import Image
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import MeCab

# 環境変数の読み込みとクライアント初期化
load_dotenv()
client = OpenAI()

# 日本語分かち書き用（BM25）
wakati = MeCab.Tagger("-Owbakati")

def tokenize_japanese(text):
    return wakati.parse(text).strip().split()

def encode_image_to_base64(image_path):
    """画像をBase64文字列に変換する関数"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ---------------------------------------------------------
# 1. 画像から詳細なメタデータ（説明テキスト）を生成 (Image-to-Text)
# ---------------------------------------------------------
def generate_image_metadata(image_path):
    print(f"[{image_path}] の解析メタデータを生成中...")
    base64_image = encode_image_to_base64(image_path)
    
    prompt = """
    この画像に含まれる情報を、RAG（検索システム）のインデックス用テキストとして詳細に言語化してください。
    以下の項目を必ず含めてください：
    1. 画像の種類（グラフ、インフォグラフィック、フローチャート、製品写真など）
    2. 写っている対象や、テーマの要約
    3. 画像内のすべてのテキスト、数字、型番、固有名詞（完全一致検索に必要です）
    4. グラフや表の場合は、軸の説明や目立つデータ傾向
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 視覚能力が高く、コスト効率の良いモデルを選択
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    metadata_text = response.choices[0].message.content
    return metadata_text

# ---------------------------------------------------------
# 2. データベースの構築クラス（BM25 + FAISS）
# ---------------------------------------------------------
class HybridImageRegistry:
    def __init__(self):
        # ベクトル検索用モデル
        self.bi_encoder = SentenceTransformer('bzk/ja-sentence-transformer-v1')
        self.image_paths = []
        self.metadatas = []
        self.faiss_index = None
        self.bm25 = None
        
    def register_images(self, image_list):
        """画像リストを読み込み、メタデータ生成とインデックス登録を行う"""
        for img_path in image_list:
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} が見つかりません。スキップします。")
                continue
                
            # メタデータの自動生成
            metadata = generate_image_metadata(img_path)
            
            self.image_paths.append(img_path)
            self.metadatas.append(metadata)
            print(f"-> メタデータ生成完了:\n{metadata[:100]}...\n")
            
        # A. FAISSインデックス（ベクトル）の構築
        embeddings = self.bi_encoder.encode(self.metadatas)
        embeddings = np.array(embeddings).astype('float32')
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        
        # B. BM25（キーワード）の構築
        tokenized_docs = [tokenize_japanese(text) for text in self.metadatas]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"データベース登録完了: {len(self.image_paths)} 件の画像をインデックスしました。")

    def search(self, query, top_k_pool=2, top_k_final=1):
        """ハイブリッド検索の実行"""
        print(f"\n【検索クエリ】: {query}")
        
        # 1. ベクトル検索 (意味の類似性)
        query_vector = self.bi_encoder.encode([query]).astype('float32')
        _, dense_indices = self.faiss_index.search(query_vector, min(top_k_pool, len(self.image_paths)))
        dense_hits = [dense_indices[0][i] for i in range(len(dense_indices[0])) if dense_indices[0][i] != -1]
        
        # 2. キーワード検索 (型番や固有名詞の完全一致)
        tokenized_query = tokenize_japanese(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # スコアが高い順のインデックスを取得
        sparse_hits = np.argsort(bm25_scores)[::-1][:top_k_pool].tolist()
        # スコアが0のものは除外
        sparse_hits = [idx for idx in sparse_hits if bm25_scores[idx] > 0]
        
        # 3. マージ（重複排除）
        # 簡易的なReciprocal Rank Fusion (RRF) 的なアプローチや、
        # 両方に引っかかったものを優先するロジック、あるいは単に結合
        candidate_indices = list(set(dense_hits + sparse_hits))
        
        print(f"-> ベクトル検索ヒット(インデックス): {dense_hits}")
        print(f"-> キーワード検索ヒット(インデックス): {sparse_hits}")
        print(f"-> 統合された候補数: {len(candidate_indices)}")
        
        # 本来はここで詳細なリランカー（Cross-Encoder）を通すのがベストですが、
        # 今回は簡易的に、ベクトル距離とBM25スコアが上位のものを最終候補として抽出します
        results = []
        for idx in candidate_indices[:top_k_final]:
            results.append({
                "image_path": self.image_paths[idx],
                "metadata": self.metadatas[idx]
            })
            
        return results

# ---------------------------------------------------------
# 3. 実行テスト
# ---------------------------------------------------------
if __name__ == "__main__":
    # 事前に適当な画像ファイルを用意してください。
    # ここでは例として2つの画像ファイルを登録します。
    # ※テスト実行時は、実際に存在する画像ファイル名に書き換えてください。
    test_images = ["sales_chart_2025.png", "server_topology_v3.png"]
    
    # テスト用のダミー画像を作成（ファイルが存在しない場合のみ）
    for img_path in test_images:
        if not os.path.exists(img_path):
            img = Image.new('RGB', (300, 300), color = (73, 109, 137))
            img.save(img_path)
            print(f"テスト用のダミー画像を作成しました: {img_path}")
            print("※高精度な検索をテストする場合、実際のグラフや文字が含まれる画像に差し替えてください。")

    # システムの初期化と画像登録
    registry = HybridImageRegistry()
    registry.register_images(test_images)
    
    # 検索テスト
    # 画像内のテキストや、LLMが解釈しそうな特徴をもとに検索をかけます
    query = "2025年の売上推移がわかるグラフの画像はどれですか？"
    search_results = registry.search(query, top_k_final=1)
    
    print("\n【最も関連度の高い画像】")
    for res in search_results:
        print(f"ファイルパス: {res['image_path']}")
        print(f"生成されたメタデータ抜粋:\n{res['metadata'][:200]}...")


import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# 環境変数の読み込みとOpenAIクライアントの初期化
load_dotenv()
client = OpenAI()

# 1次検索用のベクトル埋め込みモデル
print("埋め込みモデルをロード中...")
bi_encoder = SentenceTransformer('bzk/ja-sentence-transformer-v1')

# ---------------------------------------------------------
# [事前準備] 前述のImage-to-Text関数
# ---------------------------------------------------------
def generate_image_metadata(image_path):
    import base64
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    
    prompt = "この画像に含まれるテキスト、数値、グラフのテーマ、特徴をRAG検索用インデックスとして詳細に言語化してください。"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=600
    )
    return response.choices[0].message.content

# ---------------------------------------------------------
# 2. データベースの構築と永続化（ローカル保存）
# ---------------------------------------------------------
def build_and_save_database(image_paths, output_dir="rag_db"):
    """
    image_paths: 登録したい画像ファイルのパスのリスト
    output_dir: データベースファイルを保存するディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generated_metadatas = []
    valid_image_paths = []
    
    # Step 2-1: 各画像のテキストメタデータを生成
    for path in image_paths:
        if not os.path.exists(path):
            print(f"スキップ (ファイル不在): {path}")
            continue
        try:
            metadata = generate_image_metadata(path)
            generated_metadatas.append(metadata)
            valid_image_paths.append(path)
            print(f"成功: {path} のメタデータを生成しました。")
        except Exception as e:
            print(f"エラー ({path}): {e}")
            
    if not valid_image_paths:
        print("登録可能な画像がありませんでした。処理を中断します。")
        return

    # Step 2-2: テキストをベクトルに変換
    print("\nテキストをベクトル化しています...")
    embeddings = bi_encoder.encode(generated_metadatas)
    embeddings = np.array(embeddings).astype('float32')
    
    # Step 2-3: FAISS インデックスの構築と保存
    vector_dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(vector_dim)
    faiss_index.add(embeddings)
    
    faiss_path = os.path.join(output_dir, "image_vectors.faiss")
    faiss.write_index(faiss_index, faiss_path)
    print(f"-> ベクトルDBを保存しました: {faiss_path}")
    
    # Step 2-4: 画像パスとメタデータテキストのペアを保存 (マッピング用)
    # BM25のインデックスや最終結果の取得にこれを使用します
    store_data = {
        "image_paths": valid_image_paths,
        "metadatas": generated_metadatas
    }
    store_path = os.path.join(output_dir, "image_store.pkl")
    with open(store_path, "wb") as f:
        pickle.dump(store_data, f)
    print(f"-> メタデータストアを保存しました: {store_path}")
    print("\nデータベースの構築がすべて完了しました！")

# ---------------------------------------------------------
# 3. 構築のテスト実行
# ---------------------------------------------------------
if __name__ == "__main__":
    # データベースに登録したい手元の画像ファイル名を指定してください
    target_images = ["sales_chart_2025.png", "server_topology_v3.png"]
    
    # ダミーファイルの作成（テスト用）
    from PIL import Image
    for img in target_images:
        if not os.path.exists(img):
            Image.new('RGB', (100, 100), color = 'blue').save(img)
            
    build_and_save_database(target_images)

    