import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pyvis.network import Network

# -------------------------------------------------------------
# サンプルデータ設定（GraphDBのノード情報を定義）
# -------------------------------------------------------------
nodes_data = [
    {
        "id": "RegionCLIP",
        "label": "Model",
        "description": "CLIPを領域レベルの視覚表現へ拡張したモデル",
        "source_chunk": "RegionCLIP uses RoIAlign to extract region features...",
        "embedding": np.array([0.85, 0.12, 0.45, 0.91, 0.22])  # ダミーベクトル
    },
    {
        "id": "RoIAlign",
        "label": "Component",
        "description": "特徴マップから領域特徴量を切り出すモジュール",
        "source_chunk": "RoIAlign extracts region features from feature maps...",
        "embedding": np.array([0.78, 0.15, 0.50, 0.88, 0.30])
    },
    {
        "id": "SAM",
        "label": "Model",
        "description": "Metaが開発したセグメンテーション基礎モデル",
        "source_chunk": "SAM consists of Image Encoder, Prompt Encoder, Mask Decoder.",
        "embedding": np.array([-0.65, 0.88, -0.12, 0.05, 0.72])
    },
    {
        "id": "Image Encoder",
        "label": "Component",
        "description": "画像全体の高次元特徴量を抽出するViTバックボーン",
        "source_chunk": "Image Encoder extracts high-dimensional image embeddings.",
        "embedding": np.array([-0.58, 0.82, -0.08, 0.10, 0.65])
    }
]

edges_data = [
    {"source": "RegionCLIP", "target": "RoIAlign", "relation": "USES"},
    {"source": "SAM", "target": "Image Encoder", "relation": "HAS_COMPONENT"}
]


# =============================================================
# 1. グラフレイヤーの可視化（ネットワーク構造）
# =============================================================
def visualize_graph_structure(nodes, edges, output_html="graph_visualization.html"):
    """
    pyvis を用いてインタラクティブなグラフ構造（HTML）を生成
    """
    net = Network(height="450px", width="100%", notebook=False, directed=True)
    
    # ノード追加
    for node in nodes:
        # ホバー時に詳細（description）を表示するツールチップを設定
        hover_info = f"<b>{node['id']}</b> ({node['label']})<br>{node['description']}"
        net.add_node(
            node["id"], 
            label=node["id"], 
            title=hover_info, 
            group=node["label"]
        )
        
    # エッジ（リレーション）追加
    for edge in edges:
        net.add_edge(edge["source"], edge["target"], title=edge["relation"], label=edge["relation"])
        
    net.write_html(output_html)
    print(f"[1. グラフ構造] '{output_html}' として保存されました（ブラウザで閲覧可能）。")


# =============================================================
# 2. テキスト・属性プロパティの可視化（構造化テーブル）
# =============================================================
def visualize_node_properties(nodes):
    """
    pandas を用いて属性プロパティやメタデータを表形式で可視化
    """
    df_data = []
    for n in nodes:
        df_data.append({
            "Node ID": n["id"],
            "Type Label": n["label"],
            "Description": n["description"],
            "Source Text Chunk": n["source_chunk"],
            "Embedding Dim": len(n["embedding"])
        })
    
    df = pd.DataFrame(df_data)
    print("\n[2. 属性プロパティ・メタデータ一覧]")
    print(df.to_string(index=False))


# =============================================================
# 3. ベクトル空間レイヤーの可視化（PCAによる2次元プロット）
# =============================================================
def visualize_embeddings_space(nodes):
    """
    高次元のEmbeddingベクトルをPCAで2次元に削減し散布図プロット
    """
    embeddings = np.array([n["embedding"] for n in nodes])
    labels = [n["id"] for n in nodes]
    categories = [n["label"] for n in nodes]
    
    # PCAによる次元削減（5次元 -> 2次元）
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(7, 5))
    
    # カテゴリごとに色分け
    unique_categories = list(set(categories))
    for cat in unique_categories:
        idx = [i for i, c in enumerate(categories) if c == cat]
        plt.scatter(
            coords_2d[idx, 0], 
            coords_2d[idx, 1], 
            label=cat, 
            s=120
        )
        
    # 各点にノード名のラベルを付与
    for i, name in enumerate(labels):
        plt.annotate(
            name, 
            (coords_2d[i, 0], coords_2d[i, 1]), 
            xytext=(5, 5), 
            textcoords="offset points",
            fontsize=10,
            weight="bold"
        )
        
    plt.title("Visual Concept Space (Node Embeddings PCA)")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# 実行処理
# -------------------------------------------------------------
if __name__ == "__main__":
    # 必要なライブラリのインストール確認用の案内
    # pip install pyvis pandas matplotlib scikit-learn
    
    visualize_graph_structure(nodes_data, edges_data)
    visualize_node_properties(nodes_data)
    visualize_embeddings_space(nodes_data)
    