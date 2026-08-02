import torch
import torch.nn.functional as F


def compute_similarity_and_accuracy(
    region_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """RegionCLIP の領域特徴量とテキスト特徴量間の類似度行列および Top-1 Accuracy を計算する関数

    Args:
        region_features (torch.Tensor): 領域特徴量行列 [N, d]
        text_features (torch.Tensor): テキスト特徴量行列 [M, d]
        labels (torch.Tensor): 各領域に対する正解テキストのインデックス [N]
        temperature (float): スケーリング用の温度パラメータ（デフォルト: 1.0）

    Returns:
        similarity_matrix (torch.Tensor): コサイン類似度行列 [N, M]
        predictions (torch.Tensor): 各領域に対する予測テキストインデックス [N]
        top1_accuracy (float): Top-1 精度 (0.0 ～ 1.0)
    """
    # 1. 特徴量の L2 正規化 (コサイン類似度の前処理)
    v_norm = F.normalize(region_features, p=2, dim=-1)  # [N, d]
    t_norm = F.normalize(text_features, p=2, dim=-1)  # [M, d]

    # 2. 類似度行列 (Cosine Similarity) の計算: S = (v_norm @ t_norm^T) / temperature
    # 行列積: [N, d] @ [d, M] -> [N, M]
    similarity_matrix = torch.matmul(v_norm, t_norm.T) / temperature

    # 3. Open-Vocabulary 識別 (各領域に対して類似度が最大のテキストを選択)
    # dim=1 (テキスト軸) 方向の argmax
    predictions = torch.argmax(similarity_matrix, dim=1)  # [N]

    # 4. Top-1 Accuracy の算出
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    top1_accuracy = correct / total

    return similarity_matrix, predictions, top1_accuracy


# ==========================================
# 動作検証用サンプルコード
# ==========================================
if __name__ == "__main__":
    # 乱数シードの固定
    torch.manual_seed(42)

    # パラメータ設定
    N = 4  # 領域の数 (b_1: red mug, b_2: white plate, b_3: blue mug, b_4: green apple)
    M = 4  # テキストの数 (t_1, t_2, t_3, t_4)
    d = 512  # 埋め込み次元数

    # --- モックデータの作成 ---
    # 実際のスクリプトでは、RegionCLIPモデルの Encoder 出力を代入します
    # ここでは、正解ペアのコサイン類似度が高くなるようなダミーベクトルを生成します

    # ランダムな基底ベクトルから構築
    base_text_feats = torch.randn(M, d)
    # テキスト特徴量 (正規化済み)
    text_features = F.normalize(base_text_feats, p=2, dim=-1)

    # 領域特徴量 (正解テキストに若干のノイズを加味して生成)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # 各領域の正解インデックス
    noise_level = 0.3
    region_features = (
        text_features[labels] + torch.randn(N, d) * noise_level
    )

    # --- 計算実行 ---
    sim_matrix, preds, acc = compute_similarity_and_accuracy(
        region_features=region_features,
        text_features=text_features,
        labels=labels,
    )

    # --- 結果の表示 ---
    print("=== 類似度行列 S (N x M) ===")
    # 見やすく小数点以下3桁で表示
    print(torch.round(sim_matrix * 1000) / 1000)
    print("\n" + "=" * 30)

    print(f"正解ラベル (Ground Truth): {labels.tolist()}")
    print(f"予測結果   (Predictions) : {preds.tolist()}")
    print(f"Top-1 Accuracy          : {acc * 100:.2f}%")


    import torch
import torch.nn.functional as F


def compute_similarity_and_accuracy(
    region_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """RegionCLIP の領域特徴量とテキスト特徴量間の類似度行列および Top-1 Accuracy を計算する関数

    Args:
        region_features (torch.Tensor): 領域特徴量行列 [N, d]
        text_features (torch.Tensor): テキスト特徴量行列 [M, d]
        labels (torch.Tensor): 各領域に対する正解テキストのインデックス [N]
        temperature (float): スケーリング用の温度パラメータ（デフォルト: 1.0）

    Returns:
        similarity_matrix (torch.Tensor): コサイン類似度行列 [N, M]
        predictions (torch.Tensor): 各領域に対する予測テキストインデックス [N]
        top1_accuracy (float): Top-1 精度 (0.0 ～ 1.0)
    """
    # 1. 特徴量の L2 正規化 (コサイン類似度の前処理)
    v_norm = F.normalize(region_features, p=2, dim=-1)  # [N, d]
    t_norm = F.normalize(text_features, p=2, dim=-1)  # [M, d]

    # 2. 類似度行列 (Cosine Similarity) の計算: S = (v_norm @ t_norm^T) / temperature
    # 行列積: [N, d] @ [d, M] -> [N, M]
    similarity_matrix = torch.matmul(v_norm, t_norm.T) / temperature

    # 3. Open-Vocabulary 識別 (各領域に対して類似度が最大のテキストを選択)
    # dim=1 (テキスト軸) 方向の argmax
    predictions = torch.argmax(similarity_matrix, dim=1)  # [N]

    # 4. Top-1 Accuracy の算出
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    top1_accuracy = correct / total

    return similarity_matrix, predictions, top1_accuracy


# ==========================================
# 動作検証用サンプルコード
# ==========================================
if __name__ == "__main__":
    # 乱数シードの固定
    torch.manual_seed(42)

    # パラメータ設定
    N = 4  # 領域の数 (b_1: red mug, b_2: white plate, b_3: blue mug, b_4: green apple)
    M = 4  # テキストの数 (t_1, t_2, t_3, t_4)
    d = 512  # 埋め込み次元数

    # --- モックデータの作成 ---
    # 実際のスクリプトでは、RegionCLIPモデルの Encoder 出力を代入します
    # ここでは、正解ペアのコサイン類似度が高くなるようなダミーベクトルを生成します

    # ランダムな基底ベクトルから構築
    base_text_feats = torch.randn(M, d)
    # テキスト特徴量 (正規化済み)
    text_features = F.normalize(base_text_feats, p=2, dim=-1)

    # 領域特徴量 (正解テキストに若干のノイズを加味して生成)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # 各領域の正解インデックス
    noise_level = 0.3
    region_features = (
        text_features[labels] + torch.randn(N, d) * noise_level
    )

    # --- 計算実行 ---
    sim_matrix, preds, acc = compute_similarity_and_accuracy(
        region_features=region_features,
        text_features=text_features,
        labels=labels,
    )

    # --- 結果の表示 ---
    print("=== 類似度行列 S (N x M) ===")
    # 見やすく小数点以下3桁で表示
    print(torch.round(sim_matrix * 1000) / 1000)
    print("\n" + "=" * 30)

    print(f"正解ラベル (Ground Truth): {labels.tolist()}")
    print(f"予測結果   (Predictions) : {preds.tolist()}")
    print(f"Top-1 Accuracy          : {acc * 100:.2f}%")