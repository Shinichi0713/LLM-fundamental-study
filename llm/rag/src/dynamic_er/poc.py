# ============================================================
# TempCCA 簡易版 PoC（Google Colab用）
# ============================================================

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. エンコーダの準備（軽量モデル）
model = SentenceTransformer('all-MiniLM-L6-v2')  # Colabでも高速

# 2. 模擬データ（社内RAGを想定）
# 時間セグメントT1: 立ち上げ期
mentions_t1 = [
    "新規プロジェクトの猫のやつ",
    "Project Neko",
    "ネコ型AI開発",
]
# 既知エンティティ（KB登録済み）
entities = {
    "Q001": "Project Nekoは社内の新規AI開発プロジェクト。コードネームはNeko。",
    "Q002": "Lighthouseは既存の検索基盤プロジェクト。",
    "Q003": "山田太郎は営業部の部長。",
}

# 3. ベクトル化
mention_vecs_t1 = model.encode(mentions_t1)
entity_vecs = {qid: model.encode(desc) for qid, desc in entities.items()}

# 4. 静的エンティティ表現
static_entity_vecs = np.array(list(entity_vecs.values()))
entity_ids = list(entity_vecs.keys())

# 5. T1: クラスタリング（単純な最近傍）
def resolve_mentions(mention_vecs, entity_vecs, entity_ids, alpha=0.5, prev_mentions=None):
    """
    TempCCAの簡易版
    alpha: 静的表現 vs 動的表現のバランス
    prev_mentions: 過去にリンクされたメンションのベクトルリスト
    """
    # エンティティの動的表現を計算
    dynamic_entity_vecs = []
    for i, qid in enumerate(entity_ids):
        static = entity_vecs[qid]
        if prev_mentions and qid in prev_mentions and len(prev_mentions[qid]) > 0:
            # 継続的適応: 過去メンションの平均を混ぜる
            dynamic = alpha * static + (1 - alpha) * np.mean(prev_mentions[qid], axis=0)
        else:
            dynamic = static
        dynamic_entity_vecs.append(dynamic)
    
    dynamic_entity_vecs = np.array(dynamic_entity_vecs)
    
    # Affinity計算（コサイン類似度）
    similarities = cosine_similarity(mention_vecs, dynamic_entity_vecs)
    
    # 各メンションを最も近いエンティティに割り当て
    assignments = []
    for i, sim_row in enumerate(similarities):
        best_idx = np.argmax(sim_row)
        assignments.append((mentions_t1[i], entity_ids[best_idx], sim_row[best_idx]))
    
    return assignments, dynamic_entity_vecs

# T1の解決
assignments_t1, dynamic_vecs_t1 = resolve_mentions(
    mention_vecs_t1, entity_vecs, entity_ids, alpha=0.7, prev_mentions=None
)

print("=== T1 結果 ===")
for mention, qid, score in assignments_t1:
    print(f"'{mention}' → {qid} (score: {score:.3f})")

# 6. T2: 新しいメンションが出現（略称が定着）
mentions_t2 = [
    "PNJ",  # Project Neko Junior? → いいえ、Project Nekoの略
    "猫PJ",  # 新しい呼び方
    "ネコプロ",
]

# T1でProject Neko(Q001)に割り当てられたメンションを収集
prev_mentions = {
    "Q001": [mention_vecs_t1[i] for i, (_, qid, _) in enumerate(assignments_t1) if qid == "Q001"],
    "Q002": [],
    "Q003": [],
}

mention_vecs_t2 = model.encode(mentions_t2)

assignments_t2, dynamic_vecs_t2 = resolve_mentions(
    mention_vecs_t2, entity_vecs, entity_ids, alpha=0.5, prev_mentions=prev_mentions
)

print("\n=== T2 結果（継続的適応あり） ===")
for mention, qid, score in assignments_t2:
    print(f"'{mention}' → {qid} (score: {score:.3f})")

# 7. 比比較：継続的適応なし（alpha=1.0 = 静的のみ）
assignments_t2_static, _ = resolve_mentions(
    mention_vecs_t2, entity_vecs, entity_ids, alpha=1.0, prev_mentions=None
)

print("\n=== T2 結果（継続的適応なし = 従来手法） ===")
for mention, qid, score in assignments_t2_static:
    print(f"'{mention}' → {qid} (score: {score:.3f})")