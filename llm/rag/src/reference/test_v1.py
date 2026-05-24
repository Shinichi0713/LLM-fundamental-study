import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple

# モデル名（例：bge-reranker-large）
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"


tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
model.eval()  # 推論モード

def rerank_text_pairs(
    query: str,
    documents: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    質問と文書のペアをスコアリングし、スコア順にソートして上位k件を返す。
    """
    scored_docs = []

    for doc in documents:
        # 入力フォーマット：query + [SEP] + document
        inputs = tokenizer(
            query,
            doc,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            # スコア（ロジット）を取得（大きいほど関連度が高い）
            score = outputs.logits[0, 0].item()

        scored_docs.append((doc, score))

    # スコアで降順ソート
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]

