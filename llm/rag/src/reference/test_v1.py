import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import requests

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

query = "What are the benefits of exercise?"
docs = [
    "Exercise improves cardiovascular health.",
    "Cats are independent animals.",
    "Regular physical activity reduces the risk of chronic diseases.",
    "Dogs are loyal companions.",
]

ranked = rerank_text_pairs(query, docs, top_k=2)
for i, (doc, score) in enumerate(ranked):
    print(f"{i+1}. Score: {score:.3f} | {doc}")


import openai
from typing import List, Dict

client = openai.OpenAI(api_key="your-api-key")

Candidate = Dict[str, str]  # {"type": "text" or "image", "content": str, "id": str}

def mllm_rerank_openai(
    query_text: str,
    candidates: List[Candidate],
    top_k: int = 3,
    model: str = "gpt-4-vision-preview"
) -> List[Candidate]:
    """
    GPT-4Vを使って候補を再ランキングする。
    各候補について「質問との関連度」を0〜1のスコアで評価し、上位k件を返す。
    """
    scored_candidates = []

    for cand in candidates:
        # プロンプト構築
        if cand["type"] == "text":
            prompt = f"""
質問: {query_text}
候補テキスト: {cand['content']}

この候補テキストは質問に関連していますか？
関連している場合は1、関連していない場合は0で答えてください。
数値のみを出力してください。
"""
            messages = [{"role": "user", "content": prompt}]
        else:  # image
            prompt = f"""
質問: {query_text}
この画像は質問に関連していますか？
関連している場合は1、関連していない場合は0で答えてください。
数値のみを出力してください。
"""
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": cand["content"]}}
                    ]
                }
            ]

        # GPT-4Vに問い合わせ
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=10
            )
            score_text = response.choices[0].message.content.strip()
            # "1" or "0" を数値に変換
            score = 1 if "1" in score_text else 0
        except Exception as e:
            print(f"Error scoring candidate {cand['id']}: {e}")
            score = 0  # エラー時は関連なしとみなす

        scored_candidates.append((cand, score))

    # スコアでソート（降順）
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return [cand for cand, score in scored_candidates[:top_k]]

# 例：LLaVA-1.5（7B）
model_name = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_name)
model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def mllm_rerank_llava(
    query_text: str,
    candidates: List[Candidate],
    top_k: int = 3
) -> List[Candidate]:
    """
    LLaVAを使って候補を再ランキングする。
    各候補について「質問との関連度」を0〜1のスコアで評価。
    """
    scored_candidates = []

    for cand in candidates:
        if cand["type"] == "text":
            # テキスト候補：プロンプトのみ
            prompt = f"""
質問: {query_text}
候補テキスト: {cand['content']}

この候補テキストは質問に関連していますか？
関連している場合は「1」、関連していない場合は「0」とだけ答えてください。
"""
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
        else:
            # 画像候補：画像＋プロンプト
            image = Image.open(requests.get(cand["content"], stream=True).raw)
            prompt = f"""
質問: {query_text}
この画像は質問に関連していますか？
関連している場合は「1」、関連していない場合は「0」とだけ答えてください。
"""
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        # 生成（短い回答のみ）
        generate_ids = model.generate(**inputs, max_new_tokens=10)
        response_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

        # スコア抽出（"1" or "0"）
        score = 1 if "1" in response_text else 0
        scored_candidates.append((cand, score))

    # スコアでソート（降順）
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return [cand for cand, score in scored_candidates[:top_k]]


# 1段目検索で得た候補（例）
candidates = [
    {"type": "text", "content": "Cats are small domesticated mammals.", "id": "text_1"},
    {"type": "text", "content": "Dogs are loyal pets.", "id": "text_2"},
    {"type": "image", "content": "https://example.com/cat.jpg", "id": "img_1"},
    {"type": "image", "content": "https://example.com/dog.jpg", "id": "img_2"},
]

query = "Tell me about cats."

# MLLMリランキング
ranked = mllm_rerank_openai(query, candidates, top_k=2)

print("Ranked candidates:")
for i, cand in enumerate(ranked):
    print(f"{i+1}. {cand['type']}: {cand['content'][:50]}... (id: {cand['id']})")