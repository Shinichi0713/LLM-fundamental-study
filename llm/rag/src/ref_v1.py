from sentence_transformers import SentenceTransformer
import chromadb

# テキスト埋め込みモデル
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDBクライアント
client = chromadb.Client()
text_collection = client.create_collection(name="text_docs")

# テキストチャンクとメタデータ（画像IDなど）を用意
text_chunks = [
    "This is a document about cats.",
    "Cats are small domesticated carnivorous mammals.",
    # ...
]
image_ids = ["img_001", "img_002", ...]  # 対応する画像ID

# 埋め込みと格納
text_embeddings = text_encoder.encode(text_chunks)
for i, (chunk, img_id) in enumerate(zip(text_chunks, image_ids)):
    text_collection.add(
        embeddings=[text_embeddings[i]],
        documents=[chunk],
        metadatas=[{"image_id": img_id}],
        ids=[f"text_{i}"]
    )

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

# CLIPモデル（テキスト＋画像の埋め込み）
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 画像コレクション
image_collection = client.create_collection(name="images")

# 画像URL or ローカルパス
image_urls = [
    "https://example.com/cat1.jpg",
    "https://example.com/cat2.jpg",
    # ...
]

for i, url in enumerate(image_urls):
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = clip_processor(images=image, return_tensors="pt")
    image_embedding = clip_model.get_image_features(**inputs).detach().numpy()[0]

    image_collection.add(
        embeddings=[image_embedding],
        metadatas=[{"url": url}],
        ids=[f"img_{i}"]
    )

def retrieve_multimodal(query_text, top_k=5):
    # テキスト埋め込み
    query_embedding = text_encoder.encode([query_text])[0]

    # テキスト検索
    text_results = text_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # 画像検索（CLIPのテキストエンコーダでクエリ埋め込み）
    text_inputs = clip_processor(text=query_text, return_tensors="pt", padding=True)
    query_image_embedding = clip_model.get_text_features(**text_inputs).detach().numpy()[0]

    image_results = image_collection.query(
        query_embeddings=[query_image_embedding],
        n_results=top_k
    )

    return {
        "text_results": text_results,
        "image_results": image_results
    }


def retrieve_multimodal(query_text, top_k=5):
    # テキスト埋め込み
    query_embedding = text_encoder.encode([query_text])[0]

    # テキスト検索
    text_results = text_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # 画像検索（CLIPのテキストエンコーダでクエリ埋め込み）
    text_inputs = clip_processor(text=query_text, return_tensors="pt", padding=True)
    query_image_embedding = clip_model.get_text_features(**text_inputs).detach().numpy()[0]

    image_results = image_collection.query(
        query_embeddings=[query_image_embedding],
        n_results=top_k
    )

    return {
        "text_results": text_results,
        "image_results": image_results
    }

import numpy as np

def hybrid_rank(query_text, text_results, image_results, alpha=0.5):
    # テキストスコア（類似度）
    text_scores = [1 - dist for dist in text_results["distances"][0]]
    text_items = [
        {"type": "text", "content": doc, "score": score}
        for doc, score in zip(text_results["documents"][0], text_scores)
    ]

    # 画像スコア（類似度）
    image_scores = [1 - dist for dist in image_results["distances"][0]]
    image_items = [
        {"type": "image", "content": meta["url"], "score": score}
        for meta, score in zip(image_results["metadatas"][0], image_scores)
    ]

    # ハイブリッドスコア（単純な線形結合）
    all_items = text_items + image_items
    # 必要ならMLLMでリランキング（後述）

    # スコアでソート
    all_items.sort(key=lambda x: x["score"], reverse=True)
    return all_items[:top_k]  # 上位k件

import openai

# OpenAIクライアント（GPT-4V）
client = openai.OpenAI(api_key="your-api-key")

def mllm_rerank(query_text, candidates, top_k=3):
    # 各候補について、MLLMに「この候補は質問に関連しているか？」を聞く
    scores = []
    for cand in candidates:
        if cand["type"] == "text":
            prompt = f"""
質問: {query_text}
候補テキスト: {cand['content']}
この候補テキストは質問に関連していますか？関連している場合は1、関連していない場合は0で答えてください。
"""
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            score = 1 if "1" in response.choices[0].message.content else 0
        else:  # image
            # 画像URLをMLLMに渡して評価（GPT-4Vなど）
            # ここでは簡略化のためスキップ（実装はOpenAIのVision APIを使用）
            score = cand["score"]  # 一旦CLIPスコアを流用

        scores.append(score)

    # スコアでソート
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [cand for cand, _ in ranked[:top_k]]


def generate_answer(query_text, ranked_items):
    # プロンプト構築
    context_texts = []
    image_urls = []

    for item in ranked_items:
        if item["type"] == "text":
            context_texts.append(item["content"])
        else:
            image_urls.append(item["content"])

    context_str = "\n".join(context_texts)

    # MLLMに渡すメッセージ（テキスト＋画像）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"質問: {query_text}\n\n参考文書:\n{context_str}"}
            ] + [
                {"type": "image_url", "image_url": {"url": url}} for url in image_urls
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",  # または同等のMLLM
        messages=messages,
        max_tokens=500
    )

    return response.choices[0].message.content

def multimodal_rag_pipeline(query_text, top_k=5, rerank_top_k=3):
    # 1. 検索
    results = retrieve_multimodal(query_text, top_k=top_k)

    # 2. ハイブリッドランキング
    candidates = hybrid_rank(query_text, results["text_results"], results["image_results"])

    # 3. MLLMリランキング
    ranked = mllm_rerank(query_text, candidates, top_k=rerank_top_k)

    # 4. 生成
    answer = generate_answer(query_text, ranked)

    return answer

