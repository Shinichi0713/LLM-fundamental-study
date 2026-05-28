import requests
import json

def ask_ollama(prompt: str, model: str = "llama3.1:8b"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()["response"]

answer = ask_ollama("こんにちは、自己紹介してください。")
print(answer)


import requests
import json

def ask_ollama(prompt: str, model: str = "llama3.1:8b"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False  # ストリーミングしない（1回でレスポンス取得）
    }
    response = requests.post(url, json=data)
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    return response.json()["response"]

# 実行例
answer = ask_ollama("こんにちは、自己紹介してください。")
print(answer)


def ask_ollama_stream(prompt: str, model: str = "llama3.1:8b"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    response = requests.post(url, json=data, stream=True)
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    full_response = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            if "response" in chunk:
                print(chunk["response"], end="", flush=True)
                full_response += chunk["response"]
    return full_response

# 実行例（逐次表示）
answer = ask_ollama_stream("こんにちは、自己紹介してください。")


def chat_with_ollama(messages: list, model: str = "llama3.1:8b"):
    url = "http://localhost:11434/api/chat"
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    response = requests.post(url, json=data)
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    return response.json()["message"]["content"]

# 実行例
messages = [
    {"role": "system", "content": "あなたは役立つアシスタントです。"},
    {"role": "user", "content": "こんにちは、自己紹介してください。"}
]
answer = chat_with_ollama(messages)
print(answer)


def graph_rag_with_ollama(query: str, context_texts: list, model: str = "llama3.1:8b"):
    # コンテキストを結合
    context_str = "\n\n".join(context_texts)
    prompt = f"""
以下の文書を参考に、質問に答えてください。

参考文書:
{context_str}

質問: {query}

回答:
"""
    return ask_ollama_lib(prompt, model=model)

# 実行例
contexts = [
    "Apple Inc. is headquartered in Cupertino, California.",
    "Cupertino is a city in California."
]
answer = graph_rag_with_ollama("Where is Apple Inc. headquartered?", contexts)
print(answer)