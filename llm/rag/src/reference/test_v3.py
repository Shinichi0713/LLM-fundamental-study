import spacy
import networkx as nx

# spaCyモデルのロード
nlp = spacy.load("en_core_web_sm")

# グラフの初期化
G = nx.Graph()

def add_document_to_graph(doc_id: str, text: str):
    """
    ドキュメントを解析し、エンティティをノード、共起関係をエッジとしてグラフに追加。
    """
    doc = nlp(text)
    entities = []

    # エンティティ抽出（例：人名・組織名・地名）
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:  # 人名・組織・地名
            entity_text = ent.text.strip()
            if entity_text:
                entities.append(entity_text)
                # ノード追加（なければ作成）
                if not G.has_node(entity_text):
                    G.add_node(entity_text, type=ent.label_)

    # ドキュメントノード追加
    G.add_node(doc_id, type="DOCUMENT")
    # ドキュメントとエンティティのエッジ
    for ent in entities:
        G.add_edge(doc_id, ent, relation="MENTIONS")

    # エンティティ間の共起エッジ（同じ文書に現れたら重みを増やす）
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            if G.has_edge(entities[i], entities[j]):
                G[entities[i]][entities[j]]["weight"] += 1
            else:
                G.add_edge(entities[i], entities[j], weight=1, relation="CO_OCCURRENCE")



# 例：Wikipedia風のテキスト
documents = {
    "doc1": "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Steve Jobs co-founded Apple.",
    "doc2": "Steve Jobs was the co-founder of Apple Inc. He was born in San Francisco.",
    "doc3": "Cupertino is a city in California. It is known for being the headquarters of Apple Inc."
}

for doc_id, text in documents.items():
    add_document_to_graph(doc_id, text)

print("Graph nodes:", list(G.nodes())[:5])
print("Graph edges:", list(G.edges())[:5])

import ollama

def graph_rag_search(query: str, top_k_docs: int = 3):
    """
    1. クエリからエンティティを抽出
    2. グラフ上で関連ドキュメントを探索
    3. 関連ドキュメントのテキストを取得
    4. LLMに渡して回答生成
    """
    # 1. クエリのエンティティ抽出
    query_doc = nlp(query)
    query_entities = [ent.text for ent in query_doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]

    # 2. グラフ探索：クエリエンティティに隣接するドキュメントノードを取得
    candidate_docs = set()
    for ent in query_entities:
        if G.has_node(ent):
            neighbors = list(G.neighbors(ent))
            for n in neighbors:
                if G.nodes[n].get("type") == "DOCUMENT":
                    candidate_docs.add(n)

    # 3. 関連ドキュメントのテキストを取得（ここでは簡易に辞書から）
    context_texts = []
    for doc_id in list(candidate_docs)[:top_k_docs]:
        if doc_id in documents:
            context_texts.append(documents[doc_id])

    if not context_texts:
        return "No relevant documents found."

    # 4. LLM（Ollama）で回答生成
    context_str = "\n\n".join(context_texts)
    prompt = f"""
以下の文書を参考に、質問に答えてください。

参考文書:
{context_str}

質問: {query}

回答:
"""
    response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


query = "Where is Apple Inc. headquartered?"
answer = graph_rag_search(query)
print("質問:", query)
print("回答:", answer)


import spacy
import networkx as nx

# spaCyモデルのロード
nlp = spacy.load("en_core_web_sm")

# グラフの初期化
G = nx.Graph()

def add_document_to_graph(doc_id: str, text: str):
    """
    ドキュメントを解析し、エンティティをノード、共起関係をエッジとしてグラフに追加。
    """
    doc = nlp(text)
    entities = []

    # エンティティ抽出（例：人名・組織名・地名）
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:  # 人名・組織・地名
            entity_text = ent.text.strip()
            if entity_text:
                entities.append(entity_text)
                # ノード追加（なければ作成）
                if not G.has_node(entity_text):
                    G.add_node(entity_text, type=ent.label_)

    # ドキュメントノード追加
    G.add_node(doc_id, type="DOCUMENT")
    # ドキュメントとエンティティのエッジ
    for ent in entities:
        G.add_edge(doc_id, ent, relation="MENTIONS")

    # エンティティ間の共起エッジ（同じ文書に現れたら重みを増やす）
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            if G.has_edge(entities[i], entities[j]):
                G[entities[i]][entities[j]]["weight"] += 1
            else:
                G.add_edge(entities[i], entities[j], weight=1, relation="CO_OCCURRENCE")

# 例：Wikipedia風のテキスト
documents = {
    "doc1": "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Steve Jobs co-founded Apple.",
    "doc2": "Steve Jobs was the co-founder of Apple Inc. He was born in San Francisco.",
    "doc3": "Cupertino is a city in California. It is known for being the headquarters of Apple Inc."
}

for doc_id, text in documents.items():
    add_document_to_graph(doc_id, text)

print("Graph nodes:", list(G.nodes())[:5])
print("Graph edges:", list(G.edges())[:5])

import ollama

def graph_rag_search(query: str, top_k_docs: int = 3):
    """
    1. クエリからエンティティを抽出
    2. グラフ上で関連ドキュメントを探索
    3. 関連ドキュメントのテキストを取得
    4. Ollama（LLM）に渡して回答生成
    """
    # 1. クエリのエンティティ抽出
    query_doc = nlp(query)
    query_entities = [ent.text for ent in query_doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]

    # 2. グラフ探索：クエリエンティティに隣接するドキュメントノードを取得
    candidate_docs = set()
    for ent in query_entities:
        if G.has_node(ent):
            neighbors = list(G.neighbors(ent))
            for n in neighbors:
                if G.nodes[n].get("type") == "DOCUMENT":
                    candidate_docs.add(n)

    # 3. 関連ドキュメントのテキストを取得
    context_texts = []
    for doc_id in list(candidate_docs)[:top_k_docs]:
        if doc_id in documents:
            context_texts.append(documents[doc_id])

    if not context_texts:
        return "No relevant documents found."

    # 4. Ollamaで回答生成
    context_str = "\n\n".join(context_texts)
    prompt = f"""
以下の文書を参考に、質問に答えてください。

参考文書:
{context_str}

質問: {query}

回答:
"""
    response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

from google.colab import drive
drive.mount('/content/drive')

# グラフを保存（例：pickle）
import pickle
with open("/content/drive/MyDrive/graph.pkl", "wb") as f:
    pickle.dump(G, f)