import math
from collections import Counter
import numpy as np

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        corpus: 分かち書き（トークン化）された文書のリスト
                例: [['python', 'graph', 'library'], ['machine', 'learning', 'python']]
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        
        # 各文書の単語数と、全文書の平均単語数(avgdl)の計算
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size
        
        # 各文書内での単語の出現頻度（Term Frequency）を記録
        self.doc_freqs = [Counter(doc) for doc in corpus]
        
        # IDF（逆文書頻度）の計算
        self.idf = {}
        self._calc_idf(corpus)

    def _calc_idf(self, corpus):
        """
        各単語が「何個の文書に登場したか(DF)」を数え、IDFを計算する
        """
        # 単語が登場する文書数をカウント
        doc_counts = Counter()
        for doc in corpus:
            unique_words = set(doc)
            for word in unique_words:
                doc_counts[word] += 1
                
        # ルーカスのIDF数式（Okapi BM25の標準公式）で計算
        for word, df in doc_counts.items():
            # 負のIDFを防止するため、MAXをとるか、+0.5などの平滑化を行う
            num = self.corpus_size - df + 0.5
            denom = df + 0.5
            self.idf[word] = math.log(1 + (num / denom))

    def score(self, query, doc_index):
        """
        特定の文書(doc_index)に対する、クエリのBM25スコアを計算する
        """
        score = 0.0
        doc_freq = self.doc_freqs[doc_index]
        doc_len = self.doc_lengths[doc_index]
        
        for word in query:
            if word not in self.idf:
                continue
                
            # 単語の文書内出現頻度 (f)
            f = doc_freq[word]
            
            # BM25のコアとなる数式の分母・分子の計算
            # 分子: f * (k1 + 1)
            # 分母: f + k1 * (1 - b + b * (doc_len / avgdl))
            num = f * (self.k1 + 1)
            denom = f + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            
            # IDFスコアを掛け合わせて加算
            score += self.idf[word] * (num / denom)
            
        return score

    def search(self, query, top_n=3):
        """
        クエリに対して全文書のスコアを計算し、ランキングを返す
        """
        scores = []
        for i in range(self.corpus_size):
            score = self.score(query, i)
            scores.append((i, score))
            
        # スコアの高い順にソート
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

# --- 動作検証 ---
if __name__ == "__main__":
    # 1. サンプルコーパス（あらかじめ単語に分割したもの）
    # ※ 本来はMeCabやJanome、GiNZA等で形態素解析した結果を入れます
    raw_corpus = [
        "Python implementation of GraphRAG using NetworkX and LLM",
        "Introduction to machine learning algorithms and neural networks",
        "Graph theory and network analysis with Python libraries like NetworkX",
        "How to build a search engine with BM25 algorithm from scratch in Python"
    ]
    
    # 簡易トークナイズ（小文字化してスペース分割）
    corpus = [doc.lower().split() for doc in raw_corpus]
    
    # 2. BM25モデルの初期化
    bm25 = BM25(corpus, k1=1.5, b=0.75)
    
    # 3. クエリを入力して検索
    query_text = "Python NetworkX graph"
    query = query_text.lower().split()
    
    results = bm25.search(query, top_n=4)
    
    print(f"検索クエリ: '{query_text}'\n")
    print("--- 検索結果（BM25スコア順） ---")
    for rank, (doc_idx, score) in enumerate(results, 1):
        print(f"{rank}位 (Score: {score:.4f}): {raw_corpus[doc_idx]}")