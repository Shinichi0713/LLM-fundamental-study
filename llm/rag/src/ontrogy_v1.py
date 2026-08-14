from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple


# --- 1. データ構造定義 ---

@dataclass
class OntologyConcept:
    """オントロジーにおける概念（エンティティタイプや属性）"""
    id: str
    name: str


@dataclass
class HyperEdge:
    """
    OG-RAGの核心: 複数の概念・エンティティ・文書チャンクを単一の「意味的グループ」として結合するハイパーエッジ
    """
    id: str
    concepts: Set[str]       # 含まれるオントロジー概念 (概念ID)
    doc_chunk: str           # 紐づくドキュメントテキスト（コンテキスト）


# --- 2. OG-RAG メインエンジン ---

class OGRAGEngine:
    def __init__(self):
        self.concepts: Dict[str, OntologyConcept] = {}
        self.hyperedges: List[HyperEdge] = []

    def add_concept(self, concept_id: str, name: str):
        """オントロジーの概念を登録"""
        self.concepts[concept_id] = OntologyConcept(id=concept_id, name=name)

    def add_hyperedge(self, edge_id: str, concepts: List[str], doc_chunk: str):
        """オントロジー概念に接地（Grounding）されたハイパーエッジを追加"""
        self.hyperedges.append(
            HyperEdge(id=edge_id, concepts=set(concepts), doc_chunk=doc_chunk)
        )

    def _extract_query_concepts(self, query: str) -> Set[str]:
        """
        [簡易表現] クエリから関連するオントロジー概念を抽出
        ※ 実際の論文実装ではLLMや固有表現抽出(NER)でコンセプトにマッピングします
        """
        matched_concepts = set()
        for c_id, concept in self.concepts.items():
            if concept.name.lower() in query.lower():
                matched_concepts.add(c_id)
        return matched_concepts

    def retrieve_minimal_hyperedges(self, query: str) -> List[HyperEdge]:
        """
        クエリに対して十分な情報をカバーする「最小のハイパーエッジ集合」を検索（集合被覆問題の近似）
        """
        # Step 1: クエリをオントロジー概念にアンカー（グラウンディング）
        target_concepts = self._extract_query_concepts(query)
        if not target_concepts:
            return []

        # Step 2: ターゲット概念をカバレッジするハイパーエッジを抽出（Greedy Set Cover アルゴリズム）
        uncovered = set(target_concepts)
        selected_edges: List[HyperEdge] = []
        candidate_edges = list(self.hyperedges)

        while uncovered and candidate_edges:
            # まだカバーされていないターゲット概念を最も多く含むハイパーエッジを優先選択
            best_edge = max(
                candidate_edges,
                key=lambda edge: len(edge.concepts & uncovered)
            )

            intersection = best_edge.concepts & uncovered
            if not intersection:
                # これ以上カバーできるエッジがない場合は終了
                break

            selected_edges.append(best_edge)
            uncovered -= intersection
            candidate_edges.remove(best_edge)

        return selected_edges


# --- 3. 動作確認シナリオ ---

def run_poc_demo():
    engine = OGRAGEngine()

    # --- Step A: ドメインオントロジーの定義（例: 医療・製薬ドメイン） ---
    engine.add_concept("C_DISEASE_DIABETES", "糖尿病")
    engine.add_concept("C_DRUG_METFORMIN", "メトホルミン")
    engine.add_concept("C_SIDE_EFFECT_NAUSEA", "吐き気")
    engine.add_concept("C_DOSAGE", "用法用量")

    # --- Step B: ハイパーグラフ構築（文書チャンクをオントロジー概念でハイパーエッジ化） ---
    engine.add_hyperedge(
        edge_id="HE_1",
        concepts=["C_DISEASE_DIABETES", "C_DRUG_METFORMIN"],
        doc_chunk="[文書A] メトホルミンは2型糖尿病の第一選択薬として広く処方されています。"
    )
    engine.add_hyperedge(
        edge_id="HE_2",
        concepts=["C_DRUG_METFORMIN", "C_SIDE_EFFECT_NAUSEA"],
        doc_chunk="[文書B] メトホルミンの主な副作用には、初期投与時の吐き気や腹部不快感があります。"
    )
    engine.add_hyperedge(
        edge_id="HE_3",
        concepts=["C_DRUG_METFORMIN", "C_DOSAGE"],
        doc_chunk="[文書C] 成人の通常用量は1日500mgから開始し、維持量は1日750mg〜1500mgです。"
    )

    # --- Step C: クエリ実行 ---
    query = "糖尿病の薬であるメトホルミンの副作用として吐き気はありますか？"
    print(f"■ Query: {query}\n")

    # 抽出実行
    retrieved_edges = engine.retrieve_minimal_hyperedges(query)

    print("■ 検索結果（取得された最小ハイパーエッジ集合）:")
    for edge in retrieved_edges:
        print(f"- ID: {edge.id}")
        print(f"  関連概念: {[engine.concepts[c].name for c in edge.concepts]}")
        print(f"  テキスト: {edge.doc_chunk}\n")

from typing import Literal

from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

documents = [
    Document(text="山田部長はネコプロジェクトの責任者である。"),
    Document(text="ネコプロジェクトはCRM基盤に影響する。"),
    Document(text="山田部長は営業企画部に所属している。"),
    Document(text="CRM基盤は顧客情報を管理する社内システムである。"),
]

# エンティティ種別
entities = Literal[
    "PERSON",
    "PROJECT",
    "SYSTEM",
    "DEPARTMENT"
]

# 関係種別
relations = Literal[
    "RESPONSIBLE_FOR",
    "AFFECTS",
    "BELONGS_TO"
]

# 簡易オントロジー / スキーマ
# 「どの種類のエンティティが、どの関係を持てるか」を定義
kg_validation_schema = {
    "PERSON": ["RESPONSIBLE_FOR", "BELONGS_TO"],
    "PROJECT": ["AFFECTS"],
    "SYSTEM": [],
    "DEPARTMENT": [],
}

kg_extractor = SchemaLLMPathExtractor(
    llm=OpenAI(model="gpt-4o-mini", temperature=0),
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=kg_validation_schema,
    strict=True,
)

index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[kg_extractor],
    llm=OpenAI(model="gpt-4o-mini", temperature=0),
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    show_progress=True,
)

query_engine = index.as_query_engine(
    include_text=True,
)

response = query_engine.query(
    "山田部長が責任を持つプロジェクトは、どのシステムに影響しますか？"
)

print(response)

"""
はい、**PythonのLlamaIndexでオントロジーを使ったナレッジグラフ構築は可能**です。

ただし、正確には、

> **LlamaIndexで「オントロジーを自動で完成させる」というより、  
> こちらで定義したオントロジー／スキーマに従って、文書からエンティティと関係を抽出し、Property Graphを作る**

という使い方になります。

---

## LlamaIndexでできること

LlamaIndexには **PropertyGraphIndex** という仕組みがあります。これは、文書からエンティティや関係を抽出して、ノードとリレーションを持つプロパティグラフを構築・検索するための機能です。<source-chip title="LlamaIndex Python Documentation" url="https://llamaindex.openml.io/python/framework/module_guides/indexing/lpg_index_guide/" />

特に、**SchemaLLMPathExtractor** を使うと、

- 許可するエンティティ種別
- 許可する関係種別
- どのエンティティからどの関係を張ってよいか

を指定できます。つまり、簡易的なオントロジーを定義して、その制約に沿ってグラフを作れます。<source-chip title="LlamaIndex Python Documentation" url="https://llamaindex.openml.io/python/framework/module_guides/indexing/lpg_index_guide/" />

---

## イメージ

例えば、以下のようなオントロジーを定義できます。

```text
エンティティ:
- PERSON
- PROJECT
- SYSTEM
- DEPARTMENT

関係:
- RESPONSIBLE_FOR
- AFFECTS
- BELONGS_TO

制約:
- PERSON は PROJECT に RESPONSIBLE_FOR できる
- PROJECT は SYSTEM に AFFECTS できる
- PERSON は DEPARTMENT に BELONGS_TO できる
```

これにより、文書から、

```text
山田部長 --RESPONSIBLE_FOR--> ネコプロジェクト
ネコプロジェクト --AFFECTS--> CRM基盤
山田部長 --BELONGS_TO--> 営業企画部
```

のようなグラフを作ることができます。

---

## 最小コード例

```python
from typing import Literal

from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

documents = [
    Document(text="山田部長はネコプロジェクトの責任者である。"),
    Document(text="ネコプロジェクトはCRM基盤に影響する。"),
    Document(text="山田部長は営業企画部に所属している。"),
    Document(text="CRM基盤は顧客情報を管理する社内システムである。"),
]

# エンティティ種別
entities = Literal[
    "PERSON",
    "PROJECT",
    "SYSTEM",
    "DEPARTMENT"
]

# 関係種別
relations = Literal[
    "RESPONSIBLE_FOR",
    "AFFECTS",
    "BELONGS_TO"
]

# 簡易オントロジー / スキーマ
# 「どの種類のエンティティが、どの関係を持てるか」を定義
kg_validation_schema = {
    "PERSON": ["RESPONSIBLE_FOR", "BELONGS_TO"],
    "PROJECT": ["AFFECTS"],
    "SYSTEM": [],
    "DEPARTMENT": [],
}

kg_extractor = SchemaLLMPathExtractor(
    llm=OpenAI(model="gpt-4o-mini", temperature=0),
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=kg_validation_schema,
    strict=True,
)

index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[kg_extractor],
    llm=OpenAI(model="gpt-4o-mini", temperature=0),
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    show_progress=True,
)

query_engine = index.as_query_engine(
    include_text=True,
)

response = query_engine.query(
    "山田部長が責任を持つプロジェクトは、どのシステムに影響しますか？"
)

print(response)
```

---

## 何が嬉しいか

この方法を使うと、LLMに自由に関係を抽出させるのではなく、

```text
PERSON
PROJECT
SYSTEM
DEPARTMENT
```

のような決められた型に沿って抽出できます。

つまり、

```text
山田部長 = PERSON
ネコプロジェクト = PROJECT
CRM基盤 = SYSTEM
営業企画部 = DEPARTMENT
```

と整理したうえで、

```text
山田部長 --RESPONSIBLE_FOR--> ネコプロジェクト
ネコプロジェクト --AFFECTS--> CRM基盤
```

という関係を作れます。

これにより、通常RAGよりも、

```text
山田部長
→ ネコプロジェクト
→ CRM基盤
```

という推論経路を作りやすくなります。

---

## 注意点

LlamaIndexで「完全自動で良いオントロジーを作る」ことは、まだ簡単ではありません。

実務では、次の流れが現実的です。

```text
1. 人間がざっくりオントロジーを定義する
   例: Person, Project, System, Department

2. LlamaIndexで文書から関係を抽出する

3. 抽出結果を確認する

4. 不足しているエンティティ種別・関係種別を追加する

5. 名寄せ・表記ゆれ統合を行う
```

つまり、LlamaIndexは **オントロジー構築そのものの自動化ツール**というより、  
**定義したオントロジーに従ってナレッジグラフを作り、GraphRAGに使うための実装基盤**と考えると分かりやすいです。

---

## 結論

はい、LlamaIndexで可能です。

特に使うべき機能は、

```text
PropertyGraphIndex
SchemaLLMPathExtractor
```

です。

おすすめの使い方は、

> **人間がオントロジーを定義し、LlamaIndexでその制約に従って文書からエンティティ・関係を抽出する**

という形です。  
社内用語や固有名詞を扱うGraphRAGの試作には、かなり相性が良いです。
"""