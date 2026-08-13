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

if __name__ == "__main__":
    run_poc_demo()