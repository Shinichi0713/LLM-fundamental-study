



良い質問です。結論から言うと、**先ほどのコードではスキーマ（エンティティ種別や関係種別）は明示的に定義していません**。`SimpleLLMPathExtractor` を使っているため、LLMが自由な形式でトリプルを生成する「スキーマレス（自由抽出）」な構成になっています。

## 1. 先ほどのコードでの実際の挙動

```python
kg_extractor = SimpleLLMPathExtractor(
    llm=Settings.llm,
    max_paths_per_chunk=10,
    num_workers=4,
)
```

`SimpleLLMPathExtractor` はデフォルトのプロンプトテンプレートで各チャンクをLLMに渡し、

```
(entity1, relation, entity2)
```

という形式のトリプルを**LLMの判断に任せて自由に**生成させます。そのため、

- エンティティのラベル（`Person`, `Organization` など）
- 関係の種類（`WORKS_FOR`, `LOCATED_IN` など）

はすべて **LLMが文脈から適当に命名**します。実行するたびに表記ゆれ（例: `founded` と `Founded` と `is_founder_of` が混在する）が起きやすいのが弱点です。

Neo4j側では、各トリプルは概ね以下のような形で格納されます。

```
(:Entity {name: "..."}) -[:RELATION_NAME {...}]-> (:Entity {name: "..."})
```

`Entity` という汎用ラベルにプロパティとして名前が入り、リレーション名がLLMの出力そのまま使われる、というイメージです。

## 2. スキーマを固定したい場合: `SchemaLLMPathExtractor`

エンティティ種別・関係種別・その組み合わせを事前に定義し、LLMにその枠内で抽出させたい場合は `SchemaLLMPathExtractor` を使います。この図のように、`SchemaLLMPathExtractor` を使うとエンティティ種別・関係種別・組み合わせパターンを事前に定義でき、LLMはその枠内でのみ抽出を行います。実装例は以下の通りです。

```python
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from typing import Literal

# エンティティの種別を定義
entities = Literal["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT"]

# 関係の種別を定義
relations = Literal["WORKS_FOR", "FOUNDED", "LOCATED_IN", "PARTICIPATED_IN", "PRODUCES"]

# (エンティティ, 関係, エンティティ) の許可される組み合わせ(トリプル)を定義
validation_schema = {
    "PERSON": ["WORKS_FOR", "FOUNDED", "PARTICIPATED_IN"],
    "ORGANIZATION": ["LOCATED_IN", "PRODUCES"],
    "EVENT": ["LOCATED_IN"],
}

kg_extractor = SchemaLLMPathExtractor(
    llm=Settings.llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True,  # True: スキーマ外の抽出は除外 / False: 参考程度に留める
)
```

この `kg_extractor` を先ほどの `PropertyGraphIndex.from_documents(..., kg_extractors=[kg_extractor], ...)` の部分に差し替えるだけで、Neo4j側には一貫したラベル・関係名でノードが登録されるようになります。

## 3. Neo4j自体のスキーマ設定について

もう一点補足すると、**Neo4jはスキーマレスDB**なので、そもそもテーブル定義のようなものはありません。ただし運用上は以下のような設定を別途行うと良いです。

- **制約・インデックス**: 重複ノード防止や検索高速化のために `CREATE CONSTRAINT` や `CREATE INDEX` をCypherで発行しておく（LlamaIndexは自動では作りません）
- **ベクトルインデックス**: `embed_kg_nodes=True` にした場合、`Neo4jPropertyGraphStore` はデフォルトでベクトル検索用のインデックスを内部的に作成します

```cypher
CREATE CONSTRAINT entity_name IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;
```

## まとめ

| 抽出器 | スキーマ | 特徴 |
|---|---|---|
| `SimpleLLMPathExtractor`（元コード） | なし（自由生成） | 手軽だが表記ゆれが出やすい |
| `SchemaLLMPathExtractor` | あり（entities/relations/triple定義） | 一貫性が高く、後続のクエリ・可視化がしやすい |

実運用（RAGやグラフ検索の精度を求める場合）では `SchemaLLMPathExtractor` の使用を推奨します。ドメインに合わせたエンティティ・関係の定義を一緒に検討しましょうか？