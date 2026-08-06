LlamaIndexを通じてGraphDB（Neo4jなど）にインデックスを作成・登録した際、ノード（Node）には単なる「名前」だけでなく、検索やLLMのコンテキストとして活用するためのマルチレイヤーな情報が保持されます。

LlamaIndexの標準的なプロパティグラフ（`PropertyGraphIndex`）構成において、ノード内に保持される主な情報は以下の**4つの要素**で構成されています。

---

## ノード内に保持される主な情報構造

```
[ ノード (Entity Node) ]
  ├── 1. アイデンティティ (Name / Label)
  ├── 2. 属性プロパティ (Properties)
  ├── 3. テキストソース・メタデータ (Source Chunk Metadata)
  └── 4. ベクトル埋め込み (Embedding Vector)

```

---

### 1. エンティティ識別情報（Name / Label）

ノードが指し示す概念の基本ラベルです。

* **`name` / `id**`: エンティティの固有名（例: `"RegionCLIP"`, `"Image Encoder"`, `"PyTorch"`）
* **`type` / `label**`: エンティティのカテゴリ（例: `"Model"`, `"Component"`, `"Framework"`）

### 2. 属性プロパティ（Properties）

LLMによって抽出された、そのエンティティ自体の詳細な説明や定義情報です。

* **`description`**: エンティティの定義や概要文（例: `"画像全体の文字表現を領域レベルに拡張したVision-Languageモデル"`）
* **カスタム属性**: 登録時・更新時に付与した任意のキー・バリュー情報（作成日時、分類タグなど）

### 3. テキストソース・メタデータ（Source Chunk / Provenance）

グラフ構築の元となったドキュメント（テキストチャンク）との関連付け情報です。これにより「なぜそのノードが存在するのか」の根拠を追跡できます。

* **`source_text`**: このノードやリレーションが抽出された元々のテキスト段落（元の文章そのもの）
* **`doc_id` / `chunk_id**`: 参照元ドキュメントのID
* **`file_name`**: ドキュメントのファイル名やURL

### 4. ベクトル埋め込み（Embedding Vector）

ベクトル検索（Vector Search）とグラフ検索を組み合わせる（ハイブリッド検索）ために保存される数値列です。

* **`embedding`**: エンティティ名や `description` から生成された高次元ベクトル（例: 1536次元の数値配列）。「似た概念のノード」を意味的に検索する際に用いられます。

---

## 実際にNeo4j等に保持されている内部データの具体例

例えば、「RegionCLIPはRoIAlignを使用して領域特徴量を抽出する」という文章をインデックス化した際、**`RegionCLIP` ノード**内部には以下のようなデータが保存されます。

```json
{
  "id": "RegionCLIP",
  "labels": ["Entity", "Model"],
  "properties": {
    "name": "RegionCLIP",
    "description": "CVPR 2022で提案された、CLIPを領域レベルの視覚表現へ拡張したモデル。",
    "triplets": [
      "RegionCLIP -> uses -> RoIAlign",
      "RegionCLIP -> proposed_in -> CVPR 2022"
    ],
    "source_id": "chunk_98234",
    "file_name": "regionclip_paper.txt"
  },
  "embedding": [0.012, -0.045, 0.089, "...(1536次元の浮動小数点)"]
}

```

---

## 検索時（Retriever）にノード情報がどう活用されるか

1. **キーワード / グラフ辿り (Traversal):**
`RegionCLIP` というノード（`name`）からスタートし、繋がっている `RoIAlign` ノードやリレーション（`uses`）の構造を直接辿ります。
2. **ベクトル検索 (Vector Search):**
`embedding` プロパティを使って、表記揺れや曖昧な質問（「領域レベルのCLIP」など）に対しても最も近いノードを高速ヒットさせます。
3. **LLMのコンテキスト化:**
最終的に `description` や `source_text` の情報がLLMのプロンプト（コンテキスト）として集約され、正確な回答が生成されます。


