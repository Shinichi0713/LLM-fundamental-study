

## 概要
3D Tilesは、**大規模な3D地理空間データをインターネット上で効率的に配信・表示するためのオープン標準フォーマット**です。

### 3D Tilesとは

3D Tilesは、Cesium社が開発し、2019年にOGC（Open Geospatial Consortium）のコミュニティ標準として承認された仕様です。<source-chip title="OGC" url="https://www.ogc.org/standards/3dtiles/" /> 建物の3Dモデル、写真測量（フォトグラメトリ）データ、点群（LiDAR）、BIM/CADデータなど、異なる種類の3Dデータを統合して、Webブラウザ上でスムーズに表示できるように設計されています。<source-chip title="Cesium" url="https://cesium.com/why-cesium/3d-tiles/" />

### 3D Tilesの主な特徴

**1. 階層的なタイル構造**

Google Mapsの2Dタイル地図と同じ発想を3Dに拡張したものです。地球全体の3Dデータを「タイル」という小さな単位に分割し、**視点に近い・詳細に見たい部分だけを高解像度で読み込み**、遠くの部分は低解像度で表示します。これにより、巨大な都市全体の3Dモデルでも、必要な部分だけを効率的にストリーミングできます。<source-chip title="Cesium" url="https://cesium.com/why-cesium/3d-tiles/3d-tiles-essentials/" />

**2. 複数のデータ形式を統合**

3D Tilesは1つのコンテナ仕様であり、内部で異なるフォーマットを使い分けます。

| タイル形式 | 用途 |
|-----------|------|
| B3DM | テクスチャ付き3D建物モデル（glTFベース） |
| I3DM | インスタンス化された3Dモデル（街路樹、電柱など同じモデルを大量配置） |
| PNTS | 点群データ（LiDARスキャンなど） |
| CMPT | 上記を複合したデータ |

すべての3Dモデルは、3Dフォーマット標準である **glTF** をベースにしています。<source-chip title="OGC" url="https://docs.ogc.org/cs/22-025r4/22-025r4.html" />

**3. 空間分割の柔軟性**

タイルの分割方式として、以下のような空間インデックス構造をサポートしています。

- **quadtree（四分木）**: 平面方向に4分割
- **octree（八分木）**: 3次元空間を8分割
- **k-d tree**: 空間を再帰的に2分割

これにより、建物が密集する都市部や、地形が複雑な山岳部など、データの特性に応じた最適な分割が可能です。

### 3D Tilesのデータ構造

3D Tilesのデータは主に **JSON（tileset.json）** と **バイナリタイルファイル** で構成されます。

```
tileset.json          ← タイルセットの全体構造（JSON）
├── root/             ← ルートタイル
│   ├── content.b3dm  ← 3Dモデルデータ（バイナリ）
│   ├── children/     ← 子タイル（より詳細なLOD）
│   │   ├── content.b3dm
│   │   └── ...
│   └── ...
└── ...
```

`tileset.json` には、各タイルの**空間的な範囲（bounding volume）**、**詳細度レベル（geometricError）**、**子タイルへの参照**などが記述されています。クライアント（ブラウザ）はこのJSONを読み込み、カメラの視点に応じて必要なタイルだけを選択的にダウンロードします。


### 日本での活用例

日本では、国土交通省の **PLATEAU** プロジェクトが全国の3D都市モデルを3D Tiles形式で公開しています。約60都市の建物モデルが3D Tilesとして配信されており、Webブラウザ上で都市全体の3Dモデルを確認できます。


## LAZファイルからの変換
3D情報を保持するフォーマットとしてオーソドックスなLAZを用いて、3D Tilesのデータが再構成できるでしょうか。
答えは`Yes`です。
LAZはLiDAR点群の圧縮フォーマット（LASzip）であり、3D Tilesの点群タイル形式（PNTSや3D Tiles 1.1のglTFポイント）に変換するツールが複数存在します。

### LAZから3D Tilesへの変換ツール

| ツール名 | 言語 | 対応形式 | 特徴 |
|---------|------|---------|------|
| **Cesium ion** | クラウド | LAZ/LAS → 3D Tiles | Web UIでドラッグ&ドロップ。自動で最適化・配信。1秒あたり500万点を処理。<source-chip title="Cesium" url="https://cesium.com/platform/cesium-ion/3d-tiling-pipeline/point-clouds/" /> |
| **py3dtiles** | Python | LAZ/LAS → 3D Tiles 1.0 (PNTS) | オープンソース。PDALと連携して座標変換・分類属性も保持可能。<source-chip title="3D Geospatial" url="https://www.3d-geospatial.com/lod-management-optimization-strategies/automated-tile-generation/converting-point-clouds-to-3d-tiles-with-py3dtiles/" /> |
| **MIERUNE/point-tiler** | Rust | LAZ/LAS/CSV → 3D Tiles 1.1 | 日本のMIERUNE社製。3D Tiles v1.1対応。東京都の点群データ変換の実績あり。<source-chip title="GitHub" url="https://github.com/MIERUNE/point-tiler" /> |
| **gocesiumtiler** | Go | LAS → 3D Tiles | コマンドライン1行で変換。Windowsでも動作。<source-chip title="GitHub" url="https://github.com/mfbonfigli/gocesiumtiler" /> |
| **cesium_pnt_generator** | Node.js | LAS → PNTS | 初期のプロトタイプ実装。シンプルな変換に適する。<source-chip title="GitHub" url="https://github.com/mattshax/cesium_pnt_generator" /> |

### 変換の流れ

LAZから3D Tilesへの変換は、大きく3つのステップで行われます。

**ステップ1: LAZの展開と前処理**
- LAZは圧縮されたLASなので、多くのツールが内部で自動展開します
- 座標系の確認が重要です。日本の基盤地図情報などは平面直角座標系（EPSG:6677など）の場合があり、3D Tilesは地心座標系（EPSG:4978）が必要なため、変換が必要です
- PDALなどで前処理（不要点の除去、座標変換）を行うことが推奨されます

**ステップ2: 3D Tilesへの変換**
- 点群を空間的に分割し、階層的なタイル構造（quadtree/octree）を構築
- 各タイルをPNTS（3D Tiles 1.0）またはglTFポイント（3D Tiles 1.1）形式でエンコード
- LOD（詳細度）を生成し、遠景は粗い点群、近景は高密度の点群に最適化

**ステップ3: tileset.jsonの生成**
- タイルセット全体の構造を記述したJSONファイルを生成
- 各タイルの空間範囲（bounding volume）と誤差閾値（geometricError）を設定

### 実際の変換例（py3dtiles）

```bash
# PDALで座標変換（平面直角座標系 → 地心座標系）
pdal translate input.laz output.las reprojection \
    --filters.reprojection.in_srs="EPSG:6677" \
    --filters.reprojection.out_srs="EPSG:4978"

# py3dtilesで3D Tilesに変換
py3dtiles convert output.laz --out tileset/
```


### 注意点

**座標系の変換**
日本の基盤地図情報や自治体が公開するLAZデータは、多くの場合平面直角座標系（EPSG:6677など）です。3D Tilesは地球中心座標系（EPSG:4978）で表示されるため、変換時に座標系の指定が必要です。変換を忘れると、点群が地球の裏側や宇宙空間に表示されることがあります。<source-chip title="GIS StackExchange" url="https://gis.stackexchange.com/questions/481090/converting-las-to-pnts-for-cesium" />

**属性の保持**
LiDAR点群には「分類（地面、建物、植被など）」「強度（反射強度）」「RGB（カラー）」などの属性が含まれることがあります。ツールによってはこれらの属性が3D Tilesに引き継がれず、色分け表示ができなくなる場合があります。属性保持が必要な場合は、MIERUNE/point-tilerやカスタム属性対応のツールを選ぶとよいでしょう。<source-chip title="GitHub" url="https://github.com/3dTrees-earth/3dtrees_py3dtiles" />

**3D Tilesのバージョン**
3D Tiles 1.0では点群はPNTS形式、3D Tiles 1.1ではglTFポイントとして扱われます。利用するビューア（CesiumJS、MapLibreなど）の対応バージョンを確認してください。


