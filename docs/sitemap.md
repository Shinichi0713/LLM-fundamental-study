結論から言うと、**その GitHub Actions ワークフローのままでは、提示された「最小・推奨形の sitemap.xml を“厳密に”得ることは難しい」**です。
理由と、**確実に狙いどおりの XML を生成する方法**を整理します。

---

## なぜ今の workflow では難しいのか

### 1. `jayanta525/sitemap-generator-action` は「自動列挙型」

この Action は設計上、

* リポジトリ内の HTML ファイルを **自動走査**
* `<url>` を **機械的に列挙**
* `<lastmod>` などを自動付与

という思想です。

つまり、

```xml
<url>
  <loc>トップページ</loc>
</url>
<url>
  <loc>特定の1ページ</loc>
</url>
```

のような **「人間が意図的に絞った sitemap」** を作る用途には向いていません。

---

### 2. `%URLS%` は完全制御できない

`sitemap_template` で制御できるのは **外枠のみ** であり、

* どの URL を含めるか
* `<lastmod>` を入れない
* 特定ディレクトリだけを対象にする

といった **URL 単位の精密制御は不可** です。

---

## 推奨解決策（確実・安定）

### 方針

**「Action に sitemap を生成させない」**
→ **GitHub Actions で sitemap.xml を直接生成する**

これは Google Search Console 的にも最も安全です。

---

## 推奨ワークフロー（最小・正確）

### ポイント

* sitemap.xml を **完全に固定生成**
* XML スキーマは最小限
* Search Console が最も好む形式

---

### 改善後の workflow 例（推奨）

```yaml
name: Generate and Deploy Sitemap

on:
  push:
    branches:
      - main

jobs:
  generate-sitemap:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      # sitemap.xml を直接生成
      - name: Generate sitemap.xml manually
        run: |
          cat << 'EOF' > sitemap.xml
          <?xml version="1.0" encoding="UTF-8"?>
          <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
              <loc>https://shinichi0713.github.io/LLM-fundamental-study/</loc>
            </url>
            <url>
              <loc>https://shinichi0713.github.io/LLM-fundamental-study/attention/doc/1-query_mha.html</loc>
            </url>
          </urlset>
          EOF

      - name: Commit and push sitemap
        run: |
          git config user.name "github-actions"
          git config user.email "github-actions@github.com"
          git add sitemap.xml
          git commit -m "docs: generate sitemap.xml" || echo "No changes"
          git push
```

---

## この方法のメリット

| 項目                  | 効果                      |
| --------------------- | ------------------------- |
| Search Console 互換性 | 非常に高い                |
| XML の正確性          | 100%                      |
| `/docs/`問題        | 完全回避                  |
| Content-Type          | GitHub Pages が正しく処理 |
| 将来トラブル          | ほぼ無し                  |

---

## もし「自動生成もしたい」場合

現実的な選択肢は2つです。

### 選択肢A（安全）

* 手動 sitemap（上記）
* インデックスしたいページだけ追加

### 選択肢B（中級）

* Action で一旦 sitemap を生成
* `sed` / `grep` で **許可URLだけ抽出**
* 再構築

ただし **Search Console トラブル率が上がる**ため、
現状では A を強く推奨します。

---

## Google Search Console 視点での重要ポイント

Google は以下を最も重視します。

1. URL が実在する
2. noindex されていない
3. sitemap が単純である
4. 余計な schema や拡張がない

あなたが示した XML は **理想的** です。

---

## まとめ

* 現在の Action は用途不一致
* sitemap は「自動生成しない方が良いケース」
* GitHub Actions で **直接 XML を生成**するのが最適解
* 提示した workflow はそのまま運用可能

必要であれば、

* Jekyll / docs 構成前提での sitemap 戦略
* 「記事が増えたら自動追記する方式」
* Search Console 側の登録手順の最終確認

まで整理します。
