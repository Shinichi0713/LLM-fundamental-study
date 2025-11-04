では、**Pythonで形態素解析（MeCab）を実行する最小例**をお見せします。

日本語文を形態素解析して、**単語 + 品詞**を表示します。

---

## ✅ Python + MeCab の形態素解析例

### **インストール**

macOS / Linux

```bash
pip install mecab-python3
sudo apt install mecab libmecab-dev mecab-ipadic-utf8  # (Ubuntuの場合)
```

Windows の場合は

```bash
pip install mecab-python3
```

> ※ Windowsでエラーの場合は MeCab 本体を入れる必要があります。必要なら手順も案内します！

---

### **Pythonコード例**

```python
import MeCab

# MeCabの準備
m = MeCab.Tagger("-Ochasen")

text = "寿司が好きです。昨日は友達と食べに行きました。"

# 解析
result = m.parse(text)
print(result)
```

---

### ✅ 実行結果例（イメージ）

```
寿司    名詞
が      助詞
好き    名詞
です    助動詞
。      記号
昨日    名詞
は      助詞
友達    名詞
と      助詞
食べ    動詞
に      助詞
行き    動詞
まし    助動詞
た      助動詞
。      記号
```

---

## ✅ 何が起きている？

* 文章 → 単語に分割
* 品詞（名詞、動詞、助詞…）を自動でラベル付け

**コンピュータに意味の手がかりを与える工程**です。

---

## ✨ 追加：名詞だけ抜き出す例

```python
import MeCab

m = MeCab.Tagger()
text = "私は機械学習と自然言語処理が好きです。"

node = m.parseToNode(text)
nouns = []

while node:
    if node.feature.startswith("名詞"):
        nouns.append(node.surface)
    node = node.next

print(nouns)
```

**出力例**

```
['私', '機械', '学習', '自然', '言語', '処理']
```

---

## 📌 次にやってみたい例（作れます）

✅ WordCloud（形態素解析した単語から雲を作る）

✅ TF-IDF + 形態素解析で文章検索

✅ TransformerのTokenizerと比較学習

✅ spaCyによる日本語解析例

---

もし環境がまだ無ければ、

**Google Colab で動かせるノートブック**も作ります！

どれにしますか？👇

1. Colabノートが欲しい
2. Windows用インストール手順
3. WordCloudを作りたい
4. spaCy版も見たい

番号で教えてください 😊
