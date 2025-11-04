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