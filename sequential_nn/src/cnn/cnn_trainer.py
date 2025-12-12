import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

# サンプルデータ（非常に簡略化）
texts = [
    "This movie was fantastic and I loved it",
    "I really like this film it was great",
    "Absolutely wonderful story and characters",
    "Terrible movie I hated everything",
    "The film was boring and disappointing",
    "Bad acting and horrible plot",
]

labels = [1, 1, 1, 0, 0, 0]  # 1: positive, 0: negative

# 前処理
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text.split()

tokenized = [tokenize(t) for t in texts]

# 語彙作成
vocab = {}
for sent in tokenized:
    for w in sent:
        if w not in vocab:
            vocab[w] = len(vocab) + 1  # 0 はパディング用

# ID化
def encode(tokens):
    return [vocab[w] for w in tokens]

encoded = [encode(s) for s in tokenized]

# 長さを揃える
max_len = max(len(s) for s in encoded)
padded = [s + [0]*(max_len - len(s)) for s in encoded]

# Tensor 化
X = torch.tensor(padded)
y = torch.tensor(labels)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, embed_dim)) for k in [3,4,5]
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(100 * 3, num_classes)

    def forward(self, x):
        x = self.embedding(x)             # (B, L, D)
        x = x.unsqueeze(1)                # (B, 1, L, D)

        conv_out = []
        for conv in self.convs:
            h = torch.relu(conv(x)).squeeze(3)        # (B, 100, L-k+1)
            h = torch.max_pool1d(h, h.size(2)).squeeze(2)   # (B, 100)
            conv_out.append(h)

        out = torch.cat(conv_out, 1)     # (B, 300)
        out = self.dropout(out)
        return self.fc(out)
    

model = TextCNN(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    pred = out.argmax(1)
    acc = accuracy_score(y, pred)
    print(f"epoch {epoch}, loss={loss.item():.4f}, acc={acc:.3f}")


# すでに学習済みの model, vocab, max_len を利用する前提
# model.eval() にして推論モードへ
model.eval()


# -----------------------------
# 前処理: tokenize
# -----------------------------
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text.split()


# -----------------------------
# 前処理: encode
# -----------------------------
def encode(tokens, vocab):
    ids = []
    for w in tokens:
        if w in vocab:
            ids.append(vocab[w])
        else:
            ids.append(0)  # 未知語は 0（PAD扱い）
    return ids


# -----------------------------
# 推論関数
# -----------------------------
def predict_sentiment(sentence, model, vocab, max_len):

    tokens = tokenize(sentence)
    encoded = encode(tokens, vocab)

    # 長さを揃える
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    X = torch.tensor([encoded])  # バッチ化（1文 → shape=[1, max_len]）

    with torch.no_grad():
        out = model(X)
        pred = torch.argmax(out, dim=1).item()

    return "Positive" if pred == 1 else "Negative"


# -----------------------------
# 例：実際に文章を分類
# -----------------------------
test_sentences = [
    "I thought the movie was amazing and full of surprises",
    "The plot was extremely boring and I wanted to stop watching",
    "It was not bad but not great either",
    "I really hated the ending of the film",
    "The acting and story were both wonderful",
]

for s in test_sentences:
    label = predict_sentiment(s, model, vocab, max_len)
    print(f"{s}  -->  {label}")