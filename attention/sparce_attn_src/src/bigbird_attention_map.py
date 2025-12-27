import torch
from transformers import AutoTokenizer, BigBirdModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. モデルのロード (BigBird-base)
model_name = "google/bigbird-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# output_attentions=True が重要です
model = BigBirdModel.from_pretrained(model_name, output_attentions=True)

# 2. テキストの準備
# BigBirdは長文向けですが、まずは構造が分かりやすい短文で試します
text = "Despite the terrible weather and the long delay, the concert was surprisingly wonderful."

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 3. アテンションデータの取得
# 層(Layer)とヘッド(Head)を指定（例：Layer 0, Head 0）
layer = 0
head = 0
# 形状: (batch, num_heads, seq_len, seq_len)
attention = outputs.attentions[layer][0, head].detach().cpu().numpy()

# 4. トークンの取得
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# 5. matplotlibによる可視化
plt.figure(figsize=(12, 10))
sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap='magma', annot=False)

plt.title(f"BigBird Attention Map (Layer {layer}, Head {head})")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()