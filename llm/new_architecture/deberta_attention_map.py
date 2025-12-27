import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. モデル名の指定 (DeBERTa-v3-baseを使用します。小規模で高速です)
model_name = "microsoft/deberta-v3-base"

# 2. トークナイザとモデルの準備
# output_attentions=True を指定してアテンションの重みを出力させます
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

# GPUが利用可能ならGPUへ、なければCPUへモデルを移動
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # 推論モードに設定

# 3. アテンションマップ可視化関数
def plot_attention_map(attentions, tokens, layer_idx=0, head_idx=0):
    """
    指定された層とヘッドのアテンションマップを可視化します。

    Args:
        attentions (tuple): モデルの出力から得られるアテンション重みのタプル
        tokens (list): 入力テキストをトークン化したリスト
        layer_idx (int): 可視化するTransformer層のインデックス
        head_idx (int): 可視化するアテンションヘッドのインデックス
    """
    if layer_idx >= len(attentions) or head_idx >= attentions[0].shape[1]:
        print(f"Error: Layer {layer_idx} or Head {head_idx} is out of bounds.")
        print(f"Available layers: {len(attentions)}, Available heads: {attentions[0].shape[1]}")
        return

    # 指定された層とヘッドのアテンション重みを取得
    # attention_weightsの形状: (batch_size, num_heads, sequence_length, sequence_length)
    attention_weights = attentions[layer_idx][0, head_idx].cpu().detach().numpy()

    # マップのサイズ調整
    num_tokens = len(tokens)
    attention_weights = attention_weights[:num_tokens, :num_tokens]

    plt.figure(figsize=(num_tokens, num_tokens))
    sns.heatmap(attention_weights, cmap="viridis", annot=True, fmt=".2f",
                xticklabels=tokens, yticklabels=tokens,
                linewidths=.5, linecolor='lightgray')
    plt.title(f"DeBERTa Attention Map (Layer: {layer_idx}, Head: {head_idx})")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

text = "Despite the terrible weather and the long delay, the concert was surprisingly wonderful."

# 5. テキストの前処理と推論
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

with torch.no_grad(): # 勾配計算を無効化（推論時）
    outputs = model(**inputs)

# outputs.attentions はタプルで、各要素が (batch_size, num_heads, sequence_length, sequence_length) のテンソルです。
attentions = outputs.attentions
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# 6. アテンションマップの可視化実行
# 可視化したい層とヘッドのインデックスを指定
# DeBERTa-v3-baseは通常12層、12ヘッドです。
# 例: 0層目の0番目のヘッドを可視化
plot_attention_map(attentions, tokens, layer_idx=11, head_idx=11)