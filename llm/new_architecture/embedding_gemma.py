from transformers import AutoTokenizer, AutoModel
from bertviz import head_view
import torch

# 1. モデル名の指定
# 事前に Hugging Face で google/embeddinggemma-300m の利用規約に同意し、
# ログイン（huggingface-cli login）しておく必要があります。
model_id = "google/embeddinggemma-300m"

# 2. トークナイザとモデルの準備
# output_attentions=True を指定するのがポイントです
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, output_attentions=True, device_map="auto")

# 3. 可視化したいテキスト
# EmbeddingGemmaはプロンプト（"query: "など）を付けるのが推奨されています
text = "query: How do transformers pay attention to each word?"

# 4. 推論とアテンションの取得
inputs = tokenizer.encode_plus(text, return_tensors='pt').to(model.device)
outputs = model(**inputs)

# アテンション、トークン、インプットIDを取得
attention = outputs.attentions
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# 5. 可視化（BertVizのHead Viewを使用）
head_view(attention, tokens)