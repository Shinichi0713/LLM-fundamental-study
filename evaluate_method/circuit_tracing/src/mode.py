import torch
from transformers import BertTokenizer, BertForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

