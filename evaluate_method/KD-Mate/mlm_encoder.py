
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from transformers import BertTokenizer, BertModel, BertForMaskedLM, RobertaForMaskedLM
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('bert_mlm')
model = BertForMaskedLM.from_pretrained("bert_mlm").to(device)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to(device)
output = model(**encoded_input)
print(output)

