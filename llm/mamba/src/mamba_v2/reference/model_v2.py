import torch
import torch.nn as nn
from mamba_ssm import Mamba
from transformers import AutoTokenizer

class MambaLM(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size
def generate(model, prompt, max_len=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(input_ids)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0])

model = MambaLM(vocab_size=vocab_size)
print("=== BEFORE ===")
print(generate(model, "Machine learning is"))