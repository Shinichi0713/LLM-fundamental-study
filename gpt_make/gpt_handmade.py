
import torch
import torch.nn as nn


# self attention
class SelfAttentionHead(nn.Module):
    def __init__(self, n_mbed, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_mbed, head_size, bias=False)
        self.query = nn.Linear(n_mbed, head_size, bias=False)
        self.value = nn.Linear(n_mbed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)*C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.nn.functional.softmax(wei, dim=-1)

        out = wei @ v
        return out

# multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_mbed, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(n_mbed, head_size, block_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, n_mbed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_mbed, n_mbed),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# synthesized model
class ModelTransformer(nn.Module):
    def __init__(self, n_mbed, char_size, block_size, number_heads, device):
        super().__init__()
        self.token_embedding = nn.Embedding(char_size, n_mbed).to(device)
        self.pos_embedding = nn.Embedding(block_size, n_mbed).to(device)
        self.selfattention_multiheads = MultiHeadAttention(n_mbed, number_heads, n_mbed // number_heads, block_size).to(device)
        self.feedforward = FeedForward(n_mbed).to(device)
        self.linear = nn.Linear(n_mbed, char_size).to(device)
        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_mbed = self.token_embedding(idx)
        position_mbed = self.pos_embedding(torch.arange(T).to(self.device))
        x = token_mbed + position_mbed
        x = self.selfattention_multiheads(x)
        x = self.feedforward(x)
        logits = self.linear(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(-1, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=0)
        else:
            loss = None
        return logits, loss