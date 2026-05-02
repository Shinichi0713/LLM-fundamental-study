import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer


class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, num_heads=8, head_dim=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.in_proj = nn.Linear(d_model, self.d_inner * 2 + num_heads * 2 + d_state * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=False,
        )
        self.dt_bias =nn.Parameter(torch.rand(num_heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, num_heads + 1, dtype=torch.float32)))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x):
        batch, seq_len, _ = x.size()
        z_x_dt_bc = self.in_proj(x)
        z, x_inner, dt, B, C = torch.split(z_x_dt_bc, 
            [self.d_inner, self.d_inner, self.num_heads, self.d_state, self.d_state], dim=-1)
        
        # Conv1d 処理
        x_inner = x_inner.transpose(1, 2) # [B, D, L]
        x_inner = self.conv1d(x_inner)[:, :, :seq_len]
        x_inner = x_inner.transpose(1, 2) # [B, L, D]
        x_inner = F.silu(x_inner)

        

class SimpleSSM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)

        # SSMっぽい時間方向処理
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        x = self.linear2(x)
        return x + residual
    

class MambaLikeLM(nn.Module):
    def __init__(self, vocab_size, tokenizer=None, d_model=512, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SimpleSSM(d_model) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

    def generate(self, prompt, max_len=50, temperature=0.7):
        self.eval()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        for _ in range(max_len):
            with torch.no_grad():
                logits = self(input_ids)

            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return self.tokenizer.decode(input_ids[0])


