import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class CustomSmallEncoder(nn.Module):
    def __init__(self, dim=256, vocab_size=30522, n_layers=4, n_heads=8):
        super().__init__()
        self.dim = dim # trainerで使うため保存
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim*4, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

        # --- 追加: MLM用出力層 ---
        self.mlm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        all_hidden_states = []
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                all_hidden_states.append(x)
        x = self.norm(x)

        # --- 追加: ロジットの計算 ---
        logits = self.mlm_head(x)

        class Output:
            pass
        out = Output()
        out.last_hidden_state = x
        out.hidden_states = all_hidden_states if output_hidden_states else None
        out.logits = logits # これを返すことで評価可能になる
        return out
    
# --- 教師モデル（既存のBERT） ---
# 語彙サイズはBERT標準の30522、次元は768
teacher = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# --- 生徒モデル（あなたのカスタムモデル） ---
# 教師と同じ語彙サイズを指定し、次元は軽量な256に設定
student = CustomSmallEncoder(
    dim=256, 
    vocab_size=teacher.config.vocab_size, 
    n_layers=6
)

# --- 動作確認 ---
sample_input = torch.randint(0, 30522, (1, 128)) # Batch=1, SeqLen=128
t_out = teacher(sample_input)
s_out = student(sample_input)

print(f"Teacher hidden state shape: {t_out.hidden_states[-1].shape}") # [1, 128, 768]
print(f"Student hidden state shape: {s_out.hidden_states[-1].shape}") # [1, 128, 256]