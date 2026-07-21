import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------------------------------------------
# 1. ヘルパー関数 & RoPEモジュール
# -------------------------------------------------------------
def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    seq_len = x.size(2)
    cos = cos[:seq_len, :]
    sin = sin[:seq_len, :]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(2)
        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )

# -------------------------------------------------------------
# 2. BERT構成要素 (修正反映済み)
# -------------------------------------------------------------
class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size=128, num_attention_heads=4, dropout_prob=0.1, max_position_embeddings=512):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.rotary_emb = RotaryPositionEmbedding(
            dim=self.attention_head_size,
            max_seq_len=max_position_embeddings,
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 修正: self.rotary_emb に名前統一
        cos, sin = self.rotary_emb(query_layer)
        query_layer = apply_rotary_pos_emb(query_layer, cos, sin)
        key_layer = apply_rotary_pos_emb(key_layer, cos, sin)

        # 修正: 変数名を attention_scores に統一
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

            # 修正: logits.dtype を query_layer.dtype に変更
            if attention_mask.dtype == torch.long or (attention_mask.max() <= 1.0 and attention_mask.min() >= 0.0):
                extended_attention_mask = (1.0 - attention_mask.to(dtype=query_layer.dtype)) * -10000.0
            else:
                extended_attention_mask = attention_mask

            attention_scores = attention_scores + extended_attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size=128, dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, hidden_size=128, num_attention_heads=4, dropout_prob=0.1, max_position_embeddings=512):
        super().__init__()
        self.atten = BertSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout_prob=dropout_prob,
            max_position_embeddings=max_position_embeddings,
        )
        self.output = BertSelfOutput(hidden_size=hidden_size, dropout_prob=dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.atten(hidden_states, attention_mask, head_mask)
        attention_output = self.output(attention_output, hidden_states)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size=128, intermediate_size=512):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, hidden_size=128, intermediate_size=512, dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, hidden_size=128, num_attention_heads=4, intermediate_size=512, dropout_prob=0.1, max_position_embeddings=512):
        super().__init__()
        # 修正: max_position_embeddings を渡す
        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout_prob=dropout_prob,
            max_position_embeddings=max_position_embeddings,
        )
        self.intermediate = BertIntermediate(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.output = BertOutput(hidden_size=hidden_size, intermediate_size=intermediate_size, dropout_prob=dropout_prob)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, num_hidden_layers=4, hidden_size=128, num_attention_heads=4, intermediate_size=512, dropout_prob=0.1, max_position_embeddings=512):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layer = nn.ModuleList([
            BertLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob, max_position_embeddings) 
            for _ in range(num_hidden_layers)
        ])
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if (self.training and head_mask is not None) else None
            hidden_states = layer_module(hidden_states, attention_mask, layer_head_mask)
        return hidden_states

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=128, type_vocab_size=2, dropout_prob=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModel(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                 intermediate_size=512, max_position_embeddings=512, type_vocab_size=2, dropout_prob=0.1):
        super().__init__()
        # 修正: max_position_embeddings の渡しを整理
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            type_vocab_size=type_vocab_size,
            dropout_prob=dropout_prob,
        )
        self.encoder = BertEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout_prob=dropout_prob,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        return encoder_outputs


# -------------------------------------------------------------
# 3. デモ実行スクリプト
# -------------------------------------------------------------
if __name__ == "__main__":
    # パラメータ設定
    BATCH_SIZE = 2
    SEQ_LEN = 16
    VOCAB_SIZE = 30522
    HIDDEN_SIZE = 128

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 1. モデルのインスタンス化
    model = BertModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=512
    ).to(device)
    model.eval()

    # 2. ダミー入力データの生成
    # ランダムなトークンID (0 ~ VOCAB_SIZE-1)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

    # アテンションマスク (1: 通常トークン, 0: パディングトークン)
    # 例として後半をパディング（0）にする
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), device=device)
    attention_mask[:, 10:] = 0  # 10トークン目以降をパディング扱い

    token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=device)

    print("\n--- 入力データ形状 ---")
    print(f"input_ids:      {input_ids.shape}")
    print(f"attention_mask: {attention_mask.shape}")

    # 3. 順伝播の実行
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    print("\n--- 出力データ形状 ---")
    print(f"モデル最終出力 (hidden_states): {outputs.shape}")
    print("\n正常に順伝播が完了しました！")