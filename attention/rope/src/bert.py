import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ==============================
# 1. 各コンポーネントの定義（Hugging FaceのBert構造を模倣）
# ==============================

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=128, max_position_embeddings=512, type_vocab_size=2, dropout_prob=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size=128, num_attention_heads=4, dropout_prob=0.1):
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

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 簡易版：スケールド・ドットプロダクトアテンション
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
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
    def __init__(self, hidden_size=128, num_attention_heads=4, dropout_prob=0.1):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.output = BertSelfOutput(hidden_size, dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
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
    def __init__(self, hidden_size=128, num_attention_heads=4, intermediate_size=512, dropout_prob=0.1):
        super().__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads, dropout_prob)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size, dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, num_hidden_layers=4, hidden_size=128, num_attention_heads=4, intermediate_size=512, dropout_prob=0.1):
        super().__init__()
        self.layer = nn.ModuleList([
            BertLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size=128, vocab_size=30522):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, hidden_size=128, vocab_size=30522):
        super().__init__()
        self.predictions = BertLMPredictionHead(hidden_size, vocab_size)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertModel(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                 intermediate_size=512, max_position_embeddings=512, type_vocab_size=2, dropout_prob=0.1):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout_prob)
        self.encoder = BertEncoder(num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, dropout_prob)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        return encoder_outputs


class CustomBertForMaskedLM(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                 intermediate_size=512, max_position_embeddings=512, type_vocab_size=2, dropout_prob=0.1):
        super().__init__()
        self.bert = BertModel(vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                              intermediate_size, max_position_embeddings, type_vocab_size, dropout_prob)
        self.cls = BertOnlyMLMHead(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids)
        sequence_output = outputs
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.cls.predictions.decoder.out_features), labels.view(-1))

        return {
            "loss": loss,
            "logits": prediction_scores,
        }


# ==============================
# 2. Hugging Faceのモデルをロードして重みをコピー
# ==============================

def load_hf_model_and_copy_weights(model_name="boltuix/bert-mini"):
    # Hugging Faceのモデルとトークナイザをロード
    hf_model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 自作モデルを同じ設定で初期化
    config = hf_model.config
    custom_model = CustomBertForMaskedLM(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        type_vocab_size=config.type_vocab_size,
        dropout_prob=config.hidden_dropout_prob,
    )

    # 重みをコピー（名前が一致するもののみ）
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = custom_model.state_dict()

    # 名前マッピング（必要に応じて調整）
    name_map = {}
    for name, param in hf_state_dict.items():
        # 例: "bert.embeddings.word_embeddings.weight" → "bert.embeddings.word_embeddings.weight"
        # 自作モデルの名前と一致する場合はそのまま使う
        if name in custom_state_dict:
            name_map[name] = name
        # 必要に応じて変換ルールを追加
        # 例: "bert.encoder.layer.0.attention.self.query.weight" → "bert.encoder.layer.0.attention.self.query.weight"

    # コピー実行
    for hf_name, custom_name in name_map.items():
        if hf_name in hf_state_dict and custom_name in custom_state_dict:
            if hf_state_dict[hf_name].shape == custom_state_dict[custom_name].shape:
                custom_state_dict[custom_name].copy_(hf_state_dict[hf_name])
            else:
                print(f"[WARN] Shape mismatch: {hf_name} {hf_state_dict[hf_name].shape} vs {custom_name} {custom_state_dict[custom_name].shape}")

    custom_model.load_state_dict(custom_state_dict, strict=False)  # strict=Falseで一部だけコピー
    return custom_model, tokenizer


# ==============================
# 3. 動作確認
# ==============================

if __name__ == "__main__":
    # モデルとトークナイザをロード
    custom_model, tokenizer = load_hf_model_and_copy_weights("boltuix/bert-mini")

    # テキストをトークナイズ
    text = "The lecture was held in the [MASK] hall."
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    # 推論
    custom_model.eval()
    with torch.no_grad():
        outputs = custom_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]

    # [MASK] 位置を特定
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]

    # マスク位置の予測トップ5
    for pos in mask_positions:
        mask_logits = logits[0, pos, :]
        top_5 = torch.topk(mask_logits, 5)
        for i, (token_id, score) in enumerate(zip(top_5.indices, top_5.values)):
            token = tokenizer.decode(token_id)
            print(f"Position {pos}: {i+1}. {token} (score={score:.4f})")