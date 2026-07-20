import torch
import torch.nn as nn
import math
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import BertTokenizer

# ==========================================
# 1. RoPE（回転位置埋め込み）のヘルパークラス
# ==========================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        # 各次元に対する回転角の周波数を計算
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        
        # あらかじめ大きめのサイズで sin, cos のキャッシュを作成
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # [max_len, dim] -> [1, 1, max_len, dim] に整形してバッファ登録
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [batch, num_heads, seq_len, head_dim]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device)
        )

def rotate_half(x):
    # ベクトルの前半と後半を分けて回転させるための処理
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # RoPEの数式: R_theta * x = x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==========================================
# 2. BERTのSelfAttentionをRoPE対応に拡張
# ==========================================
class RoBERTSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        # ヘッドごとの次元数（例: 768 / 12 = 64）
        self.head_dim = config.hidden_size // config.num_attention_heads
        # RoPEモジュールの初期化
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # 通常のBERTと同様に Query, Key, Value を線形変換
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # ----------------------------------------------------
        # RoPEの適用ステップ
        # ----------------------------------------------------
        seq_len = query_layer.shape[2]
        cos, sin = self.rotary_emb(query_layer, seq_len=seq_len)
        
        # QueryとKeyに回転行列を適用（Valueはそのまま）
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)
        # ----------------------------------------------------

        # 以降は通常のBERTのアテンション行列計算（Softmax -> Valueとの積）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# ==========================================
# 3. 実験用モデルのビルド関数
# ==========================================
def build_experiment_models(model_name="bert-base-uncased"):
    """
    実験1に必要な Baseline (標準BERT) と Proposed (RoPE BERT) を読み込む関数
    """
    print(f"Loading baseline model: {model_name}")
    # ① Baseline: 標準BERT（絶対位置）
    baseline_model = BertModel.from_pretrained(model_name)
    
    print(f"Building RoPE modified model...")
    # ② Proposed: RoPE BERT
    # configを取得し、同じ重みで一度初期化する
    config = BertConfig.from_pretrained(model_name)
    rope_model = BertModel.from_pretrained(model_name)
    
    # 既存の絶対位置埋め込み（Absolute Position Embeddings）を無効化（ゼロクリア）
    # これにより、モデルは入力層で位置情報を足さなくなります
    nn.init.zeros_(rope_model.embeddings.position_embeddings.weight)
    rope_model.embeddings.position_embeddings.weight.requires_grad = False
    
    # BERTの全LayerのSelfAttentionクラスを、RoPE版（RoBERTSelfAttention）に差し替える
    for i in range(config.num_hidden_layers):
        # 既存の重みをコピーするために現在の状態を保持
        old_attention = rope_model.encoder.layer[i].attention.self
        
        # RoPE版アテンションをインスタンス化
        new_attention = RoBERTSelfAttention(config)
        
        # Query, Key, Value, Dropout などの学習済み重みをそのまま移植
        new_attention.query.load_state_dict(old_attention.query.state_dict())
        new_attention.key.load_state_dict(old_attention.key.state_dict())
        new_attention.value.load_state_dict(old_attention.value.state_dict())
        
        # 差し替えを実行
        rope_model.encoder.layer[i].attention.self = new_attention

    return baseline_model, rope_model


# ==========================================
# 4. 動作確認用のテストコード
# ==========================================
if __name__ == "__main__":
    # モデルのビルド
    baseline, rope_bert = build_experiment_models("bert-base-uncased")
    
    # ダミー入力の作成 (Batch=2, Seq_Len=10)
    dummy_input_ids = torch.randint(0, 30000, (2, 10))
    dummy_mask = torch.ones((2, 10))
    
    # 拡張されたアテンションマスク（HFのモデル内部で処理される形式を模倣）
    extended_mask = dummy_mask.unsqueeze(1).unsqueeze(2)
    extended_mask = (1.0 - extended_mask) * -10000.0
    
    # 推論テスト
    baseline.eval()
    rope_bert.eval()
    
    with torch.no_grad():
        out_base = baseline(dummy_input_ids, attention_mask=extended_mask)
        out_rope = rope_bert(dummy_input_ids, attention_mask=extended_mask)
        
    print("\n--- Test Output ---")
    print("Baseline Last Hidden State Shape:", out_base.last_hidden_state.shape)
    print("RoPE-BERT Last Hidden State Shape:", out_rope.last_hidden_state.shape)
    print("正しく両方のモデルからテンソルが出力されました。")




def test_models_pipeline():
    print("=== 1. トークナイザとモデルの準備 ===")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Baseline と RoPE-BERT をビルド
    baseline_model, rope_model = build_experiment_models(model_name)
    
    # 分類タスク（実験1）を想定し、簡単な線形層（ヘッド）を後ろに結合する
    # ここでは2クラス分類（ポジ・ネガ等）と仮定
    num_labels = 2
    baseline_head = nn.Linear(baseline_model.config.hidden_size, num_labels)
    rope_head = nn.Linear(rope_model.config.hidden_size, num_labels)
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n=== 2. テスト用データの作成 ===")
    # 実験1を模したサンプルテキスト（標準的な長さ）と正解ラベル
    sample_texts = [
        "Large language models are transforming the field of artificial intelligence.",
        "Rotary position embedding scales effectively to longer contexts in transformers.",
        "This is a short sentence to verify the baseline capability of the model.",
        "We need to ensure that modifying the attention block does not break initial weights."
    ]
    # ダミーの正解ラベル (0か1)
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    
    # トークナイズ処理 (BERTの標準上限である512未満、ここではパディング込みで最大長に合わせる)
    inputs = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    print(f"入力テンソルの形状 (Batch, SeqLen): {inputs['input_ids'].shape}")
    
    print("\n=== 3. Baselineモデルのテスト (Train Mode) ===")
    baseline_model.train()
    baseline_head.train()
    
    # 順伝播 (Forward)
    outputs_base = baseline_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"]
    )
    # [CLS]トークンの表現を取り出して分類器へ
    cls_base = outputs_base.last_hidden_state[:, 0, :]
    logits_base = baseline_head(cls_base)
    loss_base = criterion(logits_base, labels)
    
    print(f"Baseline - Loss: {loss_base.item():.4f}")
    
    # 逆伝播 (Backward) の検証
    loss_base.backward()
    print("✓ Baseline: 逆伝播と勾配計算に成功しました。")
    
    print("\n=== 4. RoPE-BERTモデルのテスト (Train Mode) ===")
    rope_model.train()
    rope_head.train()
    
    # 順伝播 (Forward) 
    # ※内部のBertSelfAttentionがRoPE版に差し替わっているため、自動的に回転行列が適用されます
    outputs_rope = rope_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"]
    )
    # [CLS]トークンの表現を取り出して分類器へ
    cls_rope = outputs_rope.last_hidden_state[:, 0, :]
    logits_rope = rope_head(cls_rope)
    loss_rope = criterion(logits_rope, labels)
    
    print(f"RoPE-BERT - Loss: {loss_rope.item():.4f}")
    
    # 逆伝播 (Backward) の検証
    loss_rope.backward()
    print("✓ RoPE-BERT: 逆伝播と勾配計算に成功しました。")
    
    print("\n=== 5. パラメータ固定状態の最終チェック ===")
    # 実験計画通り、絶対位置埋め込みの勾配が固定（False）されているか確認
    rope_pos_emb_grad = rope_model.embeddings.position_embeddings.weight.requires_grad
    print(f"RoPE-BERTの旧絶対位置埋め込みの更新可否 (Expected: False): {rope_pos_emb_grad}")
    
    # RoPEアテンション内の重みが勾配を持っているか確認
    sample_rope_weight = rope_model.encoder.layer[0].attention.self.query.weight.grad
    has_grad = sample_rope_weight is not None
    print(f"RoPE-BERTのアテンション層に勾配が正しく伝播しているか (Expected: True): {has_grad}")
    
    if not rope_pos_emb_grad and has_grad:
        print("\n[SUCCESS] すべてのテストをパスしました！実験1の追加学習フェーズに移行可能です。")
    else:
        print("\n[FAILURE] 一部モデルの設定が意図通りになっていません。")

if __name__ == "__main__":
    test_models_pipeline()