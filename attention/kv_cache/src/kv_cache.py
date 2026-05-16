import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)  # (B, S, D)
        k = self.k_proj(x)  # (B, S, D)
        v = self.v_proj(x)  # (B, S, D)

        # マルチヘッド化
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, H, S, head_dim)

        # KVキャッシュがある場合は結合
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # 新しいKVキャッシュを返す
        new_kv = (k, v)

        # スコア計算
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 出力
        out = torch.matmul(attn_weights, v)  # (B, H, S, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        return out, new_kv
    
class SimpleModelWithCache(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SimpleAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, past_kv_list=None):
        # input_ids: (batch_size, seq_len)
        x = self.embed(input_ids)  # (B, S, D)

        new_kv_list = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, new_kv = layer(x, past_kv)
            new_kv_list.append(new_kv)

        logits = self.lm_head(x)  # (B, S, vocab_size)
        return logits, new_kv_list

    def generate(self, input_ids, max_new_tokens=20):
        # 簡易的な生成ループ
        model.eval()
        with torch.no_grad():
            cur_ids = input_ids
            past_kv_list = None

            for step in range(max_new_tokens):
                logits, past_kv_list = self.forward(cur_ids, past_kv_list)
                # 最後のトークンのロジットだけを使う
                next_logits = logits[:, -1, :]  # (B, vocab_size)
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1)

                cur_ids = torch.cat([cur_ids, next_id], dim=1)

            return cur_ids
        
# transformers/src/transformers/models/gpt2/modeling_gpt2.py のイメージ
class GPT2Attention(nn.Module):
    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Cache] = None,
        # ... 他の引数
    ):
        # Query, Key, Value の計算
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # ... マルチヘッド化やRoPE適用など

        if past_key_value is not None:
            # キャッシュに新しいKey/Valueを追加し、過去分も含めた全体を取得
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        # ここで key_states, value_states には「過去 + 現在」の全トークンが入っている
        # Attention 計算（query_states と key_states, value_states の積）

        # 出力と、次のステップで使うためのKVキャッシュを返す
        return attn_output, key_states, value_states
    
# transformers/src/transformers/models/gpt2/modeling_gpt2.py のイメージ
class GPT2Model(GPT2PreTrainedModel):
    def forward(
        self,
        input_ids,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        # ... 他の引数
    ):
        # ... 埋め込みなど

        # KVキャッシュの初期化
        if past_key_values is None:
            past_key_values = DynamicCache()

        # 過去に処理済みのトークン数を取得
        past_seen_tokens = past_key_values.get_seq_length()

        # position_ids をオフセット（RoPEなどで正しい位置情報を付与するため）
        position_ids = ... + past_seen_tokens

        # 各レイヤーでAttentionを呼び出し、KVキャッシュを更新
        for idx, block in enumerate(self.h):
            layer_outputs = block(
                hidden_states,
                past_key_value=past_key_values,
                # ... 他の引数
            )
            hidden_states = layer_outputs[0]

        # 出力と更新されたKVキャッシュを返す
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            # ... 他のフィールド
        )