import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttentionWithCache(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_key_value=None):
        # x: [batch, 1, d_model]  (推論時は通常、最新の1トークンのみ入力)
        batch, seq_len, d_model = x.shape
        head_dim = d_model // self.n_head

        # 1. Q, K, V の計算
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        
        # [batch, n_head, seq_len, head_dim] に変換
        q = q.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)

        # 2. KVキャッシュの適用
        if past_key_value is not None:
            prev_k, prev_v = past_key_value
            # 過去のK, Vと現在のK, Vを結合 (seq_len方向に結合)
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        
        # 今回のステップのK, Vを保存用として出力
        present_key_value = (k, v)

        # 3. Attention計算 (Qは最新の1つ、K, Vは過去すべて)
        # q: [b, h, 1, d], k: [b, h, total_seq, d]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        
        return self.out_proj(out), present_key_value
    


def generate_with_cache(model, tokenizer, prompt, max_new_tokens=20):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # 全レイヤー分のキャッシュを保持するリスト (最初はNone)
    past_key_values = None
    
    generated_ids = input_ids
    next_token_id = input_ids

    for _ in range(max_new_tokens):
        # モデルのforwardにキャッシュを渡す
        # 最初のステップ以外は、最新の1トークン(next_token_id)だけ入力すれば良い！
        outputs = model(next_token_id, past_key_values=past_key_values)
        
        logits = outputs.logits  # [batch, 1, vocab_size]
        past_key_values = outputs.past_key_values # 更新されたキャッシュを受け取る
        
        # 次のトークンを選択
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        if next_token_id == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(generated_ids[0])