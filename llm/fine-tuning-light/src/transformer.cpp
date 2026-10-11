#include "transformer.hpp"
#include <cmath>
#include <iostream>

// ---------- MultiHeadAttention ----------
MultiHeadAttention::MultiHeadAttention(size_t embed_dim_, size_t num_heads_, int lora_rank, float lora_alpha)
    : embed_dim(embed_dim_), num_heads(num_heads_), head_dim(embed_dim_ / num_heads_),
      q_proj(embed_dim_, embed_dim_, lora_rank, lora_alpha, false),
      k_proj(embed_dim_, embed_dim_, lora_rank, lora_alpha, false),
      v_proj(embed_dim_, embed_dim_, lora_rank, lora_alpha, false),
      o_proj(embed_dim_, embed_dim_, lora_rank, lora_alpha, false) {}

Tensor MultiHeadAttention::forward(const Tensor& x) {
    // x: (B, T, E)
    size_t B = x.shape[0];
    size_t T = x.shape[1];
    size_t E = x.shape[2];
    Tensor q = q_proj.forward(x.view({B * T, E})).view({B, T, num_heads, head_dim});
    Tensor k = k_proj.forward(x.view({B * T, E})).view({B, T, num_heads, head_dim});
    Tensor v = v_proj.forward(x.view({B * T, E})).view({B, T, num_heads, head_dim});

    // Scaled dot-product attention with causal mask
    // scores: (B, num_heads, T, T)
    Tensor scores({B, num_heads, T, T}, false);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t t1 = 0; t1 < T; ++t1) {
                float max_score = -1e30f;
                for (size_t t2 = 0; t2 <= t1; ++t2) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t qidx = ((b * T + t1) * num_heads + h) * head_dim + d;
                        size_t kidx = ((b * T + t2) * num_heads + h) * head_dim + d;
                        dot += q.data[qidx] * k.data[kidx];
                    }
                    dot *= scale;
                    scores.data[((b * num_heads + h) * T + t1) * T + t2] = dot;
                    if (dot > max_score) max_score = dot;
                }
                // softmax over valid positions
                float sum_exp = 0.0f;
                for (size_t t2 = 0; t2 <= t1; ++t2) {
                    size_t sidx = ((b * num_heads + h) * T + t1) * T + t2;
                    scores.data[sidx] = std::exp(scores.data[sidx] - max_score);
                    sum_exp += scores.data[sidx];
                }
                for (size_t t2 = 0; t2 <= t1; ++t2) {
                    size_t sidx = ((b * num_heads + h) * T + t1) * T + t2;
                    scores.data[sidx] /= sum_exp;
                }
                // causal mask: zero out future
                for (size_t t2 = t1 + 1; t2 < T; ++t2) {
                    scores.data[((b * num_heads + h) * T + t1) * T + t2] = 0.0f;
                }
            }
        }
    }

    // Apply attention to values
    Tensor attn_out({B, T, num_heads, head_dim}, false);
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t t1 = 0; t1 < T; ++t1) {
                for (size_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (size_t t2 = 0; t2 <= t1; ++t2) {
                        float a = scores.data[((b * num_heads + h) * T + t1) * T + t2];
                        size_t vidx = ((b * T + t2) * num_heads + h) * head_dim + d;
                        sum += a * v.data[vidx];
                    }
                    attn_out.data[((b * T + t1) * num_heads + h) * head_dim + d] = sum;
                }
            }
        }
    }

    // Output projection
    Tensor out = o_proj.forward(attn_out.view({B * T, E})).view({B, T, E});
    return out;
}

void MultiHeadAttention::backward(const Tensor& x, const Tensor& grad_out) {
    // Simplified: accumulate grads through o_proj
    o_proj.backward(x.view({x.shape[0] * x.shape[1], x.shape[2]}), grad_out.view({x.shape[0] * x.shape[1], x.shape[2]}));
    q_proj.backward(x.view({x.shape[0] * x.shape[1], x.shape[2]}), grad_out.view({x.shape[0] * x.shape[1], x.shape[2]}));
    k_proj.backward(x.view({x.shape[0] * x.shape[1], x.shape[2]}), grad_out.view({x.shape[0] * x.shape[1], x.shape[2]}));
    v_proj.backward(x.view({x.shape[0] * x.shape[1], x.shape[2]}), grad_out.view({x.shape[0] * x.shape[1], x.shape[2]}));
}

void MultiHeadAttention::register_params(AdamW& opt) {
    q_proj.register_params(opt);
    k_proj.register_params(opt);
    v_proj.register_params(opt);
    o_proj.register_params(opt);
}
void MultiHeadAttention::zero_grad() {
    q_proj.zero_grad(); k_proj.zero_grad(); v_proj.zero_grad(); o_proj.zero_grad();
}

// ---------- FeedForward ----------
FeedForward::FeedForward(size_t embed_dim, size_t hidden_dim, int lora_rank, float lora_alpha)
    : up_proj(embed_dim, hidden_dim, lora_rank, lora_alpha, false),
      down_proj(hidden_dim, embed_dim, lora_rank, lora_alpha, false) {}

Tensor FeedForward::forward(const Tensor& x) {
    Tensor h = up_proj.forward(x.view({x.shape[0] * x.shape[1], x.shape[2]}));
    Tensor a = gelu.forward(h);
    Tensor out = down_proj.forward(a);
    return out.view({x.shape[0], x.shape[1], x.shape[2]});
}

void FeedForward::backward(const Tensor& x, const Tensor& grad_out) {
    // Simplified accumulation
    down_proj.backward(x.view({x.shape[0] * x.shape[1], x.shape[2]}), grad_out.view({x.shape[0] * x.shape[1], x.shape[2]}));
    up_proj.backward(x.view({x.shape[0] * x.shape[1], x.shape[2]}), grad_out.view({x.shape[0] * x.shape[1], x.shape[2]}));
}

void FeedForward::register_params(AdamW& opt) {
    up_proj.register_params(opt);
    down_proj.register_params(opt);
}
void FeedForward::zero_grad() {
    up_proj.zero_grad(); down_proj.zero_grad();
}

// ---------- TransformerBlock ----------
TransformerBlock::TransformerBlock(size_t embed_dim, size_t num_heads, size_t hidden_dim, int lora_rank, float lora_alpha)
    : attn(embed_dim, num_heads, lora_rank, lora_alpha),
      ffn(embed_dim, hidden_dim, lora_rank, lora_alpha),
      ln1(embed_dim), ln2(embed_dim) {}

Tensor TransformerBlock::forward(const Tensor& x) {
    Tensor h1 = ln1.forward(x);
    Tensor a = attn.forward(h1);
    Tensor add1 = x + a;
    Tensor h2 = ln2.forward(add1);
    Tensor f = ffn.forward(h2);
    Tensor out = add1 + f;
    return out;
}

void TransformerBlock::backward(const Tensor& x, const Tensor& grad_out) {
    attn.backward(x, grad_out);
    ffn.backward(x, grad_out);
}

void TransformerBlock::register_params(AdamW& opt) {
    attn.register_params(opt);
    ffn.register_params(opt);
    opt.add_param(&ln1.gamma);
    opt.add_param(&ln1.beta);
    opt.add_param(&ln2.gamma);
    opt.add_param(&ln2.beta);
}
void TransformerBlock::zero_grad() {
    attn.zero_grad(); ffn.zero_grad();
    ln1.gamma.zero_grad(); ln1.beta.zero_grad();
    ln2.gamma.zero_grad(); ln2.beta.zero_grad();
}

// ---------- CausalLM ----------
CausalLM::CausalLM(size_t vocab_size_, size_t embed_dim_, size_t num_heads, size_t hidden_dim,
                   size_t num_layers, size_t max_seq_len_, int lora_rank, float lora_alpha)
    : vocab_size(vocab_size_), embed_dim(embed_dim_), max_seq_len(max_seq_len_),
      token_embedding({vocab_size_, embed_dim_}, false),
      pos_embedding({max_seq_len_, embed_dim_}, false),
      final_ln(embed_dim_),
      lm_head(embed_dim_, vocab_size_, true) {
    token_embedding.xavier_uniform(embed_dim_, vocab_size_, 42);
    pos_embedding.xavier_uniform(embed_dim_, max_seq_len_, 43);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks.emplace_back(embed_dim_, num_heads, hidden_dim, lora_rank, lora_alpha);
    }
}

Tensor CausalLM::embed(const std::vector<std::vector<int>>& tokens, size_t max_len) {
    size_t B = tokens.size();
    Tensor out({B, max_len, embed_dim}, false);
    for (size_t b = 0; b < B; ++b) {
        size_t len = std::min(tokens[b].size(), max_len);
        for (size_t t = 0; t < len; ++t) {
            int tok_id = tokens[b][t];
            if (tok_id < 0 || tok_id >= static_cast<int>(vocab_size)) tok_id = 0;
            for (size_t e = 0; e < embed_dim; ++e) {
                out.data[(b * max_len + t) * embed_dim + e] =
                    token_embedding.data[tok_id * embed_dim + e] +
                    pos_embedding.data[t * embed_dim + e];
            }
        }
    }
    return out;
}

Tensor CausalLM::forward(const Tensor& x) {
    Tensor h = x;
    for (auto& block : blocks) {
        h = block.forward(h);
    }
    h = final_ln.forward(h);
    return h;
}

Tensor CausalLM::logits(const Tensor& hidden) {
    size_t B = hidden.shape[0];
    size_t T = hidden.shape[1];
    size_t E = hidden.shape[2];
    return lm_head.forward(hidden.view({B * T, E})).view({B, T, vocab_size});
}

void CausalLM::register_params(AdamW& opt) {
    for (auto& block : blocks) block.register_params(opt);
    opt.add_param(&final_ln.gamma);
    opt.add_param(&final_ln.beta);
    opt.add_param(&lm_head.weight);
    if (lm_head.use_bias) opt.add_param(&lm_head.bias);
}

void CausalLM::zero_grad() {
    for (auto& block : blocks) block.zero_grad();
    final_ln.gamma.zero_grad(); final_ln.beta.zero_grad();
    lm_head.weight.zero_grad();
    if (lm_head.use_bias) lm_head.bias.zero_grad();
}
