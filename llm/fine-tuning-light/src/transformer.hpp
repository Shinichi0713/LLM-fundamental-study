#pragma once
#include "tensor.hpp"
#include "lora.hpp"
#include "nn.hpp"
#include <vector>
#include <memory>

struct MultiHeadAttention {
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;
    LoRALinear q_proj;
    LoRALinear k_proj;
    LoRALinear v_proj;
    LoRALinear o_proj;

    MultiHeadAttention(size_t embed_dim_, size_t num_heads_, int lora_rank, float lora_alpha);
    Tensor forward(const Tensor& x); // x: (B, T, E)
    void backward(const Tensor& x, const Tensor& grad_out);
    void register_params(AdamW& opt);
    void zero_grad();
};

struct FeedForward {
    LoRALinear up_proj;
    LoRALinear down_proj;
    GELU gelu;

    FeedForward(size_t embed_dim, size_t hidden_dim, int lora_rank, float lora_alpha);
    Tensor forward(const Tensor& x);
    void backward(const Tensor& x, const Tensor& grad_out);
    void register_params(AdamW& opt);
    void zero_grad();
};

struct TransformerBlock {
    MultiHeadAttention attn;
    FeedForward ffn;
    LayerNorm ln1;
    LayerNorm ln2;

    TransformerBlock(size_t embed_dim, size_t num_heads, size_t hidden_dim, int lora_rank, float lora_alpha);
    Tensor forward(const Tensor& x);
    void backward(const Tensor& x, const Tensor& grad_out);
    void register_params(AdamW& opt);
    void zero_grad();
};

struct CausalLM {
    size_t vocab_size;
    size_t embed_dim;
    size_t max_seq_len;
    Tensor token_embedding; // (vocab, embed)
    Tensor pos_embedding;   // (max_seq, embed)
    std::vector<TransformerBlock> blocks;
    LayerNorm final_ln;
    Linear lm_head;

    CausalLM(size_t vocab_size_, size_t embed_dim_, size_t num_heads, size_t hidden_dim,
             size_t num_layers, size_t max_seq_len_, int lora_rank, float lora_alpha);

    // tokens: (B, T) as 2D int -> embedded (B, T, E)
    Tensor embed(const std::vector<std::vector<int>>& tokens, size_t max_len);
    Tensor forward(const Tensor& x);
    Tensor logits(const Tensor& hidden);

    void register_params(AdamW& opt);
    void zero_grad();
};
