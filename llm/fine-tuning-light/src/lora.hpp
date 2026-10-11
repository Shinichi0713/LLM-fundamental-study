#pragma once
#include "tensor.hpp"
#include "optimizer.hpp"

// LoRA low-rank adaptation layer: W' = W + (alpha/r) * B @ A
struct LoRALinear {
    Tensor weight;      // frozen pretrained weight (out, in)
    Tensor lora_a;      // trainable (in, rank)
    Tensor lora_b;      // trainable (rank, out)
    float alpha;
    int rank;
    bool use_bias;
    Tensor bias;

    LoRALinear(size_t in_features, size_t out_features, int rank_, float alpha_, bool use_bias_ = true);

    Tensor forward(const Tensor& x);
    void backward(const Tensor& x, const Tensor& grad_out);
    void register_params(AdamW& opt);
    void zero_grad();
};
