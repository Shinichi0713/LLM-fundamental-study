#pragma once
#include "tensor.hpp"
#include <vector>
#include <memory>

struct Linear {
    Tensor weight; // (out_features, in_features)
    Tensor bias;   // (out_features,)
    bool use_bias = true;

    Linear(size_t in_features, size_t out_features, bool use_bias_ = true);
    Tensor forward(const Tensor& x); // x: (..., in_features)
    void backward(const Tensor& x, const Tensor& grad_out);
    void update(AdamW& opt);
};

struct LayerNorm {
    size_t normalized_shape;
    Tensor gamma;
    Tensor beta;
    float eps = 1e-5f;

    LayerNorm(size_t normalized_shape_);
    Tensor forward(const Tensor& x);
};

struct ReLU {
    Tensor forward(const Tensor& x);
};

struct GELU {
    Tensor forward(const Tensor& x);
};

struct SoftmaxCrossEntropy {
    float loss(const Tensor& logits, const std::vector<int>& targets);
    Tensor backward(const Tensor& logits, const std::vector<int>& targets);
};

struct KLDivLoss {
    float loss(const Tensor& log_probs, const Tensor& target_probs);
    Tensor backward(const Tensor& log_probs, const Tensor& target_probs);
};

// Forward declaration for AdamW (defined in optimizer.hpp)
struct AdamW;
