#include "lora.hpp"
#include <cmath>

LoRALinear::LoRALinear(size_t in_features, size_t out_features, int rank_, float alpha_, bool use_bias_)
    : alpha(alpha_), rank(rank_), use_bias(use_bias_) {
    weight = Tensor::zeros({out_features, in_features}, false); // frozen
    lora_a = Tensor::zeros({in_features, static_cast<size_t>(rank)});
    lora_b = Tensor::zeros({static_cast<size_t>(rank), out_features});
    // Initialize A with Kaiming uniform, B with zeros
    lora_a.randn(0.0f, 1.0f / std::sqrt(static_cast<float>(in_features)), 42);
    lora_b.fill(0.0f);
    if (use_bias) {
        bias = Tensor::zeros({out_features});
    }
}

Tensor LoRALinear::forward(const Tensor& x) {
    // base = x @ W^T
    Tensor base = matmul(x, weight.transpose(0, 1));
    // lora = x @ A @ B * (alpha / rank)
    Tensor tmp = matmul(x, lora_a);
    Tensor lora_out = matmul(tmp, lora_b);
    float scale = alpha / static_cast<float>(rank);
    for (auto& v : lora_out.data) v *= scale;
    // combine
    Tensor out = base + lora_out;
    if (use_bias) {
        for (size_t i = 0; i < out.shape[0]; ++i) {
            for (size_t j = 0; j < out.shape[1]; ++j) {
                out.data[i * out.shape[1] + j] += bias.data[j];
            }
        }
    }
    return out;
}

void LoRALinear::backward(const Tensor& x, const Tensor& grad_out) {
    float scale = alpha / static_cast<float>(rank);
    // grad for lora_b = tmp^T @ grad_out * scale
    // tmp = x @ lora_a
    Tensor tmp = matmul(x, lora_a);
    if (lora_b.requires_grad) {
        for (size_t i = 0; i < tmp.shape[0]; ++i) {
            for (size_t j = 0; j < grad_out.shape[1]; ++j) {
                float go = grad_out.data[i * grad_out.shape[1] + j] * scale;
                for (size_t k = 0; k < tmp.shape[1]; ++k) {
                    lora_b.grad[k * lora_b.shape[1] + j] += tmp.data[i * tmp.shape[1] + k] * go;
                }
            }
        }
    }
    // grad for lora_a = x^T @ (grad_out @ lora_b^T) * scale
    if (lora_a.requires_grad) {
        for (size_t i = 0; i < x.shape[0]; ++i) {
            for (size_t j = 0; j < grad_out.shape[1]; ++j) {
                float go = grad_out.data[i * grad_out.shape[1] + j] * scale;
                for (size_t k = 0; k < x.shape[1]; ++k) {
                    float xik = x.data[i * x.shape[1] + k];
                    for (size_t r = 0; r < static_cast<size_t>(rank); ++r) {
                        lora_a.grad[k * lora_a.shape[1] + r] += xik * lora_b.data[r * lora_b.shape[1] + j] * go;
                    }
                }
            }
        }
    }
    if (use_bias && bias.requires_grad) {
        for (size_t j = 0; j < bias.shape[0]; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < grad_out.shape[0]; ++i) {
                sum += grad_out.data[i * grad_out.shape[1] + j];
            }
            bias.grad[j] += sum;
        }
    }
}

void LoRALinear::register_params(AdamW& opt) {
    opt.add_param(&lora_a);
    opt.add_param(&lora_b);
    if (use_bias) opt.add_param(&bias);
}

void LoRALinear::zero_grad() {
    lora_a.zero_grad();
    lora_b.zero_grad();
    if (use_bias) bias.zero_grad();
}
