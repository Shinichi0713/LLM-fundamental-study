#include "nn.hpp"
#include "optimizer.hpp"
#include <cmath>

// ---------- Linear ----------
Linear::Linear(size_t in_features, size_t out_features, bool use_bias_)
    : use_bias(use_bias_) {
    weight = Tensor::zeros({out_features, in_features});
    weight.xavier_uniform(in_features, out_features, 42);
    if (use_bias) {
        bias = Tensor::zeros({out_features});
    }
}

Tensor Linear::forward(const Tensor& x) {
    // x: (B, in_features), weight: (out, in)
    if (x.ndim() != 2) throw std::runtime_error("Linear: only 2D input supported");
    Tensor out = matmul(x, weight.transpose(0, 1));
    if (use_bias) {
        for (size_t i = 0; i < out.shape[0]; ++i) {
            for (size_t j = 0; j < out.shape[1]; ++j) {
                out.data[i * out.shape[1] + j] += bias.data[j];
            }
        }
    }
    return out;
}

void Linear::backward(const Tensor& x, const Tensor& grad_out) {
    // grad_weight = x^T @ grad_out
    if (weight.requires_grad) {
        for (size_t i = 0; i < x.shape[0]; ++i) {
            for (size_t j = 0; j < grad_out.shape[1]; ++j) {
                float go = grad_out.data[i * grad_out.shape[1] + j];
                for (size_t k = 0; k < x.shape[1]; ++k) {
                    weight.grad[j * weight.shape[1] + k] += x.data[i * x.shape[1] + k] * go;
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

void Linear::update(AdamW& opt);

// ---------- LayerNorm ----------
LayerNorm::LayerNorm(size_t normalized_shape_)
    : normalized_shape(normalized_shape_) {
    gamma = Tensor::ones({normalized_shape});
    beta = Tensor::zeros({normalized_shape});
}

Tensor LayerNorm::forward(const Tensor& x) {
    if (x.ndim() != 2) throw std::runtime_error("LayerNorm: only 2D supported");
    Tensor out(x.shape, false);
    size_t last_dim = x.shape[x.ndim() - 1];
    for (size_t i = 0; i < x.shape[0]; ++i) {
        float mean = 0.0f;
        for (size_t j = 0; j < last_dim; ++j) mean += x.data[i * last_dim + j];
        mean /= static_cast<float>(last_dim);
        float var = 0.0f;
        for (size_t j = 0; j < last_dim; ++j) {
            float d = x.data[i * last_dim + j] - mean;
            var += d * d;
        }
        var /= static_cast<float>(last_dim);
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (size_t j = 0; j < last_dim; ++j) {
            float norm = (x.data[i * last_dim + j] - mean) * inv_std;
            out.data[i * last_dim + j] = norm * gamma.data[j] + beta.data[j];
        }
    }
    return out;
}

// ---------- Activations ----------
Tensor ReLU::forward(const Tensor& x) {
    Tensor out = x;
    for (auto& v : out.data) if (v < 0) v = 0.0f;
    return out;
}

Tensor GELU::forward(const Tensor& x) {
    // Approximate GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    Tensor out(x.shape, false);
    const float sqrt_2_over_pi = 0.7978845608f;
    for (size_t i = 0; i < x.numel(); ++i) {
        float v = x.data[i];
        float cdf = 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (v + 0.044715f * v * v * v)));
        out.data[i] = v * cdf;
    }
    return out;
}

// ---------- Losses ----------
float SoftmaxCrossEntropy::loss(const Tensor& logits, const std::vector<int>& targets) {
    if (logits.shape[0] != targets.size()) throw std::runtime_error("SCE: batch size mismatch");
    Tensor sm = logits.softmax(-1);
    float total_loss = 0.0f;
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] < 0 || targets[i] >= static_cast<int>(logits.shape[1])) continue;
        total_loss -= std::log(sm.data[i * logits.shape[1] + targets[i]] + 1e-12f);
    }
    return total_loss / static_cast<float>(targets.size());
}

Tensor SoftmaxCrossEntropy::backward(const Tensor& logits, const std::vector<int>& targets) {
    Tensor grad = logits.softmax(-1);
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] >= 0 && targets[i] < static_cast<int>(logits.shape[1])) {
            grad.data[i * logits.shape[1] + targets[i]] -= 1.0f;
        }
    }
    for (auto& v : grad.data) v /= static_cast<float>(targets.size());
    return grad;
}

float KLDivLoss::loss(const Tensor& log_probs, const Tensor& target_probs) {
    if (!shapes_equal(log_probs.shape, target_probs.shape)) throw std::runtime_error("KLDiv: shape mismatch");
    float loss = 0.0f;
    size_t batch = log_probs.shape[0];
    size_t vocab = log_probs.shape[1];
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < vocab; ++j) {
            size_t idx = i * vocab + j;
            loss -= target_probs.data[idx] * log_probs.data[idx];
        }
    }
    return loss / static_cast<float>(batch);
}

Tensor KLDivLoss::backward(const Tensor& log_probs, const Tensor& target_probs) {
    // grad = -target_probs / (log_probs_exp) simplified: grad = -target_probs (since input is log_probs)
    // Actually d/dx (-t * log(p)) = -t when x=log(p)
    Tensor grad(log_probs.shape, false);
    size_t n = log_probs.numel();
    for (size_t i = 0; i < n; ++i) {
        grad.data[i] = -target_probs.data[i] / static_cast<float>(log_probs.shape[0]);
    }
    return grad;
}
