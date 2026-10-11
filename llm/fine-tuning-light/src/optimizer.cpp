#include "optimizer.hpp"

AdamW::AdamW(float lr_, float beta1_, float beta2_, float eps_, float weight_decay_)
    : lr(lr_), beta1(beta1_), beta2(beta2_), eps(eps_), weight_decay(weight_decay_), t(0) {}

void AdamW::add_param(Tensor* tensor) {
    params.push_back(tensor);
    m.emplace_back(tensor->numel(), 0.0f);
    v.emplace_back(tensor->numel(), 0.0f);
}

void AdamW::step() {
    t++;
    float bias1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    float bias2 = 1.0f - std::pow(beta2, static_cast<float>(t));
    float step_lr = lr * std::sqrt(bias2) / bias1;

    for (size_t p = 0; p < params.size(); ++p) {
        Tensor* tensor = params[p];
        if (!tensor->requires_grad) continue;
        size_t n = tensor->numel();
        for (size_t i = 0; i < n; ++i) {
            float g = tensor->grad[i];
            // Decoupled weight decay (AdamW)
            if (weight_decay != 0.0f) {
                tensor->data[i] *= (1.0f - lr * weight_decay);
            }
            m[p][i] = beta1 * m[p][i] + (1.0f - beta1) * g;
            v[p][i] = beta2 * v[p][i] + (1.0f - beta2) * g * g;
            tensor->data[i] -= step_lr * m[p][i] / (std::sqrt(v[p][i]) + eps);
        }
    }
}

void AdamW::zero_grad() {
    for (auto* p : params) {
        p->zero_grad();
    }
}
