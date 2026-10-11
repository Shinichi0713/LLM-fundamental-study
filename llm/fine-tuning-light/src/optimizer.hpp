#pragma once
#include "tensor.hpp"
#include <vector>
#include <cmath>

struct AdamW {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int t;

    std::vector<std::vector<float>> m;
    std::vector<std::vector<float>> v;

    AdamW(float lr_ = 1e-3f, float beta1_ = 0.9f, float beta2_ = 0.999f,
          float eps_ = 1e-8f, float weight_decay_ = 0.01f);

    void add_param(Tensor* tensor);
    void step();
    void zero_grad();

private:
    std::vector<Tensor*> params;
};
