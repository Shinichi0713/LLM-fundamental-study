#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <iostream>

struct Tensor {
    std::vector<float> data;
    std::vector<size_t> shape;
    std::vector<float> grad;
    bool requires_grad = true;

    Tensor() = default;
    Tensor(const std::vector<size_t>& shape_, bool req_grad = true);
    Tensor(const std::vector<size_t>& shape_, float init_val, bool req_grad = true);

    size_t numel() const { return data.size(); }
    size_t ndim() const { return shape.size(); }
    size_t stride(size_t dim) const;
    void zero_grad();
    void reshape(const std::vector<size_t>& new_shape);
    Tensor view(const std::vector<size_t>& new_shape) const;

    // Accessors
    float& operator[](size_t i) { return data[i]; }
    const float& operator[](size_t i) const { return data[i]; }
    float& at(const std::vector<size_t>& idx);
    const float& at(const std::vector<size_t>& idx) const;

    // In-place ops
    void fill(float val);
    void randn(float mean = 0.0f, float std = 1.0f, uint64_t seed = 42);
    void xavier_uniform(size_t fan_in, size_t fan_out, uint64_t seed = 42);

    // Element-wise ops (return new tensor)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const; // element-wise
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;
    Tensor operator-() const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator+=(float scalar);
    Tensor& operator*=(float scalar);

    // Math ops
    Tensor sum() const;
    Tensor mean() const;
    Tensor softmax(int dim = -1) const;
    Tensor log_softmax(int dim = -1) const;
    Tensor transpose(size_t dim0, size_t dim1) const;

    // Static factory
    static Tensor zeros(const std::vector<size_t>& shape, bool req_grad = true);
    static Tensor ones(const std::vector<size_t>& shape, bool req_grad = true);
    static Tensor randn(const std::vector<size_t>& shape, float mean = 0.0f, float std = 1.0f, uint64_t seed = 42);
};

// Matrix multiplication: (M,K) x (K,N) -> (M,N)
Tensor matmul(const Tensor& a, const Tensor& b);

// Backward helpers
void add_backward(Tensor& a, Tensor& b, const Tensor& grad_out);
void matmul_backward(const Tensor& a, const Tensor& b, Tensor& grad_a, Tensor& grad_b, const Tensor& grad_out);

// Utility
std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b);
bool shapes_equal(const std::vector<size_t>& a, const std::vector<size_t>& b);
