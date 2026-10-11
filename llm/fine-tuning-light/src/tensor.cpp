#include "tensor.hpp"

// ---------- Constructors ----------
Tensor::Tensor(const std::vector<size_t>& shape_, bool req_grad)
    : shape(shape_), requires_grad(req_grad) {
    size_t n = 1;
    for (auto s : shape) n *= s;
    data.resize(n, 0.0f);
    if (requires_grad) grad.resize(n, 0.0f);
}

Tensor::Tensor(const std::vector<size_t>& shape_, float init_val, bool req_grad)
    : shape(shape_), requires_grad(req_grad) {
    size_t n = 1;
    for (auto s : shape_) n *= s;
    data.resize(n, init_val);
    if (requires_grad) grad.resize(n, 0.0f);
}

// ---------- Shape & Indexing ----------
size_t Tensor::stride(size_t dim) const {
    size_t s = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) s *= shape[i];
    return s;
}

void Tensor::zero_grad() {
    if (!grad.empty()) std::fill(grad.begin(), grad.end(), 0.0f);
}

void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t n = 1;
    for (auto s : new_shape) n *= s;
    if (n != numel()) throw std::runtime_error("reshape: size mismatch");
    shape = new_shape;
}

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    size_t n = 1;
    for (auto s : new_shape) n *= s;
    if (n != numel()) throw std::runtime_error("view: size mismatch");
    Tensor out = *this;
    out.shape = new_shape;
    return out;
}

float& Tensor::at(const std::vector<size_t>& idx) {
    if (idx.size() != shape.size()) throw std::runtime_error("at: dimension mismatch");
    size_t offset = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] >= shape[i]) throw std::runtime_error("at: index out of bounds");
        offset += idx[i] * stride(i);
    }
    return data[offset];
}

const float& Tensor::at(const std::vector<size_t>& idx) const {
    if (idx.size() != shape.size()) throw std::runtime_error("at: dimension mismatch");
    size_t offset = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] >= shape[i]) throw std::runtime_error("at: index out of bounds");
        offset += idx[i] * stride(i);
    }
    return data[offset];
}

// ---------- Initialization ----------
void Tensor::fill(float val) { std::fill(data.begin(), data.end(), val); }

void Tensor::randn(float mean, float std, uint64_t seed) {
    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::normal_distribution<float> dist(mean, std);
    for (auto& v : data) v = dist(gen);
}

void Tensor::xavier_uniform(size_t fan_in, size_t fan_out, uint64_t seed) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto& v : data) v = dist(gen);
}

Tensor Tensor::zeros(const std::vector<size_t>& shape, bool req_grad) {
    return Tensor(shape, 0.0f, req_grad);
}
Tensor Tensor::ones(const std::vector<size_t>& shape, bool req_grad) {
    return Tensor(shape, 1.0f, req_grad);
}
Tensor Tensor::randn(const std::vector<size_t>& shape, float mean, float std, uint64_t seed) {
    Tensor t(shape, false);
    t.randn(mean, std, seed);
    t.requires_grad = true;
    return t;
}

// ---------- Element-wise ops ----------
Tensor Tensor::operator+(const Tensor& other) const {
    if (!shapes_equal(shape, other.shape)) throw std::runtime_error("+: shape mismatch");
    Tensor out(shape, false);
    for (size_t i = 0; i < numel(); ++i) out.data[i] = data[i] + other.data[i];
    return out;
}
Tensor Tensor::operator-(const Tensor& other) const {
    if (!shapes_equal(shape, other.shape)) throw std::runtime_error("-: shape mismatch");
    Tensor out(shape, false);
    for (size_t i = 0; i < numel(); ++i) out.data[i] = data[i] - other.data[i];
    return out;
}
Tensor Tensor::operator*(const Tensor& other) const {
    if (!shapes_equal(shape, other.shape)) throw std::runtime_error("*: shape mismatch");
    Tensor out(shape, false);
    for (size_t i = 0; i < numel(); ++i) out.data[i] = data[i] * other.data[i];
    return out;
}
Tensor Tensor::operator*(float scalar) const {
    Tensor out = *this;
    for (auto& v : out.data) v *= scalar;
    return out;
}
Tensor Tensor::operator/(float scalar) const {
    Tensor out = *this;
    for (auto& v : out.data) v /= scalar;
    return out;
}
Tensor Tensor::operator-() const {
    Tensor out = *this;
    for (auto& v : out.data) v = -v;
    return out;
}
Tensor& Tensor::operator+=(const Tensor& other) {
    if (!shapes_equal(shape, other.shape)) throw std::runtime_error("+=: shape mismatch");
    for (size_t i = 0; i < numel(); ++i) data[i] += other.data[i];
    return *this;
}
Tensor& Tensor::operator+=(float scalar) {
    for (auto& v : data) v += scalar;
    return *this;
}
Tensor& Tensor::operator*=(float scalar) {
    for (auto& v : data) v *= scalar;
    return *this;
}

// ---------- Reduction ops ----------
Tensor Tensor::sum() const {
    Tensor out({1}, false);
    out.data[0] = std::accumulate(data.begin(), data.end(), 0.0f);
    return out;
}
Tensor Tensor::mean() const {
    Tensor out = sum();
    out.data[0] /= static_cast<float>(numel());
    return out;
}

// Softmax over last dimension
Tensor Tensor::softmax(int dim) const {
    size_t d = (dim < 0) ? shape.size() - 1 : static_cast<size_t>(dim);
    if (d >= shape.size()) throw std::runtime_error("softmax: invalid dim");

    size_t outer = 1;
    for (size_t i = 0; i < d; ++i) outer *= shape[i];
    size_t axis_size = shape[d];
    size_t inner = 1;
    for (size_t i = d + 1; i < shape.size(); ++i) inner *= shape[i];

    Tensor out(shape, false);
    for (size_t o = 0; o < outer; ++o) {
        for (size_t inn = 0; inn < inner; ++inn) {
            float max_val = -1e30f;
            for (size_t j = 0; j < axis_size; ++j) {
                size_t idx = ((o * axis_size) + j) * inner + inn;
                if (data[idx] > max_val) max_val = data[idx];
            }
            float sum_exp = 0.0f;
            for (size_t j = 0; j < axis_size; ++j) {
                size_t idx = ((o * axis_size) + j) * inner + inn;
                out.data[idx] = std::exp(data[idx] - max_val);
                sum_exp += out.data[idx];
            }
            for (size_t j = 0; j < axis_size; ++j) {
                size_t idx = ((o * axis_size) + j) * inner + inn;
                out.data[idx] /= sum_exp;
            }
        }
    }
    return out;
}

Tensor Tensor::log_softmax(int dim) const {
    Tensor sm = softmax(dim);
    Tensor out(shape, false);
    for (size_t i = 0; i < numel(); ++i) out.data[i] = std::log(sm.data[i] + 1e-12f);
    return out;
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    if (dim0 >= shape.size() || dim1 >= shape.size()) throw std::runtime_error("transpose: invalid dims");
    std::vector<size_t> new_shape = shape;
    std::swap(new_shape[dim0], new_shape[dim1]);
    Tensor out(new_shape, false);
    // Generic transpose for any dimensions
    std::vector<size_t> old_strides(ndim()), new_strides(ndim());
    for (size_t i = 0; i < ndim(); ++i) old_strides[i] = stride(i);
    size_t n = numel();
    for (size_t flat = 0; flat < n; ++flat) {
        // decode flat index to coordinates
        std::vector<size_t> coords(ndim());
        size_t rem = flat;
        for (size_t i = 0; i < ndim(); ++i) {
            coords[i] = rem / old_strides[i];
            rem %= old_strides[i];
        }
        std::swap(coords[dim0], coords[dim1]);
        size_t new_flat = 0;
        for (size_t i = 0; i < ndim(); ++i) {
            size_t ns = 1;
            for (size_t j = i + 1; j < new_shape.size(); ++j) ns *= new_shape[j];
            new_flat += coords[i] * ns;
        }
        out.data[new_flat] = data[flat];
    }
    return out;
}

// ---------- Matrix multiplication ----------
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) throw std::runtime_error("matmul: only 2D tensors supported");
    size_t M = a.shape[0];
    size_t K = a.shape[1];
    if (b.shape[0] != K) throw std::runtime_error("matmul: inner dimension mismatch");
    size_t N = b.shape[1];
    Tensor out({M, N}, false);
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float aik = a.data[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                out.data[i * N + j] += aik * b.data[k * N + j];
            }
        }
    }
    return out;
}

void matmul_backward(const Tensor& a, const Tensor& b, Tensor& grad_a, Tensor& grad_b, const Tensor& grad_out) {
    size_t M = a.shape[0], K = a.shape[1], N = b.shape[1];
    // grad_a = grad_out @ b^T
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float sum = 0.0f;
            for (size_t j = 0; j < N; ++j) sum += grad_out.data[i * N + j] * b.data[k * N + j];
            grad_a.grad[i * K + k] += sum;
        }
    }
    // grad_b = a^T @ grad_out
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < M; ++i) sum += a.data[i * K + k] * grad_out.data[i * N + j];
            grad_b.grad[k * N + j] += sum;
        }
    }
}

void add_backward(Tensor& a, Tensor& b, const Tensor& grad_out) {
    if (a.requires_grad) {
        for (size_t i = 0; i < a.numel(); ++i) a.grad[i] += grad_out.data[i];
    }
    if (b.requires_grad) {
        for (size_t i = 0; i < b.numel(); ++i) b.grad[i] += grad_out.data[i];
    }
}

// ---------- Shape utilities ----------
bool shapes_equal(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t nd = std::max(a.size(), b.size());
    std::vector<size_t> out(nd, 1);
    size_t off_a = nd - a.size();
    size_t off_b = nd - b.size();
    for (size_t i = 0; i < nd; ++i) {
        size_t sa = (i >= off_a) ? a[i - off_a] : 1;
        size_t sb = (i >= off_b) ? b[i - off_b] : 1;
        if (sa != sb && sa != 1 && sb != 1) throw std::runtime_error("broadcast: incompatible shapes");
        out[i] = std::max(sa, sb);
    }
    return out;
}
