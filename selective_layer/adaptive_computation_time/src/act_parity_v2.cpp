// ACT Parity: 改善案1+2 実装版（C++ / libtorch）
// 改善案1: 入力再提示 + 処理済みソフトマスク
// 改案2: 不確実性(予測変動)をhalt入力に追加
//
// ビルド例（libtorch がインストール済みの場合）:
//   mkdir build && cd build
//   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
//   cmake --build .
//
// 依存: libtorch (PyTorch C++), C++17

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ---------------- 設定 ----------------
const int D = 16;
const int H = 256;
const int M = 32;
const float EPS = 0.01f;
const int BS = 128;
const float LR = 1e-3f;
const float CLIP = 5.0f;
const int MAX_ITERS = 30000;
const float TAU = 1e-3f;
const int TAU_WARMUP = 2000;
const float TAU_START_K_RATIO = 0.5f;
const float HALT_BIAS = -1.5f;
const float CURRICULUM_THRESH = 0.93f;
const int FORCE_ADVANCE_AFTER = 8;
const int SEED = 0;

// デバイス設定 (CUDA が使えれば CUDA、なければ CPU)
torch::Device dev(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

// ---------------- データ生成 ----------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> make_batch(
    int B, int k_max = D, bool k_exact = false) {

    torch::Tensor k;
    if (k_exact) {
        k = torch::full({B, 1}, k_max, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    } else {
        k = torch::randint(1, k_max + 1, {B, 1},
                           torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    }

    // ranks: rand(B,D) -> argsort -> argsort
    auto tmp = torch::rand({B, D}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    auto ranks = tmp.argsort(1).argsort(1);
    auto mask = (ranks < k).to(torch::kFloat32);

    auto sign = (torch::randint(0, 2, {B, D},
                                torch::TensorOptions().dtype(torch::kFloat32).device(dev)) * 2 - 1);

    auto x = mask * sign;
    auto y = ((x == 1.0f).sum(1) % 2).to(torch::kFloat32);

    return {x, y, mask.sum(1)};
}

// ---------------- モデル ----------------
struct ModelImpl : torch::nn::Module {
    // レイヤー
    torch::nn::GRUCell cell{nullptr};
    torch::nn::Linear attention{nullptr};
    torch::nn::Linear out{nullptr};
    torch::nn::Linear halt{nullptr};

    c10::optional<int> fixed_steps;

    ModelImpl(c10::optional<int> fixed_steps_ = c10::nullopt) : fixed_steps(fixed_steps_) {
        // 入力: x(D) + processed(D) + flag(1) + step_feat(1) = 2D+2
        cell = register_module("cell", torch::nn::GRUCell(2 * D + 2, H));
        attention = register_module("attention", torch::nn::Linear(H, D));
        out = register_module("out", torch::nn::Linear(H, 1));
        // halt: 隠れ状態 + 不確実性(1) + 未処理割合(1) = H+2
        halt = register_module("halt", torch::nn::Linear(H + 2, 1));

        // halt bias の初期化
        torch::NoGradGuard no_grad;
        halt->bias.data().fill_(HALT_BIAS);
    }

    // 1ステップのGRU更新と処理済みマスク更新
    std::tuple<torch::Tensor, torch::Tensor> step(
        const torch::Tensor& x,
        const torch::Tensor& s,
        int n,
        const torch::Tensor& processed) {

        int64_t B = x.size(0);
        auto options = x.options();

        auto step_feat = torch::full({B, 1}, static_cast<float>(n) / M, options);
        auto flag = torch::full({B, 1}, (n == 0) ? 1.0f : 0.0f, options);

        // 改善案1: xを毎ステップ再提示 + processedも入力
        auto inp = torch::cat({x, processed, flag, step_feat}, 1);
        auto new_s = cell->forward(inp, s);

        // ソフト的に「次に見る次元」を選び、processedを更新
        auto attn = torch::sigmoid(attention->forward(new_s));
        auto new_processed = torch::clamp(processed + attn * (1 - processed), 0, 1);

        return {new_s, new_processed};
    }

    // forward: (y, ponder) を返す
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        int64_t B = x.size(0);
        auto options = x.options();

        auto s = torch::zeros({B, H}, options);
        auto processed = torch::zeros({B, D}, options);

        // 固定ステップ版
        if (fixed_steps.has_value()) {
            int N = fixed_steps.value();
            for (int n = 0; n < N; ++n) {
                auto ret = step(x, s, n, processed);
                s = std::get<0>(ret);
                processed = std::get<1>(ret);
            }
            auto y = torch::sigmoid(out->forward(s)).squeeze(-1);
            auto rho = torch::full({B}, static_cast<float>(N), options);
            return {y, rho};
        }

        // ACT 版
        auto halt_sum = torch::zeros({B}, options);
        auto R = torch::zeros({B}, options);
        auto N_steps = torch::zeros({B}, options);
        auto y_acc = torch::zeros({B}, options);

        torch::Tensor y_prev = torch::Tensor();

        for (int n = 0; n < M; ++n) {
            auto ret = step(x, s, n, processed);
            s = std::get<0>(ret);
            processed = std::get<1>(ret);

            auto y = torch::sigmoid(out->forward(s)).squeeze(-1);

            // 改善案2: 不確実性
            torch::Tensor uncertainty;
            if (!y_prev.defined()) {
                uncertainty = torch::zeros({B}, options);
            } else {
                uncertainty = (y - y_prev).abs();
            }
            y_prev = y.clone();

            // 改善案1: 未処理次元の割合
            auto remaining_ratio = 1.0f - processed.mean(1);

            // halt機構への入力
            auto halt_inp = torch::cat({s, uncertainty.unsqueeze(1), remaining_ratio.unsqueeze(1)}, 1);
            auto h = torch::sigmoid(halt->forward(halt_inp)).squeeze(-1);

            auto still = halt_sum < (1.0f - EPS);
            if (!still.any().item<bool>()) {
                break;
            }

            auto h_eff = torch::where(still, h, torch::zeros_like(h));
            auto will_halt = still & ((halt_sum + h_eff >= (1.0f - EPS)) |
                                      torch::full_like(halt_sum, (n == M - 1) ? 1.0f : 0.0f).to(torch::kBool));

            auto p = torch::where(will_halt, 1.0f - halt_sum, h_eff);
            R = R + torch::where(will_halt, 1.0f - halt_sum, torch::zeros_like(R));
            N_steps = N_steps + still.to(options.dtype());
            y_acc = y_acc + p * y;
            halt_sum = halt_sum + p;
        }

        return {y_acc, N_steps + R};
    }
};
TORCH_MODULE(Model);

// ---------------- 評価 ----------------
std::tuple<float, float, float> validate(Model& model, int k_max, bool k_exact, int B = 1024) {
    model->eval();
    torch::NoGradGuard no_grad;

    auto batch = make_batch(B, k_max, k_exact);
    auto x = std::get<0>(batch);
    auto t = std::get<1>(batch);

    auto ret = model->forward(x);
    auto y = std::get<0>(ret);
    auto rho = std::get<1>(ret);

    model->train();

    auto acc = ((y > 0.5f).to(torch::kFloat32) == t).to(torch::kFloat32).mean().item<float>();
    auto ponder_mean = rho.mean().item<float>();
    auto hitM = (rho >= (M - 0.5f)).to(torch::kFloat32).mean().item<float>();

    return {acc, ponder_mean, hitM};
}

void report_by_k(Model& model, int B = 2048) {
    model->eval();
    torch::NoGradGuard no_grad;

    auto batch = make_batch(B, D, false);
    auto x = std::get<0>(batch);
    auto t = std::get<1>(batch);
    auto k = std::get<2>(batch);

    auto ret = model->forward(x);
    auto y = std::get<0>(ret);
    auto rho = std::get<1>(ret);

    model->train();

    auto c = ((y > 0.5f).to(torch::kFloat32) == t).to(torch::kFloat32);
    auto acc_mean = c.mean().item<float>();
    auto ponder_mean = rho.mean().item<float>();
    auto hitM = (rho >= (M - 0.5f)).to(torch::kFloat32).mean().item<float>();

    std::cout << "\n[最終評価] acc=" << std::fixed << std::setprecision(3) << acc_mean
              << "  ponder=" << std::setprecision(2) << ponder_mean
              << "  hit-M=" << std::setprecision(3) << hitM << std::endl;

    std::vector<float> edges = {0.0f, 4.0f, 8.0f, 12.0f, 16.0f};
    for (size_t i = 0; i + 1 < edges.size(); ++i) {
        float lo = edges[i];
        float hi = edges[i + 1];
        auto m = (k > lo) & (k <= hi);
        if (m.any().item<bool>()) {
            auto acc_seg = c.index({m}).mean().item<float>();
            auto pond_seg = rho.index({m}).mean().item<float>();
            std::cout << "  k in (" << std::setw(4) << std::setprecision(1) << lo << ","
                      << std::setw(4) << hi << "]  acc=" << std::setprecision(3) << acc_seg
                      << "  ponder=" << std::setprecision(2) << pond_seg << std::endl;
        }
    }
}

// ---------------- 学習 ----------------
void train_model(Model& model, float tau_target, const std::string& name, bool curriculum = true) {
    torch::optim::AdamOptions opt_opts(LR);
    // C++ API ではパラメータグループの設定がやや複雑なので、デフォルトのまま使用
    torch::optim::Adam optimizer(model->parameters(), opt_opts);

    int k_max = curriculum ? 2 : D;
    c10::optional<int> tau_start = c10::nullopt;
    int stuck_counter = 0;
    int last_k_max = k_max;

    auto t0 = std::chrono::steady_clock::now();

    for (int it = 0; it < MAX_ITERS; ++it) {
        auto batch = make_batch(BS, k_max, false);
        auto x = std::get<0>(batch);
        auto t = std::get<1>(batch);

        auto ret = model->forward(x);
        auto y = std::get<0>(ret);
        auto rho = std::get<1>(ret);

        auto y_clamp = y.clamp(1e-6f, 1.0f - 1e-6f);

        float tau = 0.0f;
        if (k_max >= D * TAU_START_K_RATIO) {
            if (!tau_start.has_value()) {
                tau_start = it;
            }
            int dt = it - tau_start.value();
            tau = tau_target * std::min(1.0f, static_cast<float>(dt) / TAU_WARMUP);
        }

        auto loss = torch::binary_cross_entropy(y_clamp, t) + tau * rho.mean();

        optimizer.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), CLIP);
        optimizer.step();

        if (it % 200 == 0) {
            auto [va, vr, vh] = validate(model, k_max, true, 1024);

            if (k_max == last_k_max) {
                stuck_counter++;
            } else {
                stuck_counter = 0;
                last_k_max = k_max;
            }

            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0).count();

            std::cout << "[" << name << "] it=" << std::setw(5) << it
                      << " k_max=" << std::setw(2) << k_max
                      << " tau=" << std::scientific << std::setprecision(1) << tau
                      << " loss=" << std::fixed << std::setprecision(3) << loss.item<float>()
                      << " val_acc(k=" << k_max << ")=" << std::setprecision(3) << va
                      << " ponder=" << std::setprecision(1) << vr
                      << " hit-M=" << std::setprecision(2) << vh
                      << " stuck=" << stuck_counter
                      << " (" << elapsed << "s)" << std::endl;

            if (curriculum && k_max < D) {
                if (va > CURRICULUM_THRESH || stuck_counter >= FORCE_ADVANCE_AFTER) {
                    k_max++;
                    stuck_counter = 0;
                    std::cout << "[" << name << "] >>> k_max advanced to " << k_max
                              << " (va=" << std::fixed << std::setprecision(3) << va << ")" << std::endl;
                }
            }

            if (k_max >= D && tau_start.has_value() && (it - tau_start.value()) > (TAU_WARMUP + 2000)) {
                auto [full_acc, _, __] = validate(model, D, false, 1024);
                if (full_acc > 0.95f) {
                    std::cout << "[" << name << "] 収束と判断して終了 (acc="
                              << std::fixed << std::setprecision(3) << full_acc << ")" << std::endl;
                    return;
                }
            }
        }
    }
}

// ---------------- 実行 ----------------
int main() {
    std::cout << "Device: " << (dev.is_cuda() ? "CUDA" : "CPU") << std::endl;
    torch::manual_seed(SEED);

    // Sanity Check
    std::cout << "=== Sanity: 固定ステップ(N=8)で学習できるか ===" << std::endl;
    Model sanity(8);
    sanity->to(dev);
    train_model(sanity, 0.0f, "Fixed N=8", true);
    report_by_k(sanity, 2048);
    std::cout << "→ これが学習できない場合、ACT ではなくタスク設定/最適化の問題です。\n" << std::endl;

    // ACT
    std::cout << "=== ACT (改善案1+2) ===" << std::endl;
    Model act;
    act->to(dev);
    train_model(act, TAU, std::string("ACT tau=") + std::to_string(TAU), true);
    report_by_k(act, 2048);

    return 0;
}
