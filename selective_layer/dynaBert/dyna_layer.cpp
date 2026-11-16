#include <torch/torch.h>
#include <torch/nn/functional/loss.h>
#include <torch/data/transforms/tensor_transform.h>

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <fstream>
#include <random>
#include <algorithm>
#include <numeric>

// ------------------------------------------------------------
// Config
// ------------------------------------------------------------
struct Config {
    int epochs = 30;
    int pretrain_epochs = 10;
    double eps_start = 0.3;
    double eps_decay_epochs = 5.0;
    double eps_min = 0.05;
    int batch = 128;
    double lr = 1e-3;
    double wd = 1e-2;
    double agent_lr = 1e-3;
    int holdout = 5000;
    int max_iter = 10;
    double tol = 1e-2;
    double fpi_scale = 0.5;
    double lip_coeff = 1.0;
    int pi_init = 30;
    int pi_train = 2;
    int T_steps = 3;
    double aux_w = 0.3;
    double ent_coef = 0.01;
    int untied_k = 4;
    int eval_every = 2;
    int N_ACT = 4;
    std::vector<int64_t> CH = {64, 128, 256};
    std::vector<int64_t> HW = {16, 8, 4};
} CFG;

// ------------------------------------------------------------
// CIFAR-10 binary dataset
// ------------------------------------------------------------
class Cifar10Dataset : public torch::data::datasets::Dataset<Cifar10Dataset> {
public:
    Cifar10Dataset(const std::string& img_file,
                   const std::string& lbl_file,
                   const std::string& meta_file) {
        std::ifstream m(meta_file, std::ios::binary);
        uint32_t N = 0, C = 0;
        m.read(reinterpret_cast<char*>(&N), sizeof(N));
        m.read(reinterpret_cast<char*>(&C), sizeof(C));
        images_ = torch::from_file(img_file, /*shared=*/false,
                                   {static_cast<int64_t>(N), static_cast<int64_t>(C), 32, 32},
                                   torch::kFloat32);
        labels_ = torch::from_file(lbl_file, /*shared=*/false,
                                   {static_cast<int64_t>(N)}, torch::kInt64);
    }

    torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }

    torch::optional<size_t> size() const override {
        return images_.size(0);
    }

    torch::Tensor images() const { return images_; }
    torch::Tensor labels() const { return labels_; }

private:
    torch::Tensor images_, labels_;
};

// ------------------------------------------------------------
// Subset
// ------------------------------------------------------------
class SubsetDataset : public torch::data::datasets::Dataset<SubsetDataset> {
public:
    SubsetDataset(Cifar10Dataset base, std::vector<int64_t> indices)
        : base_(std::move(base)), indices_(std::move(indices)) {}

    torch::data::Example<> get(size_t index) override {
        return base_.get(indices_[index]);
    }

    torch::optional<size_t> size() const override {
        return indices_.size();
    }

private:
    Cifar10Dataset base_;
    std::vector<int64_t> indices_;
};

// ------------------------------------------------------------
// Utils
// ------------------------------------------------------------
torch::nn::Sequential conv_gn(int64_t in, int64_t out) {
    using namespace torch::nn;
    return Sequential(
        Conv2d(Conv2dOptions(in, out, 3).padding(1).bias(false)),
        GroupNorm(GroupNormOptions(8, out)),
        ReLU(ReLUOptions().inplace(true))
    );
}

torch::nn::Sequential make_stage(int64_t in, int64_t out) {
    using namespace torch::nn;
    return Sequential(
        conv_gn(in, out),
        conv_gn(out, out),
        MaxPool2d(MaxPool2dOptions(2))
    );
}

struct MacTable {
    std::vector<double> stage;
    std::vector<double> refine;
    double head = 0;
};

MacTable mac_table() {
    std::vector<int64_t> ch = CFG.CH;
    std::vector<int64_t> res = {32, 16, 8};
    MacTable mt;
    std::vector<int64_t> ins = {3, ch[0], ch[1]};
    for (int i = 0; i < 3; ++i) {
        double s = ins[i] * ch[i] * 9 * res[i] * res[i]
                 + ch[i] * ch[i] * 9 * res[i] * res[i];
        mt.stage.push_back(s);
        double r = 2 * ch[i] * ch[i] * 9 * (res[i] / 2) * (res[i] / 2);
        mt.refine.push_back(r);
    }
    mt.head = ch[2] * 10;
    return mt;
}

torch::Tensor one_hot(const torch::Tensor& idx, int64_t n) {
    auto y = torch::zeros({idx.size(0), n},
                          torch::TensorOptions().dtype(torch::kFloat32).device(idx.device()));
    y.scatter_(1, idx.unsqueeze(1).to(torch::kLong), 1.0f);
    return y;
}

// ------------------------------------------------------------
// LipConv
// ------------------------------------------------------------
class LipConvImpl : public torch::nn::Module {
public:
    LipConvImpl(int64_t c, int64_t hw, double coeff) : coeff_(coeff) {
        weight_ = register_parameter("weight", torch::empty({c, c, 3, 3}));
        torch::nn::init::kaiming_uniform_(weight_, std::sqrt(5.0));
        bias_ = register_parameter("bias", torch::zeros({c}));
        auto u = torch::randn({1, c, hw, hw});
        u_ = register_buffer("u", u / (u.norm() + 1e-12));
        power_iter(CFG.pi_init);
    }

    void power_iter(int n) {
        torch::NoGradGuard guard;
        auto W = weight_.to(torch::kFloat32);
        auto u = u_;
        for (int i = 0; i < n; ++i) {
            auto v = torch::conv2d(u, W, {}, /*stride=*/1, /*padding=*/1);
            v = v / (v.norm() + 1e-12);
            u = torch::conv_transpose2d(v, W, {}, /*stride=*/1, /*padding=*/1);
            u = u / (u.norm() + 1e-12);
        }
        u_.copy_(u);
    }

    std::pair<torch::Tensor, torch::Tensor> get_w() {
        if (is_training()) power_iter(CFG.pi_train);
        auto W = weight_.to(torch::kFloat32);
        auto sigma = torch::conv2d(u_, W, {}, 1, 1).norm();
        auto scale = torch::clamp(sigma / coeff_, 1.0, 1e9);
        return {W / scale, bias_.to(torch::kFloat32)};
    }

    torch::Tensor weight_, bias_, u_;
    double coeff_;
};
TORCH_MODULE(LipConv);

// ------------------------------------------------------------
// FPIBlock
// ------------------------------------------------------------
class FPIBlockImpl : public torch::nn::Module {
public:
    FPIBlockImpl(int64_t c, int64_t hw, double scale) : scale_(scale) {
        c1_ = register_module("c1", LipConv(c, hw, CFG.lip_coeff));
        c2_ = register_module("c2", LipConv(c, hw, CFG.lip_coeff));
    }

    torch::Tensor forward(torch::Tensor z, torch::Tensor x,
                          c10::optional<std::pair<torch::Tensor, torch::Tensor>> w1 = c10::nullopt,
                          c10::optional<std::pair<torch::Tensor, torch::Tensor>> w2 = c10::nullopt) {
        auto [W1, b1] = w1 ? *w1 : c1_->get_w();
        auto [W2, b2] = w2 ? *w2 : c2_->get_w();
        auto h = torch::conv2d(
            torch::relu(torch::conv2d(z, W1, b1, /*stride=*/1, /*padding=*/1)),
            W2, b2, /*stride=*/1, /*padding=*/1);
        return x + scale_ * h;
    }

    LipConv c1_, c2_;
    double scale_;
};
TORCH_MODULE(FPIBlock);

// ------------------------------------------------------------
// AgentNet
// ------------------------------------------------------------
class AgentNetImpl : public torch::nn::Module {
public:
    AgentNetImpl(int64_t in_dim, int64_t hidden = 128) {
        ln_ = register_module("ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({in_dim})));
        fc1_ = register_module("fc1", torch::nn::Linear(in_dim, hidden));
        fc2_ = register_module("fc2", torch::nn::Linear(hidden, CFG.N_ACT));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = ln_->forward(x);
        x = torch::relu(fc1_->forward(x));
        x = fc2_->forward(x);
        return x;
    }

    torch::nn::LayerNorm ln_;
    torch::nn::Linear fc1_, fc2_;
};
TORCH_MODULE(AgentNet);

// ------------------------------------------------------------
// Episode output
// ------------------------------------------------------------
struct EpisodeOutput {
    torch::Tensor logits;
    torch::Tensor logits0;
    torch::Tensor macs;
    torch::Tensor iters;
    torch::Tensor logp;
    torch::Tensor penalty;
    torch::Tensor ent_sum;
    torch::Tensor n_alive;
    std::vector<torch::Tensor> acts;
};

// ------------------------------------------------------------
// DynaNet
// ------------------------------------------------------------
class DynaNetImpl : public torch::nn::Module {
public:
    DynaNetImpl(const std::string& kind, const std::string& policy = "none")
        : kind_(kind), policy_(policy) {
        stages_.push_back(register_module("stages_0", make_stage(3, CFG.CH[0])));
        stages_.push_back(register_module("stages_1", make_stage(CFG.CH[0], CFG.CH[1])));
        stages_.push_back(register_module("stages_2", make_stage(CFG.CH[1], CFG.CH[2])));
        head_ = register_module("head", torch::nn::Linear(CFG.CH[2], 10));

        auto mt = mac_table();
        macs_base_d_ = std::accumulate(mt.stage.begin(), mt.stage.end(), 0.0) + mt.head;
        macs_ref_ = mt.refine;
        macs_down_.resize(3);
        for (int i = 0; i < 3; ++i) {
            macs_down_[i] = std::accumulate(mt.stage.begin() + i + 1, mt.stage.end(), 0.0) + mt.head;
        }

        if (kind == "static" || kind == "dyna") {
            for (int i = 0; i < 3; ++i) {
                refine_.push_back(register_module(
                    "refine_" + std::to_string(i),
                    FPIBlock(CFG.CH[i], CFG.HW[i], CFG.fpi_scale)));
            }
        }

        if (kind == "untied") {
            for (int i = 0; i < 3; ++i) {
                refine_u_.emplace_back();
                for (int k = 0; k < CFG.untied_k; ++k) {
                    refine_u_[i].push_back(register_module(
                        "refine_u_" + std::to_string(i) + "_" + std::to_string(k),
                        FPIBlock(CFG.CH[i], CFG.HW[i], CFG.fpi_scale)));
                }
            }
        }

        if (kind == "dyna" && policy == "agent") {
            int64_t feat_dim = CFG.CH[0] + CFG.CH[1] + CFG.CH[2] + 2 + CFG.N_ACT + CFG.T_steps;
            agent_ = register_module("agent", AgentNet(feat_dim));
        }
    }

    torch::Tensor cls(torch::Tensor h) {
        return head_->forward(h.mean({2, 3}));
    }

    std::pair<torch::Tensor, torch::Tensor> fpi(int64_t li, torch::Tensor x) {
        auto f = refine_[li];
        auto [W1, b1] = f->c1_->get_w();
        auto [W2, b2] = f->c2_->get_w();
        torch::Tensor z_out, iters_t;
        {
            torch::NoGradGuard guard;
            auto xd = x.detach();
            auto z = xd;
            auto B = x.size(0);
            auto active = torch::ones({B}, torch::TensorOptions().dtype(torch::kBool).device(x.device()));
            auto iters = torch::zeros({B}, torch::TensorOptions().device(x.device()));
            for (int i = 0; i < CFG.max_iter; ++i) {
                auto fz = f->forward(z, xd,
                                     std::make_pair(W1, b1),
                                     std::make_pair(W2, b2));
                auto rel = (fz - z).flatten(1).norm(1) /
                           (fz.flatten(1).norm(1) + 1e-8);
                z = torch::where(active.view({-1, 1, 1, 1}), fz, z);
                iters = iters + active.to(iters.dtype());
                active = active & (rel >= CFG.tol);
                if (!active.any().item<bool>()) break;
            }
            z_out = z;
            iters_t = iters + 1.0;
        }
        z_out = f->forward(z_out, x,
                           std::make_pair(W1, b1),
                           std::make_pair(W2, b2));
        return {z_out, iters_t};
    }

    torch::Tensor agent_feats(const std::vector<torch::Tensor>& h,
                              torch::Tensor logits,
                              torch::Tensor prev, int t) {
        torch::NoGradGuard guard;
        auto B = logits.size(0);
        std::vector<torch::Tensor> pooled;
        for (const auto& s : h) {
            pooled.push_back(s.to(torch::kFloat32).mean({2, 3}));
        }
        auto p = torch::softmax(logits.to(torch::kFloat32), 1);
        auto conf = std::get<0>(p.max(1, true));
        auto ent = -(p * torch::log(p + 1e-8)).sum(1, true) /
                    std::log(static_cast<double>(p.size(1)));
        auto tt = one_hot(torch::full({B}, t,
                          torch::TensorOptions().dtype(torch::kLong).device(logits.device())),
                          CFG.T_steps);
        std::vector<torch::Tensor> cat_vec = pooled;
        cat_vec.push_back(conf);
        cat_vec.push_back(ent);
        cat_vec.push_back(one_hot(prev, CFG.N_ACT));
        cat_vec.push_back(tt);
        return torch::cat(cat_vec, 1);
    }

    EpisodeOutput run(torch::Tensor x, double eps, bool greedy) {
        auto B = x.size(0);
        auto dev = x.device();
        auto zero = torch::zeros({B}, torch::TensorOptions().device(dev));

        if (kind_ == "plain") {
            auto h = x;
            for (auto& s : stages_) h = s->forward(h);
            auto lg = cls(h);
            EpisodeOutput o;
            o.logits = lg; o.logits0 = lg; o.macs = zero + macs_base_d_;
            o.iters = zero; o.logp = zero; o.penalty = zero;
            return o;
        }

        if (kind_ == "static") {
            auto h = x;
            auto it_tot = zero.clone();
            auto macs = zero + macs_base_d_;
            for (int li = 0; li < 3; ++li) {
                h = stages_[li]->forward(h);
                auto [z, it] = fpi(li, h);
                h = z;
                it_tot = it_tot + it;
                macs = macs + it * macs_ref_[li];
            }
            auto lg = cls(h);
            EpisodeOutput o;
            o.logits = lg; o.logits0 = lg; o.macs = macs;
            o.iters = it_tot; o.logp = zero; o.penalty = zero;
            return o;
        }

        if (kind_ == "untied") {
            auto h = x;
            for (int li = 0; li < 3; ++li) {
                h = stages_[li]->forward(h);
                auto z = h;
                for (auto& blk : refine_u_[li]) z = blk->forward(z, h);
                h = z;
            }
            auto lg = cls(h);
            EpisodeOutput o;
            o.logits = lg; o.logits0 = lg;
            o.macs = zero + macs_base_d_ + CFG.untied_k *
                     std::accumulate(macs_ref_.begin(), macs_ref_.end(), 0.0);
            o.iters = zero + 3 * CFG.untied_k;
            return o;
        }

        // ---------- dyna episode ----------
        std::vector<torch::Tensor> h;
        auto hh = x;
        for (auto& s : stages_) {
            hh = s->forward(hh);
            h.push_back(hh);
        }
        auto logits = cls(h[2]);
        auto logits0 = logits;

        auto macs = torch::full({B}, macs_base_d_, torch::TensorOptions().device(dev));
        auto iters_tot = torch::zeros({B}, torch::TensorOptions().device(dev));
        auto penalty = torch::zeros({B}, torch::TensorOptions().device(dev));
        auto logp = torch::zeros({B}, torch::TensorOptions().device(dev));
        auto ent_sum = torch::zeros({}, torch::TensorOptions().device(dev));
        auto n_alive = torch::zeros({}, torch::TensorOptions().device(dev));

        auto alive = torch::ones({B}, torch::TensorOptions().dtype(torch::kBool).device(dev));
        auto prev = torch::zeros({B}, torch::TensorOptions().dtype(torch::kLong).device(dev));
        auto consec = torch::zeros({B}, torch::TensorOptions().device(dev));
        std::vector<torch::Tensor> acts;

        bool use_agent = !force_.has_value() && (policy_ == "agent");

        for (int t = 0; t < CFG.T_steps; ++t) {
            torch::Tensor a, lp, e;
            if (force_.has_value()) {
                a = torch::full({B}, t == 0 ? *force_ : 0,
                                torch::TensorOptions().dtype(torch::kLong).device(dev));
                lp = torch::zeros({B}, torch::TensorOptions().device(dev));
                e = torch::zeros({B}, torch::TensorOptions().device(dev));
            } else if (policy_ == "random") {
                a = torch::randint(0, CFG.N_ACT, {B}, torch::TensorOptions().device(dev));
                lp = torch::full({B}, std::log(1.0 / CFG.N_ACT),
                                 torch::TensorOptions().device(dev));
                e = torch::full({B}, std::log(static_cast<double>(CFG.N_ACT)),
                                 torch::TensorOptions().device(dev));
            } else {
                auto feats = agent_feats(h, logits, prev, t);
                auto logits_a = agent_->forward(feats).to(torch::kFloat32);
                auto probs = torch::softmax(logits_a, 1);
                auto mix = (1.0 - eps) * probs + eps / CFG.N_ACT;
                if (greedy) {
                    a = std::get<1>(probs.max(1));
                } else {
                    a = torch::multinomial(mix, 1, false).squeeze(1);
                }
                lp = torch::log(mix.gather(1, a.unsqueeze(1)).squeeze(1) + 1e-8);
                e = -(probs * torch::log(probs + 1e-8)).sum(1);
            }

            logp = logp + torch::where(alive, lp, torch::zeros_like(lp));
            ent_sum = ent_sum + (e * alive.to(e.dtype())).sum();
            n_alive = n_alive + alive.to(n_alive.dtype()).sum();

            a = torch::where(alive, a, torch::zeros_like(a));
            acts.push_back(torch::where(alive, a, torch::full_like(a, -1)).detach());
            alive = alive & (a != 0);
            auto same = (a == prev) & (a != 0);
            consec = torch::where(a == 0, torch::zeros_like(consec),
                                  torch::where(same, consec + 1, torch::ones_like(consec)));
            prev = a;

            auto step_cost = torch::zeros({B}, torch::TensorOptions().device(dev));
            for (int l = 1; l <= 3; ++l) {
                auto mask = (a == l);
                if (!mask.any().item<bool>()) continue;
                auto idx = mask.nonzero().squeeze(1);
                int li = l - 1;
                auto [z, it] = fpi(li, h[li].index_select(0, idx));
                auto hl = h[li].clone();
                hl.index_copy_(0, idx, z.to(hl.dtype()));
                h[li] = hl;

                hh = z;
                for (int m = li + 1; m < 3; ++m) {
                    hh = stages_[m]->forward(hh);
                    auto hn = h[m].clone();
                    hn.index_copy_(0, idx, hh.to(hn.dtype()));
                    h[m] = hn;
                }
                auto lg = logits.clone();
                lg.index_copy_(0, idx, cls(hh).to(lg.dtype()));
                logits = lg;

                step_cost.index_put_({idx}, it * macs_ref_[li] + macs_down_[li]);
                auto iters_idx = iters_tot.index_select(0, idx) + it;
                iters_tot.index_copy_(0, idx, iters_idx);
            }
            macs = macs + step_cost;
            penalty = penalty + step_cost / macs_base_d_ * consec;
        }

        EpisodeOutput o;
        o.logits = logits; o.logits0 = logits0; o.macs = macs;
        o.iters = iters_tot; o.logp = logp; o.penalty = penalty;
        o.ent_sum = ent_sum; o.n_alive = n_alive; o.acts = acts;
        return o;
    }

    void set_force(c10::optional<int64_t> f) { force_ = f; }

    std::string kind_, policy_;
    torch::nn::Linear head_;
    std::vector<torch::nn::Sequential> stages_;
    std::vector<FPIBlock> refine_;
    std::vector<std::vector<FPIBlock>> refine_u_;
    AgentNet agent_;
    double macs_base_d_ = 0;
    std::vector<double> macs_ref_, macs_down_;
    c10::optional<int64_t> force_;
};
TORCH_MODULE(DynaNet);

// ------------------------------------------------------------
// Evaluation
// ------------------------------------------------------------
struct EvalResult {
    double acc = 0;
    double macs = 0;
    double iters = 0;
    torch::Tensor hist;
};

template <typename Loader>
EvalResult evaluate(DynaNet& model, Loader& loader, torch::Device device) {
    model->eval();
    torch::NoGradGuard guard;
    int64_t N = 0, correct = 0;
    double macs_sum = 0, it_sum = 0;
    auto hist = torch::zeros({CFG.T_steps, CFG.N_ACT}, torch::kCPU);
    for (auto& batch : loader) {
        auto x = batch.data.to(device);
        auto y = batch.target.to(device).to(torch::kLong);
        auto out = model->run(x, 0.0, true);
        auto pred = out.logits.argmax(1);
        N += y.size(0);
        correct += (pred == y).sum().item<int64_t>();
        macs_sum += out.macs.sum().item<double>();
        it_sum += out.iters.sum().item<double>();
        if (!out.acts.empty()) {
            for (int t = 0; t < out.acts.size(); ++t) {
                for (int k = 0; k < CFG.N_ACT; ++k) {
                    hist[t][k] += (out.acts[t] == k).sum().item<int64_t>();
                }
            }
        }
    }
    EvalResult r;
    r.acc = 100.0 * correct / N;
    r.macs = macs_sum / N;
    r.iters = it_sum / N;
    r.hist = hist / static_cast<double>(N);
    return r;
}

// ------------------------------------------------------------
// Training
// ------------------------------------------------------------
struct MethodConfig {
    std::string kind;
    std::string policy = "none";
    std::string reward = "ce";
    double lam = 1.0;
    double alpha = 1.0;
    double beta = 0.05;
};

EvalResult train_run(const std::string& name,
                     const MethodConfig& mc,
                     int seed,
                     Cifar10Dataset& train_ds,
                     Cifar10Dataset& hold_ds,
                     Cifar10Dataset& test_ds,
                     torch::Device device) {
    torch::manual_seed(seed);
    if (device.is_cuda()) torch::cuda::manual_seed_all(seed);

    auto model = DynaNet(mc.kind, mc.policy);
    model->to(device);

    std::vector<torch::Tensor> main_params, agent_params;
    for (const auto& p : model->named_parameters()) {
        if (p.key().find("agent.") != std::string::npos)
            agent_params.push_back(p.value());
        else
            main_params.push_back(p.value());
    }

    auto opt = std::make_unique<torch::optim::AdamW>(
        main_params, torch::optim::AdamWOptions(CFG.lr).weight_decay(CFG.wd));

    std::unique_ptr<torch::optim::Adam> opt_a;
    if (!agent_params.empty()) {
        opt_a = std::make_unique<torch::optim::Adam>(
            agent_params, torch::optim::AdamOptions(CFG.agent_lr));
    }

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        train_ds.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(CFG.batch).workers(2));

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        test_ds.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(256).workers(2));

    // holdout tensors for fast random batching
    torch::Tensor hold_images, hold_labels;
    {
        std::vector<torch::Tensor> himgs, hlabels;
        auto hold_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            hold_ds.map(torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions().batch_size(128).workers(1));
        for (auto& batch : *hold_loader) {
            himgs.push_back(batch.data);
            hlabels.push_back(batch.target);
        }
        hold_images = torch::cat(himgs, 0);
        hold_labels = torch::cat(hlabels, 0);
    }

    for (int ep = 0; ep < CFG.epochs; ++ep) {
        model->train();
        double eps = 0.0;
        bool agent_phase = false;
        if (mc.policy == "random") {
            eps = 1.0;
        } else if (!agent_params.empty()) {
            if (ep < CFG.pretrain_epochs) {
                eps = 1.0;
            } else {
                eps = std::max(CFG.eps_min,
                    CFG.eps_start - (CFG.eps_start - CFG.eps_min) *
                    (ep - CFG.pretrain_epochs) / std::max(1.0, CFG.eps_decay_epochs));
                agent_phase = true;
            }
        }

        double loss_sum = 0;
        int n_b = 0;
        for (auto& batch : *train_loader) {
            auto x = batch.data.to(device);
            auto y = batch.target.to(device).to(torch::kLong);

            opt->zero_grad();
            auto out = model->run(x, eps, false);
            auto loss = torch::nn::functional::cross_entropy(
                out.logits.to(torch::kFloat32), y,
                torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
            if (mc.kind == "dyna") {
                loss = loss + CFG.aux_w * torch::nn::functional::cross_entropy(
                    out.logits0.to(torch::kFloat32), y,
                    torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
            }
            loss.backward();
            opt->step();

            if (agent_phase) {
                auto idx = torch::randperm(hold_images.size(0),
                                           torch::TensorOptions().device(device)).slice(0, 0, x.size(0));
                auto xh = hold_images.index_select(0, idx).to(device);
                auto yh = hold_labels.index_select(0, idx).to(device).to(torch::kLong);

                opt_a->zero_grad();
                auto oh = model->run(xh, eps, false);

                torch::Tensor R;
                {
                    torch::NoGradGuard guard;
                    auto lg = oh.logits.to(torch::kFloat32);
                    auto lg0 = oh.logits0.to(torch::kFloat32);
                    torch::Tensor acc_term;
                    if (mc.reward == "paper") {
                        acc_term = (lg.argmax(1) == yh).to(torch::kFloat32);
                    } else {
                        auto ce0 = torch::nn::functional::cross_entropy(
                            lg0, yh, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
                        auto ce1 = torch::nn::functional::cross_entropy(
                            lg, yh, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
                        acc_term = (ce0 - ce1).clamp(-2.0, 2.0);
                    }
                    R = mc.alpha * acc_term - mc.beta * mc.lam * oh.penalty;
                    auto adv = R - R.mean();
                    auto ent = oh.ent_sum / oh.n_alive.clamp(1.0);
                    auto agent_loss = -(adv * oh.logp).mean() - CFG.ent_coef * ent;
                    agent_loss.backward();
                }

                // main model gradients must not be updated by agent loss
                for (auto& p : main_params) {
                    if (p.grad().defined()) p.grad().zero_();
                }
                opt_a->step();
            }

            loss_sum += loss.item<double>();
            ++n_b;
        }

        // simple cosine annealing
        double lr_now = CFG.lr * 0.5 * (1.0 + std::cos(M_PI * ep / CFG.epochs));
        for (auto& group : opt->param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr_now);
        }

        if ((ep + 1) % CFG.eval_every == 0 || ep == CFG.epochs - 1) {
            auto res = evaluate(model, *test_loader, device);
            auto h0 = res.hist[0];
            std::cout << "[" << name << " s" << seed << "] ep" << (ep + 1) << "/"
                      << CFG.epochs << " loss=" << loss_sum / n_b
                      << " acc=" << res.acc << "% macs=" << res.macs / 1e6 << "M";
            if (mc.kind == "dyna") {
                std::cout << " | step1 NOP/L1/L2/L3=";
                for (int k = 0; k < CFG.N_ACT; ++k) {
                    std::cout << static_cast<int>(std::round(100 * h0[k].item<double>())) << "%" << (k < 3 ? "/" : "");
                }
            }
            std::cout << std::endl;
        }
    }
    return evaluate(model, *test_loader, device);
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main() {
    torch::manual_seed(0);
    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        torch::cuda::manual_seed_all(0);
    }
    std::cout << "Device: " << device << std::endl;

    Cifar10Dataset full_train("train_images.bin", "train_labels.bin", "train_meta.bin");
    Cifar10Dataset test_ds("test_images.bin", "test_labels.bin", "test_meta.bin");

    // deterministic permutation with seed 0
    std::vector<int64_t> perm(full_train.size().value());
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 g(0);
    std::shuffle(perm.begin(), perm.end(), g);

    std::vector<int64_t> hold_idx(perm.begin(), perm.begin() + CFG.holdout);
    std::vector<int64_t> tr_idx(perm.begin() + CFG.holdout, perm.end());

    SubsetDataset train_ds(full_train, tr_idx);
    SubsetDataset hold_ds(full_train, hold_idx);

    std::cout << "Train: " << train_ds.size().value()
              << " Hold: " << hold_ds.size().value()
              << " Test: " << test_ds.size().value() << std::endl;

    // ----- cnn3 baseline -----
    {
        MethodConfig mc; mc.kind = "plain";
        auto res = train_run("cnn3", mc, 0, train_ds, hold_ds, test_ds, device);
        std::cout << ">>> cnn3  acc=" << res.acc << "% macs=" << res.macs / 1e6 << "M\n\n";
    }

    // ----- static_fpi -----
    {
        MethodConfig mc; mc.kind = "static";
        auto res = train_run("static_fpi", mc, 0, train_ds, hold_ds, test_ds, device);
        std::cout << ">>> static_fpi  acc=" << res.acc << "% macs=" << res.macs / 1e6 << "M iters=" << res.iters << "\n\n";
    }

    // ----- dynalay_ce -----
    {
        MethodConfig mc;
        mc.kind = "dyna"; mc.policy = "agent"; mc.reward = "ce";
        mc.lam = 1.0; mc.alpha = 1.0; mc.beta = 0.05;
        auto res = train_run("dynalay_ce", mc, 0, train_ds, hold_ds, test_ds, device);
        std::cout << ">>> dynalay_ce  acc=" << res.acc << "% macs=" << res.macs / 1e6
                  << "M iters=" << res.iters << "\n";
        if (res.iters < 0.05) {
            std::cout << "!!! WARNING: エージェントがNOPに崩壊しています\n";
        }
    }

    return 0;
}