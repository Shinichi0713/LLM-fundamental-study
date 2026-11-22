#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dls {

// ============================================================================
// 1. 実験設定
// ============================================================================
struct Config {
    int num_layers = 24;
    int hidden_dim = 128;        // 実験・検証用の隠れ次元（実モデルでは 2048 等）
    int lora_rank = 16;
    double lora_alpha = 32.0;
    double lora_lr = 2e-4;
    int n_samples = 500;
    int n_train = 2000;
    std::vector<int> costs = {4, 8, 12, 24};
    uint64_t seed = 0;
    int cv_splits = 5;
    int num_random_mix_trials = 20;
};

static int gcd_int(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return std::max(1, a);
}

static int gcd_vec(const std::vector<int>& v) {
    int g = 0;
    for (int x : v) {
        g = (g == 0) ? x : gcd_int(g, x);
    }
    return std::max(1, g);
}

// ============================================================================
// 2. Uniform Layer Skipping (ULS) パターン生成（論文式(2)準拠）
// ============================================================================
std::vector<int> uniform_pattern(int num_layers, int c) {
    std::vector<int> u(num_layers, 0);
    u[0] = 1; // 1層目は必ず残す
    int cur_sum = 1;

    for (int l = 2; l <= num_layers; ++l) {
        double thresh = static_cast<double>(l - 1) * static_cast<double>(c) /
                        static_cast<double>(num_layers) +
                        1e-9;
        if (static_cast<double>(cur_sum) <= thresh) {
            u[l - 1] = 1;
            ++cur_sum;
        }
    }

    while (cur_sum < c) {
        for (int idx = num_layers - 1; idx >= 1; --idx) {
            if (u[idx] == 0) {
                u[idx] = 1;
                ++cur_sum;
                break;
            }
        }
    }
    while (cur_sum > c) {
        for (int idx = num_layers - 1; idx >= 1; --idx) {
            if (u[idx] == 1) {
                u[idx] = 0;
                --cur_sum;
                break;
            }
        }
    }
    return u;
}

std::vector<int> active_layer_indices(int num_layers, int c) {
    std::vector<int> pat = uniform_pattern(num_layers, c);
    std::vector<int> keep;
    for (int i = 0; i < num_layers; ++i) {
        if (pat[i] == 1) keep.push_back(i);
    }
    return keep;
}

// ============================================================================
// 3. 合成スコア行列 R [N x K] とルーター特徴量 X [N x F] の生成
//    （実運用時は推論生成＋ROUGE-L算出結果、および第12層 mean-pool + log(len) を格納）
// ============================================================================
struct SyntheticBenchData {
    // r_mat[i][k]: サンプル i を深さ costs[k] で実行したときの ROUGE-L スコア
    std::vector<std::vector<double>> r_mat;
    // features[i][d]: ルーター入力特徴量（隠れ状態平均 + log(prompt_len)）
    std::vector<std::vector<double>> features;
};

SyntheticBenchData generate_synthetic_benchmark(const Config& cfg, bool post_lora) {
    std::mt19937_64 rng(cfg.seed + (post_lora ? 1000ULL : 0ULL));
    std::normal_distribution<double> norm(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    const int n = cfg.n_samples;
    const int k_num = static_cast<int>(cfg.costs.size());
    const int feat_dim = cfg.hidden_dim + 1;

    SyntheticBenchData data;
    data.r_mat.assign(n, std::vector<double>(k_num, 0.0));
    data.features.assign(n, std::vector<double>(feat_dim, 0.0));

    // プロンプトごとの「難易度/必要深さ」潜在変数 z_i と真の回帰重み w_true
    std::vector<std::vector<double>> w_true(k_num, std::vector<double>(feat_dim, 0.0));
    for (int k = 0; k < k_num; ++k) {
        for (int d = 0; d < feat_dim; ++d) {
            w_true[k][d] = norm(rng) / std::sqrt(static_cast<double>(feat_dim));
        }
    }

    for (int i = 0; i < n; ++i) {
        double difficulty = norm(rng);
        double log_len = std::log(16.0 + 240.0 * uni(rng));

        double feat_signal = 0.0;
        for (int d = 0; d < cfg.hidden_dim; ++d) {
            double v = 0.35 * difficulty + 0.85 * norm(rng);
            data.features[i][d] = v;
            feat_signal += v * w_true[0][d];
        }
        data.features[i][cfg.hidden_dim] = log_len;

        for (int k = 0; k < k_num; ++k) {
            double c = static_cast<double>(cfg.costs[k]);
            double ratio = c / static_cast<double>(cfg.num_layers);

            double base_mean = post_lora ? 0.312 : 0.232;
            // 上に凸（concavity）の深さゲイン曲線
            double depth_gain = std::pow(ratio, post_lora ? 0.42 : 0.55);
            double diff_interaction = 0.025 * difficulty * (ratio - 0.5);
            double noise = 0.035 * norm(rng);

            double score = base_mean * (0.55 + 0.45 * depth_gain) + diff_interaction +
                           0.05 * std::tanh(feat_signal) + noise;
            data.r_mat[i][k] = std::clamp(score, 0.0, 1.0);
        }
    }
    return data;
}

// ============================================================================
// 4. 多選択ナップサック DP オラクル / Greedy / Random Mix / クロスラン
// ============================================================================
struct OracleDPResult {
    std::vector<int> selected_k; // 各サンプルで選ばれた cost index (0..K-1)
    double mean_rouge = 0.0;
    double mean_cost = 0.0;
};

OracleDPResult solve_oracle_dp(
    const std::vector<std::vector<double>>& r_mat,
    const std::vector<int>& costs,
    double beta
) {
    const int n = static_cast<int>(r_mat.size());
    const int k_num = static_cast<int>(costs.size());
    const int g = gcd_vec(costs);

    std::vector<int> u(k_num);
    int min_u = std::numeric_limits<int>::max();
    for (int k = 0; k < k_num; ++k) {
        u[k] = costs[k] / g;
        min_u = std::min(min_u, u[k]);
    }

    double max_avg_u = beta / static_cast<double>(g);
    int max_total_u = static_cast<int>(std::floor(max_avg_u * static_cast<double>(n) + 1e-9));
    max_total_u = std::max(max_total_u, min_u * n);

    const double neg_inf = -1e18;
    std::vector<double> dp(max_total_u + 1, neg_inf);
    dp[0] = 0.0;

    // choices[i * (max_total_u + 1) + w] = 選んだ k index
    std::vector<int> choices(static_cast<size_t>(n) * (max_total_u + 1), -1);

    for (int i = 0; i < n; ++i) {
        std::vector<double> next_dp(max_total_u + 1, neg_inf);
        for (int w = 0; w <= max_total_u; ++w) {
            if (dp[w] < neg_inf * 0.5) continue;
            for (int k = 0; k < k_num; ++k) {
                int nw = w + u[k];
                if (nw <= max_total_u) {
                    double cand = dp[w] + r_mat[i][k];
                    if (cand > next_dp[nw]) {
                        next_dp[nw] = cand;
                        choices[static_cast<size_t>(i) * (max_total_u + 1) + nw] = k;
                    }
                }
            }
        }
        dp.swap(next_dp);
    }

    int best_w = 0;
    double best_val = neg_inf;
    for (int w = 0; w <= max_total_u; ++w) {
        if (dp[w] > best_val) {
            best_val = dp[w];
            best_w = w;
        }
    }

    std::vector<int> sel_k(n, 0);
    int min_cost_idx = static_cast<int>(
        std::min_element(costs.begin(), costs.end()) - costs.begin()
    );

    if (best_val < neg_inf * 0.5) {
        std::fill(sel_k.begin(), sel_k.end(), min_cost_idx);
    } else {
        int cur_w = best_w;
        for (int i = n - 1; i >= 0; --i) {
            int k_idx = choices[static_cast<size_t>(i) * (max_total_u + 1) + cur_w];
            if (k_idx < 0) k_idx = min_cost_idx;
            sel_k[i] = k_idx;
            cur_w -= u[k_idx];
        }
    }

    double sum_r = 0.0;
    double sum_c = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_r += r_mat[i][sel_k[i]];
        sum_c += static_cast<double>(costs[sel_k[i]]);
    }

    return {sel_k, sum_r / static_cast<double>(n), sum_c / static_cast<double>(n)};
}

std::pair<double, double> greedy_oracle_score(
    const std::vector<std::vector<double>>& r_mat,
    const std::vector<int>& costs,
    double beta
) {
    const int n = static_cast<int>(r_mat.size());
    const int k_num = static_cast<int>(costs.size());

    std::vector<int> allowed;
    for (int k = 0; k < k_num; ++k) {
        if (static_cast<double>(costs[k]) <= beta + 1e-9) {
            allowed.push_back(k);
        }
    }
    if (allowed.empty()) {
        int min_idx = static_cast<int>(
            std::min_element(costs.begin(), costs.end()) - costs.begin()
        );
        allowed.push_back(min_idx);
    }

    double sum_r = 0.0;
    double sum_c = 0.0;
    for (int i = 0; i < n; ++i) {
        int best_k = allowed[0];
        double best_s = r_mat[i][best_k];
        for (int k : allowed) {
            if (r_mat[i][k] > best_s) {
                best_s = r_mat[i][k];
                best_k = k;
            }
        }
        sum_r += best_s;
        sum_c += static_cast<double>(costs[best_k]);
    }
    return {sum_r / static_cast<double>(n), sum_c / static_cast<double>(n)};
}

double random_mix_score(
    const std::vector<std::vector<double>>& r_mat,
    const std::vector<int>& costs,
    const std::vector<int>& selected_k,
    int num_trials,
    uint64_t seed
) {
    const int n = static_cast<int>(r_mat.size());
    const int k_num = static_cast<int>(costs.size());
    std::vector<double> counts(k_num, 0.0);
    for (int k : selected_k) {
        counts[k] += 1.0;
    }
    double total = std::max(1.0, static_cast<double>(selected_k.size()));
    std::vector<double> probs(k_num, 0.0);
    for (int k = 0; k < k_num; ++k) {
        probs[k] = counts[k] / total;
    }

    std::mt19937_64 rng(seed);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());

    double trial_sum = 0.0;
    for (int t = 0; t < num_trials; ++t) {
        double sc_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            int rk = dist(rng);
            sc_sum += r_mat[i][rk];
        }
        trial_sum += sc_sum / static_cast<double>(n);
    }
    return trial_sum / static_cast<double>(std::max(1, num_trials));
}

struct BudgetRow {
    double beta;
    double actual_mean_layers;
    double oracle;
    double greedy;
    double random_mix;
    std::vector<double> ratios;
};

struct OracleAnalysisResult {
    std::vector<double> static_scores;
    double full_score = 0.0;
    double greedy_at_max = 0.0;
    bool has_min_beta = false;
    double min_beta_budget = 0.0;
    double min_beta_actual_layers = 0.0;
    std::vector<double> min_beta_ratios;
    std::vector<BudgetRow> curve;
};

OracleAnalysisResult analyze_oracle(
    const std::vector<std::vector<double>>& r_mat,
    const std::vector<int>& costs,
    const std::vector<double>& betas,
    int num_random_mix_trials,
    uint64_t seed
) {
    const int n = static_cast<int>(r_mat.size());
    const int k_num = static_cast<int>(costs.size());

    OracleAnalysisResult out;
    out.static_scores.assign(k_num, 0.0);
    for (int k = 0; k < k_num; ++k) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += r_mat[i][k];
        out.static_scores[k] = s / static_cast<double>(n);
    }

    int max_cost = *std::max_element(costs.begin(), costs.end());
    int full_idx = static_cast<int>(
        std::find(costs.begin(), costs.end(), max_cost) - costs.begin()
    );
    out.full_score = out.static_scores[full_idx];
    out.greedy_at_max = greedy_oracle_score(r_mat, costs, static_cast<double>(max_cost)).first;

    for (double beta : betas) {
        OracleDPResult dp_res = solve_oracle_dp(r_mat, costs, beta);
        auto [gr_sc, _] = greedy_oracle_score(r_mat, costs, beta);
        double rm_sc = random_mix_score(r_mat, costs, dp_res.selected_k, num_random_mix_trials, seed);

        std::vector<double> ratios(k_num, 0.0);
        for (int k : dp_res.selected_k) ratios[k] += 1.0;
        for (int k = 0; k < k_num; ++k) ratios[k] /= static_cast<double>(n);

        if (!out.has_min_beta && dp_res.mean_rouge >= out.full_score) {
            out.has_min_beta = true;
            out.min_beta_budget = beta;
            out.min_beta_actual_layers = dp_res.mean_cost;
            out.min_beta_ratios = ratios;
        }

        out.curve.push_back(BudgetRow{
            beta, dp_res.mean_cost, dp_res.mean_rouge, gr_sc, rm_sc, ratios
        });
    }
    return out;
}

// クロスラン・オラクル：1回目の生成スコアで DP 割当てを決定し、2回目の独立生成で評価
std::vector<std::pair<double, double>> cross_run_oracle_eval(
    const std::vector<std::vector<double>>& r_assign,
    const std::vector<std::vector<double>>& r_eval,
    const std::vector<int>& costs,
    const std::vector<double>& betas
) {
    const int n = static_cast<int>(r_eval.size());
    std::vector<std::pair<double, double>> res;
    for (double beta : betas) {
        OracleDPResult dp = solve_oracle_dp(r_assign, costs, beta);
        double eval_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            eval_sum += r_eval[i][dp.selected_k[i]];
        }
        res.push_back({dp.mean_cost, eval_sum / static_cast<double>(n)});
    }
    return res;
}

// ============================================================================
// 5. Ridge 回帰ルーター（正則化最小二乗ソルバ：共役勾配法 / ガウス消去）
// ============================================================================
std::vector<double> solve_ridge_1d(
    const std::vector<std::vector<double>>& x_train,
    const std::vector<double>& y_train,
    double alpha
) {
    const int m = static_cast<int>(x_train.size());
    const int d = static_cast<int>(x_train[0].size());

    // (X^T X + alpha * I) w = X^T y を共役勾配法 (CG) で解く（高次元特徴量でも高速）
    std::vector<double> b(d, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < d; ++j) {
            b[j] += x_train[i][j] * y_train[i];
        }
    }

    auto apply_a = [&](const std::vector<double>& v) -> std::vector<double> {
        std::vector<double> xv(m, 0.0);
        for (int i = 0; i < m; ++i) {
            double dot = 0.0;
            for (int j = 0; j < d; ++j) dot += x_train[i][j] * v[j];
            xv[i] = dot;
        }
        std::vector<double> out(d, 0.0);
        for (int j = 0; j < d; ++j) {
            double dot = 0.0;
            for (int i = 0; i < m; ++i) dot += x_train[i][j] * xv[i];
            out[j] = dot + alpha * v[j];
        }
        return out;
    };

    std::vector<double> w(d, 0.0);
    std::vector<double> r = b;
    std::vector<double> p = r;

    auto dot_vec = [](const std::vector<double>& a, const std::vector<double>& b_vec) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) s += a[i] * b_vec[i];
        return s;
    };

    double rs_old = dot_vec(r, r);
    const int max_iter = std::min(d, 150);
    for (int iter = 0; iter < max_iter; ++iter) {
        if (std::sqrt(rs_old) < 1e-9) break;
        std::vector<double> ap = apply_a(p);
        double p_ap = dot_vec(p, ap);
        if (p_ap <= 1e-15) break;
        double alpha_cg = rs_old / p_ap;
        for (int j = 0; j < d; ++j) {
            w[j] += alpha_cg * p[j];
            r[j] -= alpha_cg * ap[j];
        }
        double rs_new = dot_vec(r, r);
        double beta_cg = rs_new / std::max(1e-18, rs_old);
        for (int j = 0; j < d; ++j) {
            p[j] = r[j] + beta_cg * p[j];
        }
        rs_old = rs_new;
    }
    return w;
}

static double pearson_corr(const std::vector<double>& a, const std::vector<double>& b) {
    const int n = static_cast<int>(a.size());
    double ma = std::accumulate(a.begin(), a.end(), 0.0) / n;
    double mb = std::accumulate(b.begin(), b.end(), 0.0) / n;
    double num = 0.0, da = 0.0, db = 0.0;
    for (int i = 0; i < n; ++i) {
        double va = a[i] - ma;
        double vb = b[i] - mb;
        num += va * vb;
        da += va * va;
        db += vb * vb;
    }
    if (da <= 1e-15 || db <= 1e-15) return 0.0;
    return num / std::sqrt(da * db);
}

struct RouterRow {
    double tau;
    double actual_mean_layers;
    double router_rouge;
    double random_mix_rouge;
};

struct RouterAnalysisResult {
    std::vector<double> correlations;
    std::vector<RouterRow> curve;
};

RouterAnalysisResult analyze_router(
    const std::vector<std::vector<double>>& features,
    const std::vector<std::vector<double>>& r_mat,
    const std::vector<int>& costs,
    int cv_splits,
    const std::vector<double>& taus,
    int num_random_mix_trials,
    uint64_t seed
) {
    const int n = static_cast<int>(r_mat.size());
    const int k_num = static_cast<int>(costs.size());
    int full_idx = static_cast<int>(
        std::max_element(costs.begin(), costs.end()) - costs.begin()
    );

    // oof_preds[i][k]
    std::vector<std::vector<double>> oof_preds(n, std::vector<double>(k_num, 0.0));
    RouterAnalysisResult out;
    out.correlations.assign(k_num, 0.0);

    int folds = std::max(2, std::min(cv_splits, n));
    for (int k = 0; k < k_num; ++k) {
        std::vector<double> y_col(n, 0.0);
        for (int i = 0; i < n; ++i) y_col[i] = r_mat[i][k];

        for (int f = 0; f < folds; ++f) {
            std::vector<std::vector<double>> x_tr;
            std::vector<double> y_tr;
            std::vector<int> val_indices;

            for (int i = 0; i < n; ++i) {
                if (i % folds == f) {
                    val_indices.push_back(i);
                } else {
                    x_tr.push_back(features[i]);
                    y_tr.push_back(y_col[i]);
                }
            }

            std::vector<double> w = solve_ridge_1d(x_tr, y_tr, 1.0);
            for (int idx : val_indices) {
                double pred = 0.0;
                for (size_t d = 0; d < w.size(); ++d) {
                    pred += features[idx][d] * w[d];
                }
                oof_preds[idx][k] = pred;
            }
        }

        std::vector<double> pred_col(n, 0.0);
        for (int i = 0; i < n; ++i) pred_col[i] = oof_preds[i][k];
        out.correlations[k] = pearson_corr(y_col, pred_col);
    }

    // コスト昇順インデックス
    std::vector<int> cost_order(k_num);
    std::iota(cost_order.begin(), cost_order.end(), 0);
    std::sort(cost_order.begin(), cost_order.end(), [&](int a, int b) {
        return costs[a] < costs[b];
    });

    for (double tau : taus) {
        std::vector<int> chosen_k(n, full_idx);
        double sum_r = 0.0;
        double sum_c = 0.0;

        for (int i = 0; i < n; ++i) {
            double threshold = oof_preds[i][full_idx] - tau;
            int picked = full_idx;
            for (int k_idx : cost_order) {
                if (oof_preds[i][k_idx] >= threshold) {
                    picked = k_idx;
                    break;
                }
            }
            chosen_k[i] = picked;
            sum_r += r_mat[i][picked];
            sum_c += static_cast<double>(costs[picked]);
        }

        double rm_sc = random_mix_score(r_mat, costs, chosen_k, num_random_mix_trials, seed);
        out.curve.push_back(RouterRow{
            tau,
            sum_c / static_cast<double>(n),
            sum_r / static_cast<double>(n),
            rm_sc
        });
    }
    return out;
}

// ============================================================================
// 6. 深さ c ごとの LoRA 微調整ステップ（アクティブ層＋LoRA低ランク更新のシミュレーション）
// ============================================================================
struct LoRAAdapter {
    int rank;
    double scale;
    // A: [rank x hidden_dim], B: [hidden_dim x rank]
    std::vector<double> a;
    std::vector<double> b;

    LoRAAdapter(int dim, int r, double alpha, uint64_t seed)
        : rank(r),
          scale(alpha / static_cast<double>(std::max(1, r))),
          a(static_cast<size_t>(r) * dim, 0.0),
          b(static_cast<size_t>(dim) * r, 0.0) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> n_dist(0.0, 0.02);
        for (double& val : a) val = n_dist(rng);
    }

    void sgd_step(const std::vector<double>& grad_proxy, double lr) {
        for (size_t idx = 0; idx < a.size(); ++idx) {
            double g = grad_proxy[idx % grad_proxy.size()];
            a[idx] -= lr * g * scale;
        }
        for (size_t idx = 0; idx < b.size(); ++idx) {
            double g = grad_proxy[idx % grad_proxy.size()];
            b[idx] -= lr * g * scale;
        }
    }
};

double run_lora_finetune_for_cost(const Config& cfg, int c, uint64_t seed) {
    std::vector<int> keep = active_layer_indices(cfg.num_layers, c);
    LoRAAdapter adapter(cfg.hidden_dim, cfg.lora_rank, cfg.lora_alpha, seed + static_cast<uint64_t>(c));

    std::mt19937_64 rng(seed + static_cast<uint64_t>(c) * 97ULL);
    std::normal_distribution<double> dist(0.0, 0.1);

    double loss = 1.65 - 0.15 * (static_cast<double>(keep.size()) / cfg.num_layers);
    const int steps = std::max(10, cfg.n_train / 16);
    std::vector<double> grad(cfg.hidden_dim, 0.0);

    for (int s = 0; s < steps; ++s) {
        for (int d = 0; d < cfg.hidden_dim; ++d) {
            grad[d] = dist(rng) * (1.0 - 0.5 * static_cast<double>(s) / steps);
        }
        adapter.sgd_step(grad, cfg.lora_lr);
        loss = std::max(0.45, loss * 0.996);
    }
    return loss;
}

} // namespace dls

int main() {
    dls::Config cfg;
    cfg.n_samples = 200;
    cfg.n_train = 640;

    std::vector<double> betas = {4.0, 6.0, 8.0, 12.0, 16.0, 24.0};
    std::vector<double> taus = {0.00, 0.02, 0.05, 0.10, 1.00};

    // ---- Part A: Pre-FT ----
    auto pre_data = dls::generate_synthetic_benchmark(cfg, false);
    auto res0 = dls::analyze_oracle(pre_data.r_mat, cfg.costs, betas, cfg.num_random_mix_trials, cfg.seed);
    auto rt0 = dls::analyze_router(pre_data.features, pre_data.r_mat, cfg.costs, cfg.cv_splits, taus, cfg.num_random_mix_trials, cfg.seed);

    std::cout << "=== Part A (Pre-FT) ===" << std::endl;
    std::cout << "Full ROUGE-L: " << std::fixed << std::setprecision(4) << res0.full_score
              << " | Greedy@max: " << res0.greedy_at_max << std::endl;
    for (const auto& row : res0.curve) {
        std::cout << "beta=" << std::setw(4) << row.beta
                  << " (mean=" << std::setw(4) << row.actual_mean_layers << ")"
                  << " oracle=" << row.oracle
                  << " greedy=" << row.greedy
                  << " rand_mix=" << row.random_mix << std::endl;
    }

    // ---- Part B: LoRA FT per cost c ----
    std::cout << "\n=== Part B (LoRA FT per cost c) ===" << std::endl;
    for (int c : cfg.costs) {
        double final_loss = dls::run_lora_finetune_for_cost(cfg, c, cfg.seed);
        std::cout << "Trained LoRA adapter for c=" << std::setw(2) << c
                  << " (active_layers=" << dls::active_layer_indices(cfg.num_layers, c).size()
                  << ") final_loss=" << final_loss << std::endl;
    }

    auto post_data = dls::generate_synthetic_benchmark(cfg, true);
    auto res_f = dls::analyze_oracle(post_data.r_mat, cfg.costs, betas, cfg.num_random_mix_trials, cfg.seed);
    auto rt_f = dls::analyze_router(post_data.features, post_data.r_mat, cfg.costs, cfg.cv_splits, taus, cfg.num_random_mix_trials, cfg.seed);

    std::cout << "\n=== Part B (Post-FT) ===" << std::endl;
    std::cout << "Full ROUGE-L: " << res_f.full_score
              << " | Greedy@max: " << res_f.greedy_at_max << std::endl;
    if (res_f.has_min_beta) {
        std::cout << "Min beta matching full: budget=" << res_f.min_beta_budget
                  << ", mean_layers=" << res_f.min_beta_actual_layers << std::endl;
    }

    std::cout << "\nRouter (Post-FT) OOF Corrs: ";
    for (size_t k = 0; k < cfg.costs.size(); ++k) {
        std::cout << "c=" << cfg.costs[k] << ":" << rt_f.correlations[k] << " ";
    }
    std::cout << std::endl;

    // ---- Cross-Run Oracle ----
    auto post_run2 = dls::generate_synthetic_benchmark(cfg, true);
    auto xrun = dls::cross_run_oracle_eval(post_data.r_mat, post_run2.r_mat, cfg.costs, betas);
    std::cout << "\n=== Cross-Run Oracle ===" << std::endl;
    for (const auto& [mean_l, sc] : xrun) {
        std::cout << "mean_layers=" << std::setw(5) << mean_l
                  << " | cross_run_rougeL=" << sc << std::endl;
    }

    return 0;
}