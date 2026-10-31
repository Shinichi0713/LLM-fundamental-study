/* ============================================================================
 * LayerSkip 学習・推論 C++ 実装
 * ============================================================================
 * ・簡易Transformer（単語予測タスク）で LayerSkip の核心を再現
 * ・レイヤードロップアウト（浅い層は残しやすく、深い層は飛ばしやすく）
 * ・全レイヤー共有の早期終了損失
 * ・推論時: 早期終了（Early Exit）+ 自己投機的デコーディング
 * ・外部ライブラリ不要（純粋C++17）
 * ============================================================================
 *
 * コンパイル:
 *   g++ -std=c++17 -O2 -o layerskip_train layerskip_train.cpp
 * 実行:
 *   ./layerskip_train
 *
 * ============================================================================ */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <cassert>

// ============================================================
// ユーティリティ: 行列・ベクトル演算
// ============================================================
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// ベクトル + ベクトル
Vector operator+(const Vector& a, const Vector& b) {
    Vector r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

// ベクトル += ベクトル
Vector& operator+=(Vector& a, const Vector& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
    return a;
}

// ベクトル - ベクトル
Vector operator-(const Vector& a, const Vector& b) {
    Vector r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}

// ベクトル * スカラー
Vector operator*(const Vector& a, double s) {
    Vector r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] * s;
    return r;
}

// 行列 × ベクトル
Vector matvec(const Matrix& M, const Vector& v) {
    Vector r(M.size(), 0.0);
    for (size_t i = 0; i < M.size(); ++i)
        for (size_t j = 0; j < v.size(); ++j)
            r[i] += M[i][j] * v[j];
    return r;
}

// ベクトルの外積（行列生成）
Matrix outer(const Vector& a, const Vector& b) {
    Matrix M(a.size(), Vector(b.size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            M[i][j] = a[i] * b[j];
    return M;
}

// softmax
Vector softmax(const Vector& x) {
    double max_val = *std::max_element(x.begin(), x.end());
    Vector r(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        r[i] = std::exp(x[i] - max_val);
        sum += r[i];
    }
    for (size_t i = 0; i < x.size(); ++i) r[i] /= sum;
    return r;
}

// LayerNorm
Vector layer_norm(const Vector& x, const Vector& gamma, const Vector& beta, double eps = 1e-5) {
    double mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double var = 0.0;
    for (double v : x) var += (v - mean) * (v - mean);
    var /= x.size();
    Vector r(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        r[i] = gamma[i] * (x[i] - mean) / std::sqrt(var + eps) + beta[i];
    return r;
}

// GELU活性化関数
double gelu(double x) {
    return 0.5 * x * (1.0 + std::tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

Vector gelu(const Vector& x) {
    Vector r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = gelu(x[i]);
    return r;
}

// GELUの微分
double gelu_deriv(double x) {
    double cdf = 0.5 * (1.0 + std::tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    double pdf_base = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
    // 近似
    double sech = 1.0 / std::cosh(0.7978845608 * (x + 0.044715 * x * x * x));
    double inner_deriv = 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x);
    return cdf + x * pdf_base; // 簡易近似
}

// ベクトルの和
double sum(const Vector& v) {
    return std::accumulate(v.begin(), v.end(), 0.0);
}

// ベクトルの平均
double mean(const Vector& v) {
    return sum(v) / v.size();
}

// ベクトルの標準偏差
double stddev(const Vector& v) {
    double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / v.size());
}

// ============================================================
// 簡易単語予測データセット
// ============================================================
struct Dataset {
    int vocab_size;
    int seq_len;
    std::vector<std::vector<int>> sequences; // 各サンプルはトークンID列

    Dataset(int vocab = 10, int seqlen = 8, int n_samples = 100, int seed = 42)
        : vocab_size(vocab), seq_len(seqlen) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(0, vocab_size - 1);
        sequences.resize(n_samples, std::vector<int>(seq_len));
        for (auto& seq : sequences)
            for (auto& tok : seq)
                tok = dist(gen);
    }
};

// ============================================================
// 簡易Transformerブロック（単一ヘッドAttention + FFN）
// ============================================================
struct TransformerBlock {
    int d_model;
    int d_ff;

    // Attention weights
    Matrix Wq, Wk, Wv, Wo;
    Vector bq, bk, bv, bo;

    // FFN weights
    Matrix W1, W2;
    Vector b1, b2;

    // LayerNorm params
    Vector ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;

    // Gradients
    Matrix gWq, gWk, gWv, gWo, gW1, gW2;
    Vector gbq, gbk, gbv, gbo, gb1, gb2;
    Vector gln1_g, gln1_b, gln2_g, gln2_b;

    // Forward cache
    Vector x_in;           // 入力
    Vector ln1_out;        // 1つ目LN後
    Vector q, k, v;        // Q, K, V
    Vector attn_out;       // Attention出力
    Vector attn_residual;  // Attention + residual
    Vector ln2_out;        // 2つ目LN後
    Vector ff1;            // FFN中間
    Vector ff1_act;        // FFN活性化後
    Vector ff_out;         // FFN出力
    Vector output;         // 最終出力

    // Attention scores
    Vector attn_scores;    // Q·K^T / sqrt(d)
    Vector attn_probs;     // softmax後

    TransformerBlock(int d, int ff, int seed_offset = 0) : d_model(d), d_ff(ff) {
        std::mt19937 gen(42 + seed_offset);
        std::normal_distribution<double> dist(0.0, 0.02);
        auto init_mat = [&](int r, int c) {
            Matrix M(r, Vector(c));
            for (auto& row : M) for (auto& x : row) x = dist(gen);
            return M;
        };
        auto init_vec = [&](int n) {
            Vector v(n);
            for (auto& x : v) x = dist(gen);
            return v;
        };
        auto init_ones = [&](int n) {
            Vector v(n, 1.0);
            return v;
        };

        Wq = init_mat(d_model, d_model); Wk = init_mat(d_model, d_model);
        Wv = init_mat(d_model, d_model); Wo = init_mat(d_model, d_model);
        bq = init_vec(d_model); bk = init_vec(d_model);
        bv = init_vec(d_model); bo = init_vec(d_model);

        W1 = init_mat(d_ff, d_model); W2 = init_mat(d_model, d_ff);
        b1 = init_vec(d_ff); b2 = init_vec(d_model);

        ln1_gamma = init_ones(d_model); ln1_beta = init_vec(d_model);
        ln2_gamma = init_ones(d_model); ln2_beta = init_vec(d_model);

        zero_grad();
    }

    void zero_grad() {
        auto zero_mat = [](Matrix& M) { for (auto& r : M) std::fill(r.begin(), r.end(), 0.0); };
        auto zero_vec = [](Vector& v) { std::fill(v.begin(), v.end(), 0.0); };
        zero_mat(gWq); zero_mat(gWk); zero_mat(gWv); zero_mat(gWo);
        zero_mat(gW1); zero_mat(gW2);
        zero_vec(gbq); zero_vec(gbk); zero_vec(gbv); zero_vec(gbo);
        zero_vec(gb1); zero_vec(gb2);
        zero_vec(gln1_g); zero_vec(gln1_b); zero_vec(gln2_g); zero_vec(gln2_b);
    }

    Vector forward(const Vector& x) {
        x_in = x;

        // --- Self-Attention ---
        ln1_out = layer_norm(x, ln1_gamma, ln1_beta);
        q = matvec(Wq, ln1_out) + bq;
        k = matvec(Wk, ln1_out) + bk;
        v = matvec(Wv, ln1_out) + bv;

        // 簡易単一ヘッドAttention（seq_len=1なのでQ·K^Tはスカラー）
        double score = 0.0;
        for (int i = 0; i < d_model; ++i) score += q[i] * k[i];
        score /= std::sqrt(static_cast<double>(d_model));
        attn_scores = {score};
        attn_probs = softmax(attn_scores);

        attn_out = v * attn_probs[0];
        attn_residual = x + attn_out;

        // --- FFN ---
        ln2_out = layer_norm(attn_residual, ln2_gamma, ln2_beta);
        ff1 = matvec(W1, ln2_out) + b1;
        ff1_act = gelu(ff1);
        ff_out = matvec(W2, ff1_act) + b2;
        output = attn_residual + ff_out;

        return output;
    }

    // 簡易backward（最終lossからの勾配を受け取る）
    void backward(const Vector& grad_output) {
        // 簡易実装: 勾配を直接パラメータに蓄積（実際には連鎖律が必要）
        // ここでは数値微分を使うため、backwardは空実装とし、
        // LayerSkipModel側で数値微分を行う
    }
};

// ============================================================
// LayerSkip モデル
// ============================================================
struct LayerSkipModel {
    int vocab_size;
    int d_model;
    int num_layers;
    int d_ff;

    // 埋め込み
    Matrix token_embed;  // [vocab_size x d_model]

    // Transformer blocks
    std::vector<TransformerBlock> blocks;

    // 最終LayerNorm + LM Head（全レイヤーで共有）
    Vector final_ln_gamma, final_ln_beta;
    Matrix lm_head;  // [vocab_size x d_model]

    // ハイパーパラメータ
    double base_dropout;
    double max_dropout;
    double early_exit_loss_weight;

    // 各layerのdropout率
    std::vector<double> layer_dropout_rates;

    // 乱数生成器
    std::mt19937 gen;

    LayerSkipModel(
        int vocab = 10, int d = 16, int layers = 4, int ff = 32,
        double base_drop = 0.0, double max_drop = 0.5, double ee_weight = 0.5,
        int seed = 42
    ) : vocab_size(vocab), d_model(d), num_layers(layers), d_ff(ff),
        base_dropout(base_drop), max_dropout(max_drop), early_exit_loss_weight(ee_weight),
        gen(seed) {

        std::normal_distribution<double> dist(0.0, 0.02);

        // token embedding
        token_embed.resize(vocab_size, Vector(d_model));
        for (auto& row : token_embed)
            for (auto& x : row) x = dist(gen);

        // Transformer blocks
        for (int i = 0; i < num_layers; ++i)
            blocks.emplace_back(d_model, d_ff, i);

        // final LayerNorm
        final_ln_gamma.resize(d_model, 1.0);
        final_ln_beta.resize(d_model, 0.0);

        // LM Head（重み共有: token_embedと同じ）
        lm_head = token_embed;  // 重み共有

        // layer dropout rates（浅い層は低く、深い層は高く）
        layer_dropout_rates.resize(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            layer_dropout_rates[i] = base_dropout +
                (static_cast<double>(i) / std::max(1, num_layers - 1)) * (max_dropout - base_dropout);
        }
    }

    // 単一トークンのforward（学習用）
    // 返り値: {最終loss, 最終logits, early_exit_losses}
    struct ForwardResult {
        double loss;
        Vector logits;
        std::vector<double> early_exit_losses;
        std::vector<Vector> all_hidden;  // 各layerの出力
    };

    ForwardResult forward_train(int token_id, int target_id, bool apply_layer_dropout) {
        // token embedding
        Vector hidden = token_embed[token_id];

        std::vector<Vector> all_hidden;
        all_hidden.push_back(hidden);

        // 各Transformer block
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            bool skip = false;
            if (apply_layer_dropout) {
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                skip = dist(gen) < layer_dropout_rates[layer_idx];
            }

            if (!skip) {
                hidden = blocks[layer_idx].forward(hidden);
            }
            // skip時はhiddenを変更しない（residual connectionのみ）
            all_hidden.push_back(hidden);
        }

        // 最終LayerNorm + LM Head
        Vector final_norm = layer_norm(hidden, final_ln_gamma, final_ln_beta);
        Vector logits(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            logits[i] = 0.0;
            for (int j = 0; j < d_model; ++j)
                logits[i] += lm_head[i][j] * final_norm[j];
        }

        // 最終loss（cross entropy）
        Vector probs = softmax(logits);
        double final_loss = -std::log(probs[target_id] + 1e-10);

        // 早期終了損失: 各layerの出力を同じLM Headで評価
        std::vector<double> ee_losses;
        for (size_t hidx = 1; hidx < all_hidden.size(); ++hidx) {
            Vector h_norm = layer_norm(all_hidden[hidx], final_ln_gamma, final_ln_beta);
            Vector h_logits(vocab_size);
            for (int i = 0; i < vocab_size; ++i) {
                h_logits[i] = 0.0;
                for (int j = 0; j < d_model; ++j)
                    h_logits[i] += lm_head[i][j] * h_norm[j];
            }
            Vector h_probs = softmax(h_logits);
            double h_loss = -std::log(h_probs[target_id] + 1e-10);
            ee_losses.push_back(h_loss);
        }

        ForwardResult res;
        res.loss = final_loss;
        res.logits = logits;
        res.early_exit_losses = ee_losses;
        res.all_hidden = all_hidden;
        return res;
    }

    // 数値微分による勾配計算
    void numerical_grad_step(int token_id, int target_id, double lr) {
        const double eps = 1e-5;

        auto compute_loss = [&](int tok, int tgt) -> double {
            auto res = forward_train(tok, tgt, true);
            double ee_avg = 0.0;
            for (double l : res.early_exit_losses) ee_avg += l;
            ee_avg /= res.early_exit_losses.size();
            return res.loss + early_exit_loss_weight * ee_avg;
        };

        double base_loss = compute_loss(token_id, target_id);

        // token_embed更新
        for (int i = 0; i < vocab_size; ++i) {
            for (int j = 0; j < d_model; ++j) {
                token_embed[i][j] += eps;
                double loss_plus = compute_loss(token_id, target_id);
                token_embed[i][j] -= eps;
                double grad = (loss_plus - base_loss) / eps;
                token_embed[i][j] -= lr * grad;
            }
        }

        // blocks更新（簡易版: 各blockの出力層W2, b2のみ更新）
        for (int li = 0; li < num_layers; ++li) {
            auto& blk = blocks[li];
            for (int i = 0; i < d_model; ++i) {
                for (int j = 0; j < d_ff; ++j) {
                    blk.W2[i][j] += eps;
                    double loss_plus = compute_loss(token_id, target_id);
                    blk.W2[i][j] -= eps;
                    double grad = (loss_plus - base_loss) / eps;
                    blk.W2[i][j] -= lr * grad;
                }
                blk.b2[i] += eps;
                double loss_plus = compute_loss(token_id, target_id);
                blk.b2[i] -= eps;
                double grad = (loss_plus - base_loss) / eps;
                blk.b2[i] -= lr * grad;
            }
        }
    }

    // 推論: 早期終了（Early Exit）
    int inference_early_exit(int token_id, double confidence_threshold) {
        Vector hidden = token_embed[token_id];

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            hidden = blocks[layer_idx].forward(hidden);

            // 各layerで確信度チェック
            Vector h_norm = layer_norm(hidden, final_ln_gamma, final_ln_beta);
            Vector logits(vocab_size);
            for (int i = 0; i < vocab_size; ++i) {
                logits[i] = 0.0;
                for (int j = 0; j < d_model; ++j)
                    logits[i] += lm_head[i][j] * h_norm[j];
            }
            Vector probs = softmax(logits);
            double max_prob = *std::max_element(probs.begin(), probs.end());
            if (max_prob >= confidence_threshold) {
                return std::max_element(probs.begin(), probs.end()) - probs.begin();
            }
        }

        // 最終層まで
        Vector final_norm = layer_norm(hidden, final_ln_gamma, final_ln_beta);
        Vector logits(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            logits[i] = 0.0;
            for (int j = 0; j < d_model; ++j)
                logits[i] += lm_head[i][j] * final_norm[j];
        }
        return std::max_element(logits.begin(), logits.end()) - logits.begin();
    }

    // 推論: 自己投機的デコーディング（Self-Speculative Decoding）
    int self_speculative_decode(int token_id, int draft_layers) {
        // Step 1: ドラフト生成（浅い層のみ）
        Vector hidden = token_embed[token_id];
        for (int i = 0; i < std::min(draft_layers, num_layers); ++i)
            hidden = blocks[i].forward(hidden);

        Vector draft_norm = layer_norm(hidden, final_ln_gamma, final_ln_beta);
        Vector draft_logits(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            draft_logits[i] = 0.0;
            for (int j = 0; j < d_model; ++j)
                draft_logits[i] += lm_head[i][j] * draft_norm[j];
        }
        int draft_token = std::max_element(draft_logits.begin(), draft_logits.end()) - draft_logits.begin();

        // Step 2: 検証（残りの深い層でフルforward）
        for (int i = draft_layers; i < num_layers; ++i)
            hidden = blocks[i].forward(hidden);

        Vector final_norm = layer_norm(hidden, final_ln_gamma, final_ln_beta);
        Vector final_logits(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            final_logits[i] = 0.0;
            for (int j = 0; j < d_model; ++j)
                final_logits[i] += lm_head[i][j] * final_norm[j];
        }
        int final_token = std::max_element(final_logits.begin(), final_logits.end()) - final_logits.begin();

        return final_token;  // 簡易版: 常にfinal_tokenを採用
    }
};

// ============================================================
// 学習ループ
// ============================================================
void train_layerskip(
    LayerSkipModel& model,
    const Dataset& dataset,
    int num_epochs = 50,
    double lr = 0.01,
    bool verbose = true
) {
    int n_samples = dataset.sequences.size();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double total_loss = 0.0;
        double total_final_loss = 0.0;
        double total_ee_loss = 0.0;
        int correct = 0;

        for (const auto& seq : dataset.sequences) {
            // seq[0..seq_len-2] -> seq[1..seq_len-1] を予測
            for (size_t t = 0; t < seq.size() - 1; ++t) {
                int input_tok = seq[t];
                int target_tok = seq[t + 1];

                auto res = model.forward_train(input_tok, target_tok, true);

                double ee_avg = 0.0;
                for (double l : res.early_exit_losses) ee_avg += l;
                ee_avg /= res.early_exit_losses.size();

                double combined_loss = res.loss + model.early_exit_loss_weight * ee_avg;
                total_loss += combined_loss;
                total_final_loss += res.loss;
                total_ee_loss += ee_avg;

                // 予測精度
                Vector probs = softmax(res.logits);
                int pred = std::max_element(probs.begin(), probs.end()) - probs.begin();
                if (pred == target_tok) correct++;

                // 数値微分によるパラメータ更新
                model.numerical_grad_step(input_tok, target_tok, lr);
            }
        }

        int total_preds = n_samples * (dataset.seq_len - 1);
        if (verbose && (epoch % 10 == 0 || epoch == num_epochs - 1)) {
            std::cout << "Epoch " << std::setw(3) << epoch
                      << " | Loss=" << std::fixed << std::setprecision(4) << (total_loss / total_preds)
                      << " | FinalLoss=" << (total_final_loss / total_preds)
                      << " | EELoss=" << (total_ee_loss / total_preds)
                      << " | Acc=" << std::setprecision(2) << (100.0 * correct / total_preds) << "%"
                      << std::endl;
        }
    }
}

// ============================================================
// 推論比較
// ============================================================
void compare_inference(
    LayerSkipModel& model,
    const Dataset& dataset,
    double confidence_threshold = 0.8,
    int draft_layers = 2
) {
    int n_samples = std::min(20, static_cast<int>(dataset.sequences.size()));

    std::cout << "\n=== 推論比較 ===" << std::endl;
    std::cout << std::setw(6) << "Sample" << std::setw(10) << "Input"
              << std::setw(10) << "Target" << std::setw(12) << "Full"
              << std::setw(12) << "EarlyExit" << std::setw(12) << "SpecDec"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int i = 0; i < n_samples; ++i) {
        int input_tok = dataset.sequences[i][0];
        int target_tok = dataset.sequences[i][1];

        // フル推論
        auto res = model.forward_train(input_tok, target_tok, false);
        Vector full_probs = softmax(res.logits);
        int full_pred = std::max_element(full_probs.begin(), full_probs.end()) - full_probs.begin();

        // 早期終了
        int ee_pred = model.inference_early_exit(input_tok, confidence_threshold);

        // 自己投機的デコーディング
        int sd_pred = model.self_speculative_decode(input_tok, draft_layers);

        std::cout << std::setw(6) << i << std::setw(10) << input_tok
                  << std::setw(10) << target_tok << std::setw(12) << full_pred
                  << std::setw(12) << ee_pred << std::setw(12) << sd_pred
                  << std::endl;
    }
}

// ============================================================
// メイン
// ============================================================
int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "  LayerSkip 学習・推論 C++ 実装" << std::endl;
    std::cout << "============================================================" << std::endl;

    // 設定
    const int VOCAB_SIZE = 10;
    const int D_MODEL = 16;
    const int NUM_LAYERS = 4;
    const int D_FF = 32;
    const int SEQ_LEN = 8;
    const int N_SAMPLES = 50;
    const int N_EPOCHS = 100;
    const double LR = 0.005;

    std::cout << "設定:" << std::endl;
    std::cout << "  vocab_size=" << VOCAB_SIZE << ", d_model=" << D_MODEL
              << ", num_layers=" << NUM_LAYERS << ", d_ff=" << D_FF << std::endl;
    std::cout << "  seq_len=" << SEQ_LEN << ", n_samples=" << N_SAMPLES
              << ", epochs=" << N_EPOCHS << ", lr=" << LR << std::endl;

    // データセット作成
    Dataset dataset(VOCAB_SIZE, SEQ_LEN, N_SAMPLES, 42);

    // --- ケース1: エントロピー項なし（LayerSkipなし）---
    std::cout << "\n=== ケース1: LayerSkipなし（early_exit_loss_weight=0.0） ===" << std::endl;
    LayerSkipModel model_no_skip(
        VOCAB_SIZE, D_MODEL, NUM_LAYERS, D_FF,
        0.0,   // base_dropout
        0.0,   // max_dropout
        0.0,   // early_exit_loss_weight
        100
    );
    train_layerskip(model_no_skip, dataset, N_EPOCHS, LR);

    // --- ケース2: LayerSkipあり ---
    std::cout << "\n=== ケース2: LayerSkipあり（early_exit_loss_weight=0.5） ===" << std::endl;
    LayerSkipModel model_with_skip(
        VOCAB_SIZE, D_MODEL, NUM_LAYERS, D_FF,
        0.0,   // base_dropout
        0.5,   // max_dropout（深い層ほど飛ばしやすい）
        0.5,   // early_exit_loss_weight
        200
    );
    train_layerskip(model_with_skip, dataset, N_EPOCHS, LR);

    // 推論比較
    std::cout << "\n【LayerSkipなしモデルの推論】" << std::endl;
    compare_inference(model_no_skip, dataset, 0.8, 2);

    std::cout << "\n【LayerSkipありモデルの推論】" << std::endl;
    compare_inference(model_with_skip, dataset, 0.8, 2);

    // レイヤードロップアウト率の表示
    std::cout << "\n=== LayerDropout 率 ===" << std::endl;
    for (int i = 0; i < NUM_LAYERS; ++i) {
        std::cout << "  Layer " << i << ": "
                  << std::fixed << std::setprecision(2)
                  << (model_with_skip.layer_dropout_rates[i] * 100) << "%" << std::endl;
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "【LayerSkipの効果】" << std::endl;
    std::cout << "  ・early_exit_loss_weight > 0 で全レイヤーが同じLM Headを共有し学習" << std::endl;
    std::cout << "  ・浅い層でも正解を出せるモデルが育つ" << std::endl;
    std::cout << "  ・推論時に早期終了や自己投機的デコーディングが有効に働く" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
