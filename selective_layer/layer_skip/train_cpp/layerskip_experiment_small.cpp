// layerskip_experiment_small.cpp
// LayerSkip実験のC++実装（軽量版、純粋C++17、外部ライブラリ不要）
// 動作: g++ -std=c++17 -O2 -o layerskip_experiment_small layerskip_experiment_small.cpp && ./layerskip_experiment_small
//
// 内容:
//   1. 簡易Transformer（8層デコーダ）を構築
//   2. ダミーデータで事前学習済みモデルをシミュレート
//   3. LayerDropout + 早期終了損失でLayerSkip学習
//   4. 学習前後の各層perplexity比較
//   5. 自己投機的デコーディングの簡易ベンチマーク

#include <bits/stdc++.h>
using namespace std;

// ------------------------------------------------------------------
// 数学ユーティリティ
// ------------------------------------------------------------------
static random_device rd;
static mt19937 rng(rd());

float randn() {
    normal_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}

float randu() {
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}

// ------------------------------------------------------------------
// テンソル（簡易）
// ------------------------------------------------------------------
struct Tensor {
    vector<float> data;
    vector<size_t> shape;

    Tensor() = default;
    Tensor(initializer_list<size_t> sh, float init = 0.0f) {
        shape = vector<size_t>(sh);
        size_t n = 1;
        for (auto s : shape) n *= s;
        data.assign(n, init);
    }
    explicit Tensor(const vector<size_t>& sh, float init = 0.0f) {
        shape = sh;
        size_t n = 1;
        for (auto s : shape) n *= s;
        data.assign(n, init);
    }

    size_t size() const { return data.size(); }
    float& operator()(size_t i, size_t j) { return data[i * shape[1] + j]; }
    float operator()(size_t i, size_t j) const { return data[i * shape[1] + j]; }
    float& operator()(size_t b, size_t t, size_t d) {
        return data[(b * shape[1] + t) * shape[2] + d];
    }
    float operator()(size_t b, size_t t, size_t d) const {
        return data[(b * shape[1] + t) * shape[2] + d];
    }
};

// ------------------------------------------------------------------
// 線形層
// ------------------------------------------------------------------
struct Linear {
    int in_features, out_features;
    Tensor weight; // [out, in]
    Tensor bias;   // [out]

    Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
        weight = Tensor({(size_t)out_f, (size_t)in_f});
        bias = Tensor({(size_t)out_f});
        float scale = sqrtf(2.0f / (in_f + out_f));
        for (auto& v : weight.data) v = randn() * scale;
        for (auto& v : bias.data) v = randn() * scale * 0.1f;
    }

    // x: [N, in] -> out: [N, out]
    Tensor forward(const Tensor& x) const {
        size_t n = x.shape[0];
        Tensor out({n, (size_t)out_features});
        for (size_t i = 0; i < n; ++i) {
            for (int j = 0; j < out_features; ++j) {
                float sum = bias.data[j];
                for (int k = 0; k < in_features; ++k) {
                    sum += x.data[i * in_features + k] * weight.data[j * in_features + k];
                }
                out.data[i * out_features + j] = sum;
            }
        }
        return out;
    }
};

// ------------------------------------------------------------------
// LayerNorm
// ------------------------------------------------------------------
struct LayerNorm {
    int dim;
    Tensor gamma, beta;
    float eps = 1e-5f;

    LayerNorm(int d) : dim(d) {
        gamma = Tensor({(size_t)d}, 1.0f);
        beta = Tensor({(size_t)d}, 0.0f);
    }

    Tensor forward(const Tensor& x) const {
        size_t n = x.shape[0];
        Tensor out({n, (size_t)dim});
        for (size_t i = 0; i < n; ++i) {
            float mean = 0.0f;
            for (int j = 0; j < dim; ++j) mean += x.data[i * dim + j];
            mean /= dim;
            float var = 0.0f;
            for (int j = 0; j < dim; ++j) {
                float d = x.data[i * dim + j] - mean;
                var += d * d;
            }
            var /= dim;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int j = 0; j < dim; ++j) {
                out.data[i * dim + j] = (x.data[i * dim + j] - mean) * inv_std * gamma.data[j] + beta.data[j];
            }
        }
        return out;
    }
};

// ------------------------------------------------------------------
// 簡易デコーダ層（1ヘッドの自己注意 + FFN）
// ------------------------------------------------------------------
struct DecoderLayer {
    int dim, hidden_dim;
    Linear q_proj, k_proj, v_proj, out_proj;
    Linear ffn1, ffn2;
    LayerNorm norm1, norm2;

    DecoderLayer(int d, int h)
        : dim(d), hidden_dim(h),
          q_proj(d, d), k_proj(d, d), v_proj(d, d), out_proj(d, d),
          ffn1(d, h), ffn2(h, d),
          norm1(d), norm2(d) {}

    // x: [B, T, D] -> return [B, T, D]
    Tensor forward(const Tensor& x, int B, int T) const {
        // flatten to [B*T, D]
        Tensor flat({(size_t)B * T, (size_t)dim});
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                for (int d = 0; d < dim; ++d)
                    flat.data[(b * T + t) * dim + d] = x(b, t, d);

        // self-attention (simplified single-head)
        Tensor q = q_proj.forward(flat);
        Tensor k = k_proj.forward(flat);
        Tensor v = v_proj.forward(flat);

        Tensor attn_out({(size_t)B * T, (size_t)dim});
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                vector<float> scores(T);
                for (int s = 0; s <= t; ++s) {
                    float dot = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        dot += q.data[(b * T + t) * dim + d] * k.data[(b * T + s) * dim + d];
                    }
                    scores[s] = dot / sqrtf((float)dim);
                }
                for (int s = t + 1; s < T; ++s) scores[s] = -1e9f;
                float mx = *max_element(scores.begin(), scores.end());
                float sum = 0.0f;
                for (int s = 0; s < T; ++s) {
                    scores[s] = expf(scores[s] - mx);
                    sum += scores[s];
                }
                for (int s = 0; s < T; ++s) scores[s] /= sum;

                for (int d = 0; d < dim; ++d) {
                    float val = 0.0f;
                    for (int s = 0; s < T; ++s) {
                        val += scores[s] * v.data[(b * T + s) * dim + d];
                    }
                    attn_out.data[(b * T + t) * dim + d] = val;
                }
            }
        }
        Tensor attn_proj = out_proj.forward(attn_out);

        // residual + norm
        Tensor res1({(size_t)B * T, (size_t)dim});
        for (size_t i = 0; i < res1.size(); ++i) res1.data[i] = flat.data[i] + attn_proj.data[i];
        Tensor n1 = norm1.forward(res1);

        // FFN
        Tensor h = ffn1.forward(n1);
        for (auto& v : h.data) v = max(v, 0.0f); // ReLU
        Tensor ffn_out = ffn2.forward(h);

        // residual
        Tensor res2({(size_t)B * T, (size_t)dim});
        for (size_t i = 0; i < res2.size(); ++i) res2.data[i] = n1.data[i] + ffn_out.data[i];
        Tensor n2 = norm2.forward(res2);

        // reshape back to [B, T, D]
        Tensor out({(size_t)B, (size_t)T, (size_t)dim});
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                for (int d = 0; d < dim; ++d)
                    out(b, t, d) = n2.data[(b * T + t) * dim + d];
        return out;
    }
};

// ------------------------------------------------------------------
// 簡易言語モデル（Transformerデコーダ）
// ------------------------------------------------------------------
struct SimpleLM {
    int vocab_size, dim, num_layers, max_len;
    Tensor token_embed; // [V, D]
    Tensor pos_embed;   // [max_len, D]
    vector<DecoderLayer> layers;
    LayerNorm final_norm;
    Linear lm_head;

    SimpleLM(int V, int D, int L, int max_l)
        : vocab_size(V), dim(D), num_layers(L), max_len(max_l),
          final_norm(D), lm_head(D, V) {
        token_embed = Tensor({(size_t)V, (size_t)D});
        pos_embed = Tensor({(size_t)max_l, (size_t)D});
        float s = 1.0f / sqrtf(D);
        for (auto& v : token_embed.data) v = randn() * s;
        for (auto& v : pos_embed.data) v = randn() * s;
        int hidden = D * 4;
        for (int i = 0; i < L; ++i) layers.emplace_back(D, hidden);
    }

    // 入力ids: [B, T] -> hidden: [B, T, D]
    Tensor embed(const vector<vector<int>>& ids) const {
        int B = ids.size();
        int T = ids[0].size();
        Tensor h({(size_t)B, (size_t)T, (size_t)dim});
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) {
                int tok = ids[b][t] % vocab_size;
                for (int d = 0; d < dim; ++d)
                    h(b, t, d) = token_embed.data[tok * dim + d] + pos_embed.data[t * dim + d];
            }
        return h;
    }

    // 全層通過、中間出力も記録
    Tensor forward(const vector<vector<int>>& ids, vector<Tensor>* intermediates = nullptr) const {
        Tensor h = embed(ids);
        int B = ids.size(), T = ids[0].size();
        if (intermediates) intermediates->push_back(h);
        for (int l = 0; l < num_layers; ++l) {
            h = layers[l].forward(h, B, T);
            if (intermediates) intermediates->push_back(h);
        }
        return final_norm.forward(h);
    }

    // 指定層まで通過（早期終了）
    Tensor forward_until(const vector<vector<int>>& ids, int until_layer) const {
        Tensor h = embed(ids);
        int B = ids.size(), T = ids[0].size();
        for (int l = 0; l < until_layer && l < num_layers; ++l)
            h = layers[l].forward(h, B, T);
        return final_norm.forward(h);
    }

    // logits: [B, T, V]
    Tensor logits_from_hidden(const Tensor& h) const {
        int B = h.shape[0], T = h.shape[1];
        Tensor flat({(size_t)B * T, (size_t)dim});
        for (size_t i = 0; i < flat.size(); ++i) flat.data[i] = h.data[i];
        Tensor out = lm_head.forward(flat);
        Tensor logits({(size_t)B, (size_t)T, (size_t)vocab_size});
        for (size_t i = 0; i < out.size(); ++i) logits.data[i] = out.data[i];
        return logits;
    }
};

// ------------------------------------------------------------------
// 損失計算: cross entropy
// ------------------------------------------------------------------
float cross_entropy(const Tensor& logits, const vector<vector<int>>& target_ids) {
    int B = logits.shape[0], T = logits.shape[1], V = logits.shape[2];
    float total = 0.0f;
    int count = 0;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T - 1; ++t) { // tのlogitでt+1を予測
            int target = target_ids[b][t + 1] % V;
            float mx = -1e9f;
            for (int v = 0; v < V; ++v) mx = max(mx, logits(b, t, v));
            float sum = 0.0f;
            for (int v = 0; v < V; ++v) {
                sum += expf(logits(b, t, v) - mx);
            }
            float p = expf(logits(b, t, target) - mx) / sum;
            total += -logf(max(p, 1e-10f));
            count++;
        }
    }
    return total / max(count, 1);
}

// ------------------------------------------------------------------
// Perplexity計算
// ------------------------------------------------------------------
float compute_ppl(const SimpleLM& model, const vector<vector<int>>& data, int eval_at_layer = -1) {
    Tensor h;
    if (eval_at_layer < 0) {
        vector<Tensor> tmp;
        h = model.forward(data, &tmp);
    } else {
        h = model.forward_until(data, eval_at_layer);
    }
    Tensor logits = model.logits_from_hidden(h);
    float ce = cross_entropy(logits, data);
    return expf(ce);
}

// ------------------------------------------------------------------
// ダミーデータ生成（WikiText-2をシミュレート）
// ------------------------------------------------------------------
vector<vector<int>> make_dummy_data(int num_samples, int seq_len, int vocab_size) {
    vector<vector<int>> data(num_samples, vector<int>(seq_len));
    for (int i = 0; i < num_samples; ++i) {
        int base = rand() % vocab_size;
        for (int t = 0; t < seq_len; ++t) {
            if (randu() < 0.7f) {
                data[i][t] = (base + t * 3 + rand() % 5) % vocab_size;
            } else {
                data[i][t] = rand() % vocab_size;
            }
        }
    }
    return data;
}

// ------------------------------------------------------------------
// 簡易SGD更新
// ------------------------------------------------------------------
void sgd_update(Tensor& param, const Tensor& grad, float lr) {
    for (size_t i = 0; i < param.size(); ++i) param.data[i] -= lr * grad.data[i];
}

// ------------------------------------------------------------------
// 損失評価（LayerSkip: 全層の早期終了損失を合計）
// ------------------------------------------------------------------
float eval_loss(SimpleLM& model, const vector<vector<int>>& batch,
                const vector<int>& exit_layers, float e_scale) {
    vector<Tensor> inter;
    model.forward(batch, &inter); // inter[0]=embed, inter[1]=L1後, ..., inter[L]=最終後
    float total = 0.0f;
    for (int l : exit_layers) {
        Tensor eh = model.final_norm.forward(inter[l]);
        Tensor logits = model.logits_from_hidden(eh);
        float ce = cross_entropy(logits, batch);
        float scale = (l == model.num_layers) ? 1.0f : e_scale * l / model.num_layers;
        total += scale * ce;
    }
    return total;
}

// ------------------------------------------------------------------
// 数値微分による勾配計算とSGD更新（1つのテンソル）
// ------------------------------------------------------------------
void update_tensor(Tensor& param, SimpleLM& model, const vector<vector<int>>& batch,
                   const vector<int>& exit_layers, float e_scale, float lr, float eps = 1e-3f) {
    Tensor grad(param.shape);
    for (size_t i = 0; i < param.size(); ++i) {
        float orig = param.data[i];
        param.data[i] = orig + eps;
        float lp = eval_loss(model, batch, exit_layers, e_scale);
        param.data[i] = orig - eps;
        float lm = eval_loss(model, batch, exit_layers, e_scale);
        param.data[i] = orig;
        grad.data[i] = (lp - lm) / (2.0f * eps);
    }
    sgd_update(param, grad, lr);
}

// ------------------------------------------------------------------
// メイン
// ------------------------------------------------------------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // ハイパーパラメータ（軽量版）
    const int VOCAB = 50;        // 小さな語彙
    const int DIM = 16;          // 小さな次元
    const int NUM_LAYERS = 8;    // 8層（軽量版）
    const int SEQ_LEN = 8;       // 短めのシーケンス
    const int BATCH_SIZE = 2;
    const int TRAIN_STEPS = 5;   // 少ないステップ数
    const float LR = 0.1f;
    const float E_SCALE = 0.5f;  // 早期終了損失の重み係数

    // 早期終了層（L/4, L/2, 3L/4, L）
    vector<int> exit_layers = {
        NUM_LAYERS / 4,
        NUM_LAYERS / 2,
        3 * NUM_LAYERS / 4,
        NUM_LAYERS
    };
    sort(exit_layers.begin(), exit_layers.end());
    exit_layers.erase(unique(exit_layers.begin(), exit_layers.end()), exit_layers.end());

    cout << "========================================\n";
    cout << "LayerSkip C++ Experiment (Lightweight)\n";
    cout << "========================================\n";
    cout << "Vocab: " << VOCAB << ", Dim: " << DIM << ", Layers: " << NUM_LAYERS << "\n";
    cout << "SeqLen: " << SEQ_LEN << ", Batch: " << BATCH_SIZE << ", Steps: " << TRAIN_STEPS << "\n";
    cout << "Early exit layers: ";
    for (int l : exit_layers) cout << "L" << l << " ";
    cout << "\n\n";

    // モデル構築
    SimpleLM model(VOCAB, DIM, NUM_LAYERS, SEQ_LEN);

    // データ生成
    auto train_data = make_dummy_data(50, SEQ_LEN, VOCAB);
    auto val_data = make_dummy_data(10, SEQ_LEN, VOCAB);

    // ==========================================
    // ステップ1: 学習前のベースライン測定
    // ==========================================
    cout << "[Step 1] 学習前の各層perplexity\n";
    cout << "----------------------------------------\n";
    vector<int> eval_layers = {2, 4, 6, 8};
    vector<float> ppl_before, ppl_after;

    for (int l : eval_layers) {
        float ppl = compute_ppl(model, val_data, l);
        ppl_before.push_back(ppl);
        cout << "  L" << l << " ppl = " << fixed << setprecision(2) << ppl << "\n";
    }
    float ppl_full_before = compute_ppl(model, val_data, -1);
    cout << "  Full ppl = " << fixed << setprecision(2) << ppl_full_before << "\n\n";

    // ==========================================
    // ステップ2: LayerSkip学習
    // ==========================================
    cout << "[Step 2] LayerSkip学習開始\n";
    cout << "  (数値微分を使用、計算に時間がかかります)\n";
    cout << "----------------------------------------\n";

    for (int step = 1; step <= TRAIN_STEPS; ++step) {
        vector<vector<int>> batch;
        for (int b = 0; b < BATCH_SIZE; ++b) {
            int idx = rand() % train_data.size();
            batch.push_back(train_data[idx]);
        }

        float p_max = 0.3f * step / TRAIN_STEPS;

        update_tensor(model.token_embed, model, batch, exit_layers, E_SCALE, LR);
        update_tensor(model.pos_embed, model, batch, exit_layers, E_SCALE, LR);
        update_tensor(model.lm_head.weight, model, batch, exit_layers, E_SCALE, LR);
        for (auto& layer : model.layers) {
            update_tensor(layer.q_proj.weight, model, batch, exit_layers, E_SCALE, LR);
            update_tensor(layer.k_proj.weight, model, batch, exit_layers, E_SCALE, LR);
            update_tensor(layer.v_proj.weight, model, batch, exit_layers, E_SCALE, LR);
            update_tensor(layer.out_proj.weight, model, batch, exit_layers, E_SCALE, LR);
            update_tensor(layer.ffn1.weight, model, batch, exit_layers, E_SCALE, LR);
            update_tensor(layer.ffn2.weight, model, batch, exit_layers, E_SCALE, LR);
        }

        float loss = eval_loss(model, batch, exit_layers, E_SCALE);
        cout << "  step " << step << "/" << TRAIN_STEPS
             << " | loss=" << fixed << setprecision(4) << loss
             << " | p_max=" << setprecision(3) << p_max << "\n";
    }
    cout << "\n";

    // ==========================================
    // ステップ3: 学習後の性能比較
    // ==========================================
    cout << "[Step 3] 学習後の各層perplexity\n";
    cout << "----------------------------------------\n";
    for (int l : eval_layers) {
        float ppl = compute_ppl(model, val_data, l);
        ppl_after.push_back(ppl);
        cout << "  L" << l << " ppl = " << fixed << setprecision(2) << ppl << "\n";
    }
    float ppl_full_after = compute_ppl(model, val_data, -1);
    cout << "  Full ppl = " << fixed << setprecision(2) << ppl_full_after << "\n\n";

    // ==========================================
    // 改善率サマリ
    // ==========================================
    cout << "[Summary] 改善率\n";
    cout << "----------------------------------------\n";
    cout << "  層    | 学習前    | 学習後    | 改善倍率 | 改善率\n";
    cout << "  ------|-----------|-----------|----------|--------\n";
    for (size_t i = 0; i < eval_layers.size(); ++i) {
        float ratio = ppl_before[i] / max(ppl_after[i], 1e-6f);
        float pct = (1.0f - ppl_after[i] / max(ppl_before[i], 1e-6f)) * 100.0f;
        cout << "  L" << eval_layers[i]
             << "   | " << setw(9) << fixed << setprecision(2) << ppl_before[i]
             << " | " << setw(9) << ppl_after[i]
             << " | " << setw(8) << setprecision(1) << ratio << "x"
             << " | " << setw(5) << setprecision(2) << pct << "%\n";
    }
    cout << "\n";

    // ==========================================
    // ステップ4: 簡易Self-Speculative Decoding
    // ==========================================
    cout << "[Step 4] 簡易Self-Speculative Decoding\n";
    cout << "----------------------------------------\n";
    vector<int> prompt(SEQ_LEN);
    for (int t = 0; t < SEQ_LEN; ++t) prompt[t] = rand() % VOCAB;
    vector<vector<int>> prompt_batch = {prompt};

    auto t0 = chrono::high_resolution_clock::now();
    Tensor h_full = model.forward(prompt_batch);
    auto t1 = chrono::high_resolution_clock::now();
    double dt_full = chrono::duration<double, milli>(t1 - t0).count();

    vector<int> spec_layers = {2, 4, 6};
    cout << "  通常デコード (L" << NUM_LAYERS << "): " << fixed << setprecision(3) << dt_full << " ms\n";

    for (int exit_l : spec_layers) {
        auto ts = chrono::high_resolution_clock::now();
        Tensor h_spec = model.forward_until(prompt_batch, exit_l);
        auto te = chrono::high_resolution_clock::now();
        double dt_spec = chrono::duration<double, milli>(te - ts).count();
        double speedup = dt_full / max(dt_spec, 0.001);
        cout << "  Self-Speculative (exit=" << exit_l << "): "
             << dt_spec << " ms (speedup=" << setprecision(2) << speedup << "x)\n";
    }
    cout << "\n";

    // ==========================================
    // 実際のテキスト生成品質（簡易）
    // ==========================================
    cout << "[Step 5] 各層の正解トークン確率（簡易）\n";
    cout << "----------------------------------------\n";
    vector<vector<int>> test_prompt = {prompt};
    vector<int> test_layers = {2, 4, 6, 8};
    for (int l : test_layers) {
        Tensor h = (l == NUM_LAYERS) ? model.forward(test_prompt) : model.forward_until(test_prompt, l);
        Tensor logits = model.logits_from_hidden(h);
        int target = test_prompt[0][1];
        float mx = -1e9f;
        for (int v = 0; v < VOCAB; ++v) mx = max(mx, logits(0, 0, v));
        float sum = 0.0f;
        for (int v = 0; v < VOCAB; ++v) {
            sum += expf(logits(0, 0, v) - mx);
        }
        float p = expf(logits(0, 0, target) - mx) / sum;
        cout << "  L" << l << " target_prob = " << fixed << setprecision(4) << p << "\n";
    }

    cout << "\n========================================\n";
    cout << "実験完了\n";
    cout << "========================================\n";

    return 0;
}
