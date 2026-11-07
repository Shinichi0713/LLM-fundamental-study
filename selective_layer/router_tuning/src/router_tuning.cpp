#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>

// 簡易的な線形層（Linear Layer）
class Linear {
public:
    int in_features;
    int out_features;
    std::vector<std::vector<float>> weight; // [out_features][in_features]
    std::vector<float> bias;                  // [out_features]

    Linear(int in_f, int out_f) : in_features(in_f), out_f(out_f) {
        weight.resize(out_f, std::vector<float>(in_f, 0.01f));
        bias.resize(out_f, 0.0f);
    }

    // フォワード計算: input [in_features] -> output [out_features]
    std::vector<float> forward(const std::vector<float>& x) const {
        std::vector<float> out(out_f, 0.0f);
        for (int i = 0; i < out_f; ++i) {
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += weight[i][j] * x[j];
            }
            out[i] = sum;
        }
        return out;
    }
};

// 簡易的なエキスパート層（FFN）
class Expert {
public:
    Linear fc1;
    Linear fc2;

    Expert(int d_model, int d_ff) : fc1(d_model, d_ff), fc2(d_ff, d_model) {}

    std::vector<float> forward(const std::vector<float>& x) const {
        // ReLU 活性化関数を挟んだ 2層パーセプトロン
        std::vector<float> h = fc1.forward(x);
        for (auto& val : h) {
            val = std::max(0.0f, val); // ReLU
        }
        return fc2.forward(h);
    }
};

// Softmax関数の計算
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] /= sum;
    }
    return probs;
}

// Top-k 選択を行う構造体
struct TopKResult {
    std::vector<int> indices;
    std::vector<float> weights;
};

// 確率配列から上位 k 個を抽出する関数
TopKResult get_top_k(const std::vector<float>& probs, int k) {
    int num_experts = probs.size();
    std::vector<std::pair<float, int>> indexed_probs(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        indexed_probs[i] = {probs[i], i};
    }

    // 確率の降順にソート
    std::sort(indexed_probs.begin(), indexed_probs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    TopKResult res;
    res.indices.resize(k);
    res.weights.resize(k);
    float sum_w = 0.0f;

    for (int i = 0; i < k; ++i) {
        res.indices[i] = indexed_probs[i].second;
        res.weights[i] = indexed_probs[i].first;
        sum_w += res.weights[i];
    }

    // 重みの正規化 (sum = 1.0 にする)
    if (sum_w > 0.0f) {
        for (int i = 0; i < k; ++i) {
            res.weights[i] /= sum_w;
        }
    }

    return res;
}

// MoEレイヤー本体（ルーターチューニング・ディスパッチ処理）
class MoELayer {
public:
    int d_model;
    int num_experts;
    int top_k;
    Linear router;
    std::vector<Expert> experts;

    MoELayer(int d_model, int d_ff, int num_experts, int top_k)
        : d_model(d_model), num_experts(num_experts), top_k(top_k),
          router(d_model, num_experts) {
        for (int i = 0; i < num_experts; ++i) {
            experts.emplace_back(d_model, d_ff);
        }
    }

    // 順伝播とロードバランス損失の計算
    std::pair<std::vector<std::vector<float>>, float> forward(const std::vector<std::vector<float>>& inputs) {
        int total_tokens = inputs.size();
        std::vector<std::vector<float>> final_outputs(total_tokens, std::vector<float>(d_model, 0.0f));

        // 各エキスパートへのルーティング確率の平均を蓄積する配列 (expert_gate_load)
        std::vector<float> expert_gate_load(num_experts, 0.0f);
        // 実際に選ばれた割合を計算するためのカウンター (expert_hit_load)
        std::vector<float> expert_hit_load(num_experts, 0.0f);

        std::vector<TopKResult> all_topk(total_tokens);
        std::vector<std::vector<float>> all_probs(total_tokens);

        // 1. ルーティング計算とTop-k選出
        for (int t = 0; t < total_tokens; ++t) {
            std::vector<float> logits = router.forward(inputs[t]);
            std::vector<float> probs = softmax(logits);
            all_probs[t] = probs;

            for (int e = 0; e < num_experts; ++e) {
                expert_gate_load[e] += probs[e];
            }

            TopKResult topk = get_top_k(probs, top_k);
            all_topk[t] = topk;

            for (int k = 0; k < top_k; ++k) {
                expert_hit_load[topk.indices[k]] += 1.0f;
            }
        }

        // 平均化
        for (int e = 0; e < num_experts; ++e) {
            expert_gate_load[e] /= total_tokens;
            expert_hit_load[e] /= (total_tokens * top_k);
        }

        // 2. ロードバランス損失（Auxiliary Loss）の計算
        // Load Balance Loss = num_experts * sum(expert_gate_load * expert_hit_load)
        float load_balance_loss = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            load_balance_loss += expert_gate_load[e] * expert_hit_load[e];
        }
        load_balance_loss *= num_experts;

        // 3. 各エキスパートへのディスパッチと重み付き結合
        for (int e = 0; e < num_experts; ++e) {
            // このエキスパートを担当するトークンを集めて処理
            std::vector<float> expert_accumulated(d_model, 0.0f);
            bool has_token = false;

            for (int t = 0; t < total_tokens; ++t) {
                const auto& topk = all_topk[t];
                for (int k = 0; k < top_k; ++k) {
                    if (topk.indices[e] == e) { // 該当エキスパートの場合
                        // 実際にはトークンごとにルーティング先が異なるため、
                        // 各トークンごとにエキスパート出力を計算して重みを掛け合わせる
                    }
                }
            }
        }

        // より正確なトークンごとのルーティング処理
        for (int t = 0; t < total_tokens; ++t) {
            std::vector<float> token_out(d_model, 0.0f);
            const auto& topk = all_topk[t];

            for (int k = 0; k < top_k; ++k) {
                int expert_idx = topk.indices[k];
                float weight = topk.weights[k];

                // エキスパートの計算実行
                std::vector<float> exp_out = experts[expert_idx].forward(inputs[t]);

                // 重みを掛けて足し合わせる
                for (int d = 0; d < d_model; ++d) {
                    token_out[d] += exp_out[d] * weight;
                }
            }
            final_outputs[t] = token_out;
        }

        return {final_outputs, load_balance_loss};
    }
};

// メイン関数（動作確認用）
int main() {
    int d_model = 16;
    int d_ff = 32;
    int num_experts = 4;
    int top_k = 2;
    int batch_seq_len = 5; // バッチサイズ×シーケンス長相当のトークン数

    // MoEレイヤーの初期化
    MoELayer moe(d_model, d_ff, num_experts, top_k);

    // ダニー入力データの作成 (5トークン、各16次元)
    std::vector<std::vector<float>> inputs(batch_seq_len, std::vector<float>(d_model, 0.5f));

    // フォワード実行
    auto [outputs, aux_loss] = moe.forward(inputs);

    // 結果の出力
    std::cout << "--- MoE Router & Dispatch C++ Simulation ---" << std::endl;
    std::cout << "Load Balance Auxiliary Loss: " << std::fixed << std::setprecision(4) << aux_loss << std::endl;
    std::cout << "Output shape: (" << outputs.size() << ", " << outputs[0].size() << ")" << std::endl;
    std::cout << "First token output sample (first 4 dims): ";
    for (int i = 0; i < 4; ++i) {
        std::cout << outputs[0][i] << " ";
    }
    std::cout << std::endl;

    return 0;
}