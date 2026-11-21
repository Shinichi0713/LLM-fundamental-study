#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>

// ==========================================
// 1. Configuration & Constants (移植)
// ==========================================
struct DynaBertConfig {
    int vocab_size = 30522;
    int hidden_size = 768;
    int num_hidden_layers = 6; // 実験用に短縮
    int num_attention_heads = 12;
    int intermediate_size = 3072;
    double hidden_dropout_prob = 0.1;
    double attention_probs_dropout_prob = 0.1;
    int max_position_embeddings = 512;
    int type_vocab_size = 2;
    double initializer_range = 0.02;
    
    // DynaBERT specific
    std::vector<double> width_mult = {1.0, 0.5}; // サポートする幅
    int num_labels = 2;
};

// 再現性のためのSeed固定 (移植)
void set_seed(int seed) {
    torch::manual_seed(seed);
    // C++のrandomも固定
    std::srand(seed);
    // GPU環境で再現性を高める設定 (今回はCPUのみを想定)
    // torch::cuda::manual_seed_all(seed);
}

// ==========================================
// 2. Tiny BERT Implementation (C++ Port)
// ==========================================

// BERT Embeddings (移植)
class BertEmbeddingsImpl : public torch::nn::Module {
public:
    BertEmbeddingsImpl(const DynaBertConfig& config)
        : word_embeddings(torch::nn::Embedding(torch::nn::EmbeddingOptions(config.vocab_size, config.hidden_size))),
          position_embeddings(torch::nn::Embedding(torch::nn::EmbeddingOptions(config.max_position_embeddings, config.hidden_size))),
          token_type_embeddings(torch::nn::Embedding(torch::nn::EmbeddingOptions(config.type_vocab_size, config.hidden_size))),
          LayerNorm(torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.hidden_size}).eps(1e-12))),
          dropout(torch::nn::Dropout(torch::nn::DropoutOptions(config.hidden_dropout_prob))) {
        
        register_module("word_embeddings", word_embeddings);
        register_module("position_embeddings", position_embeddings);
        register_module("token_type_embeddings", token_type_embeddings);
        register_module("LayerNorm", LayerNorm);
        register_module("dropout", dropout);
        
        // 重み初期化 (Xavierに近いものを手動で設定)
        torch::nn::init::normal_(word_embeddings->weight, 0.0, config.initializer_range);
        torch::nn::init::normal_(position_embeddings->weight, 0.0, config.initializer_range);
        torch::nn::init::normal_(token_type_embeddings->weight, 0.0, config.initializer_range);
    }

    torch::Tensor forward(torch::Tensor input_ids) {
        int64_t seq_length = input_ids.size(1);
        torch::Tensor position_ids = torch::arange(seq_length, input_ids.options().dtype(torch::kLong));
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids);
        
        torch::Tensor token_type_ids = torch::zeros_like(input_ids);

        torch::Tensor words_embeddings = word_embeddings(input_ids);
        torch::Tensor pos_embeddings = position_embeddings(position_ids);
        torch::Tensor tk_embeddings = token_type_embeddings(token_type_ids);

        torch::Tensor embeddings = words_embeddings + pos_embeddings + tk_embeddings;
        embeddings = LayerNorm(embeddings);
        embeddings = dropout(embeddings);
        return embeddings;
    }

private:
    torch::nn::Embedding word_embeddings;
    torch::nn::Embedding position_embeddings;
    torch::nn::Embedding token_type_embeddings;
    torch::nn::LayerNorm LayerNorm;
    torch::nn::Dropout dropout;
};
TORCH_MODULE(BertEmbeddings);

// BERT Self-Attention (C++ Port)
// 動的なヘッドサイズ計算ロジックを含む
class ElasticBertSelfAttentionImpl : public torch::nn::Module {
public:
    ElasticBertSelfAttentionImpl(const DynaBertConfig& config, int current_hidden_size, int current_heads) {
        if (current_hidden_size % current_heads != 0) {
            throw std::runtime_error("Hidden size must be divisible by number of attention heads");
        }
        
        num_attention_heads = current_heads;
        attention_head_size = current_hidden_size / current_heads;
        int all_head_size = current_heads * attention_head_size;

        // PyTorch C++ではオプションオブジェクトを使ってコンストラクタを記述
        query = register_module("query", torch::nn::Linear(torch::nn::LinearOptions(current_hidden_size, all_head_size)));
        key = register_module("key", torch::nn::Linear(torch::nn::LinearOptions(current_hidden_size, all_head_size)));
        value = register_module("value", torch::nn::Linear(torch::nn::LinearOptions(current_hidden_size, all_head_size)));
        dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(config.attention_probs_dropout_prob)));
        
        // Weight Initialization
        torch::nn::init::normal_(query->weight, 0.0, config.initializer_range);
        torch::nn::init::normal_(key->weight, 0.0, config.initializer_range);
        torch::nn::init::normal_(value->weight, 0.0, config.initializer_range);
    }

    std::vector<torch::Tensor> transpose_for_scores(torch::Tensor x) {
        // x形状: [batch, seq_len, all_head_size]
        auto new_x_shape = x.sizes().vec();
        new_x_shape.pop_back(); // 最後の次元を削除
        new_x_shape.insert(new_x_shape.end(), {num_attention_heads, attention_head_size});
        
        // x.view の代わりに x.reshape を使用
        x = x.reshape(new_x_shape);
        // 形状: [batch, seq_len, num_heads, head_size]
        
        // PyTorch transpose: (dim1, dim2)
        return {x.permute({0, 2, 1, 3})}; 
        // 形状: [batch, num_heads, seq_len, head_size]
    }

    torch::Tensor forward(torch::Tensor hidden_states, torch::optional<torch::Tensor> attention_mask = torch::nullopt) {
        torch::Tensor mixed_query_layer = query(hidden_states);
        torch::Tensor mixed_key_layer = key(hidden_states);
        torch::Tensor mixed_value_layer = value(hidden_states);

        // query_layer.view 相当の reshape を transpose_for_scores で実行
        // Note: C++の関数戻り値はPythonのようなアンパックができないので手動で展開
        torch::Tensor query_layer = transpose_for_scores(mixed_query_layer)[0];
        torch::Tensor key_layer = transpose_for_scores(mixed_key_layer)[0];
        torch::Tensor value_layer = transpose_for_scores(mixed_value_layer)[0];

        // Take the dot product between "query" and "key" to get the raw attention scores.
        // key_layer.transpose(-1, -2) 相当
        torch::Tensor attention_scores = torch::matmul(query_layer, key_layer.transpose(-2, -1));
        attention_scores = attention_scores / std::sqrt(attention_head_size);
        if (attention_mask.has_value()) {
            attention_scores = attention_scores + attention_mask.value();
        }

        // Normalize the attention scores to probabilities.
        // dim=-1 は C++では dim=-1 または dim(3) と指定
        torch::Tensor attention_probs = torch::softmax(attention_scores, -1);

        attention_probs = dropout(attention_probs);

        torch::Tensor context_layer = torch::matmul(attention_probs, value_layer);

        // context_layer.permute({0, 2, 1, 3}).contiguous()
        context_layer = context_layer.permute({0, 2, 1, 3}).contiguous();
        
        // new_context_layer_shape (shape[:-2] + (all_head_size,))
        auto current_shape = context_layer.sizes().vec();
        int64_t batch = current_shape[0];
        int64_t seq_len = current_shape[2];
        int64_t all_head_size = num_attention_heads * attention_head_size;
        
        context_layer = context_layer.reshape({batch, seq_len, all_head_size});

        return context_layer;
    }

private:
    torch::nn::Linear query{nullptr}, key{nullptr}, value{nullptr};
    torch::nn::Dropout dropout{nullptr};
    int num_attention_heads;
    int attention_head_size;
};
TORCH_MODULE(ElasticBertSelfAttention);

// BERT Output Path (SelfOutput + Intermediate + Output) (移植)
class ElasticBertLayerImpl : public torch::nn::Module {
public:
    ElasticBertLayerImpl(const DynaBertConfig& config, int current_hidden_size, int current_intermediate_size, int current_heads)
        : self_attention(config, current_hidden_size, current_heads),
          self_output_dense(torch::nn::Linear(torch::nn::LinearOptions(current_heads * (current_hidden_size / current_heads), current_hidden_size))),
          self_output_LayerNorm(torch::nn::LayerNorm(torch::nn::LayerNormOptions({current_hidden_size}).eps(1e-12))),
          self_output_dropout(torch::nn::Dropout(torch::nn::DropoutOptions(config.hidden_dropout_prob))),
          intermediate_dense(torch::nn::Linear(torch::nn::LinearOptions(current_hidden_size, current_intermediate_size))),
          output_dense2(torch::nn::Linear(torch::nn::LinearOptions(current_intermediate_size, current_hidden_size))),
          output_LayerNorm2(torch::nn::LayerNorm(torch::nn::LayerNormOptions({current_hidden_size}).eps(1e-12))),
          output_dropout2(torch::nn::Dropout(torch::nn::DropoutOptions(config.hidden_dropout_prob))) {
        
        register_module("self_attention", self_attention);
        register_module("self_output_dense", self_output_dense);
        register_module("self_output_LayerNorm", self_output_LayerNorm);
        register_module("self_output_dropout", self_output_dropout);
        register_module("intermediate_dense", intermediate_dense);
        register_module("output_dense2", output_dense2);
        register_module("output_LayerNorm2", output_LayerNorm2);
        register_module("output_dropout2", output_dropout2);

        // 重み初期化
        torch::nn::init::normal_(self_output_dense->weight, 0.0, config.initializer_range);
        torch::nn::init::normal_(intermediate_dense->weight, 0.0, config.initializer_range);
        torch::nn::init::normal_(output_dense2->weight, 0.0, config.initializer_range);
    }

    torch::Tensor forward(torch::Tensor hidden_states, torch::optional<torch::Tensor> attention_mask = torch::nullopt) {
        // Attention
        torch::Tensor attention_output = self_attention(hidden_states, attention_mask);
        attention_output = self_output_dense(attention_output);
        attention_output = self_output_dropout(attention_output);
        // LayerNorm (入力 + 出力)
        torch::Tensor layer_input = hidden_states + attention_output;
        torch::Tensor layernorm_output1 = self_output_LayerNorm(layer_input);
        
        // Intermediate (MLP)
        torch::Tensor intermediate_output = intermediate_dense(layernorm_output1);
        // GELU
        intermediate_output = torch::gelu(intermediate_output);
        
        // Output Dense2
        torch::Tensor layer_output = output_dense2(intermediate_output);
        layer_output = output_dropout2(layer_output);
        
        // LayerNorm2 (LayerNorm1の出力 + dense2の出力)
        torch::Tensor second_input = layernorm_output1 + layer_output;
        torch::Tensor final_output = output_LayerNorm2(second_input);
        
        return final_output;
    }

private:
    ElasticBertSelfAttention self_attention;
    torch::nn::Linear self_output_dense, intermediate_dense, output_dense2;
    torch::nn::LayerNorm self_output_LayerNorm, output_LayerNorm2;
    torch::nn::Dropout self_output_dropout, output_dropout2;
};
TORCH_MODULE(ElasticBertLayer);

// BERT Encoder (LayerList) (移植)
class ElasticBertEncoderImpl : public torch::nn::Module {
public:
    ElasticBertEncoderImpl(const DynaBertConfig& config, int current_hidden_size, int current_intermediate_size, int current_heads) {
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            // C++のModuleListへの登録方法: layers.push_back(register_module("layer" + std::to_string(i), ...))
            auto layer = register_module("layer" + std::to_string(i), ElasticBertLayer(config, current_hidden_size, current_intermediate_size, current_heads));
            layers.push_back(layer);
        }
    }

    std::vector<torch::Tensor> forward(torch::Tensor hidden_states, torch::optional<torch::Tensor> attention_mask = torch::nullopt) {
        std::vector<torch::Tensor> all_hidden_states;
        for (auto& layer_module : layers) {
            hidden_states = layer_module->forward(hidden_states, attention_mask);
        }
        // ここでは簡単のため、最後のhidden stateのみを返す実装にする
        // (Pythonコードの蒸留ロジックに合わせてすべての隠れ層状態を返す必要がある場合は、all_hidden_statesに追加していく)
        all_hidden_states.push_back(hidden_states); 
        return all_hidden_states;
    }

private:
    std::vector<ElasticBertLayer> layers;
};
TORCH_MODULE(ElasticBertEncoder);

// TinyBert For Sequence Classification (移植)
class ElasticBertForSequenceClassificationImpl : public torch::nn::Module {
public:
    ElasticBertForSequenceClassificationImpl(const DynaBertConfig& config, double width_mult)
        : current_width_mult(width_mult) {
        
        // 幅に応じたサイズ計算
        cur_h = static_cast<int>(config.hidden_size * width_mult);
        cur_i = static_cast<int>(config.intermediate_size * width_mult);
        cur_a = std::max(1, static_cast<int>(config.num_attention_heads * width_mult));
        
        embeddings = register_module("embeddings", BertEmbeddings(config));
        
        // 埋め込み次元を合わせるためのProjection layer (width < 1.0の場合)
        if (cur_h != config.hidden_size) {
            embed_proj = register_module("embed_proj", torch::nn::Linear(torch::nn::LinearOptions(config.hidden_size, cur_h)));
            torch::nn::init::normal_(embed_proj->weight, 0.0, config.initializer_range);
        } else {
            embed_proj = register_module("embed_proj", torch::nn::Identity());
        }

        bert_encoder = register_module("bert_encoder", ElasticBertEncoder(config, cur_h, cur_i, cur_a));
        dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(config.hidden_dropout_prob)));
        classifier = register_module("classifier", torch::nn::Linear(torch::nn::LinearOptions(cur_h, config.num_labels)));
        
        torch::nn::init::normal_(classifier->weight, 0.0, config.initializer_range);
    }

    std::vector<torch::Tensor> forward(torch::Tensor input_ids) {
        torch::Tensor embedding_output = embeddings(input_ids);
        embedding_output = embed_proj->as<torch::nn::Identity>()->is_enabled() ? 
                            embedding_output : embed_proj->as<torch::nn::Linear>()->forward(embedding_output);
                            // Note: C++では as<T>() を使ってダウンキャストするが、IdentityとLinearは別物なので
                            // ここはもう少しスマートな方法が必要。簡単のためif文で分ける。

        // 修正: if文で安全に実行
        torch::Tensor projected_embeddings;
        if (cur_h != 30522 /* config.vocab_size doesn't have hidden_size, let's use hardcoded value for conceptual simplicity */ 
            /* Actually, a better check is using the module type: dynamic_cast */
        ) {
            // 上記のキャストロジックが複雑になるため、別のスマートな実装方法を採用
        }
        // --- 改良版フォワード ---
        return forward_simplified(input_ids);
    }

    // コンセプトを明確にするためのフォワード
    std::vector<torch::Tensor> forward_simplified(torch::Tensor input_ids) {
        torch::Tensor embedding_output = embeddings(input_ids);
        
        // 埋め込み次元の投影 (width < 1.0の場合)
        torch::Tensor hidden_states;
        if (cur_h != 768 /* config.hidden_size */) {
             // register_moduleの戻り値はModulePtrなのでキャストが必要
             // ここではキャストが複雑になるため、コンストラクタでキャストしておく
             hidden_states = cached_proj_linear->as<torch::nn::Linear>()->forward(embedding_output);
        } else {
             hidden_states = embedding_output; // Identity
        }
        
        std::vector<torch::Tensor> encoder_outputs = bert_encoder->forward(hidden_states);
        torch::Tensor last_hidden_state = encoder_outputs.back();
        
        // [CLS] token (batch, 0, hidden_size)
        torch::Tensor pooled_output = last_hidden_state.index({torch::indexing::Slice(), 0});
        pooled_output = dropout(pooled_output);
        torch::Tensor logits = classifier(pooled_output);
        
        return {logits}; // (lossはTrainerで計算するため返さない)
    }

    // コンストラクタでダウンキャストを保存しておく
    void post_init_casts() {
        if (embed_proj->name() == "torch::nn::LinearImpl") {
            cached_proj_linear = embed_proj;
        }
    }

private:
    int cur_h, cur_i, cur_a;
    double current_width_mult;
    BertEmbeddings embeddings{nullptr};
    torch::nn::ModuleHolder embed_proj{nullptr}; 
    torch::nn::ModuleHolder cached_proj_linear{nullptr}; // Linearへのキャストキャッシュ
    ElasticBertEncoder bert_encoder{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear classifier{nullptr};
};
// 上記のキャスト問題によりModulePtrの定義が難しいため、簡易的な実装にする
// コンセプトを追うため、以下のラッパークラスでキャスト問題を回避します。
class ElasticBertForSequenceClassification : public torch::nn::Module {
public:
    ElasticBertForSequenceClassification(const DynaBertConfig& config, double wm) 
        : config_(config), wm_(wm) {
        
        cur_h = static_cast<int>(config.hidden_size * wm);
        cur_i = static_cast<int>(config.intermediate_size * wm);
        cur_a = std::max(1, static_cast<int>(config.num_attention_heads * wm));

        embeddings = register_module("embeddings", BertEmbeddings(config));
        if (cur_h != config.hidden_size) {
            embed_proj_linear = register_module("embed_proj_linear", torch::nn::Linear(torch::nn::LinearOptions(config.hidden_size, cur_h)));
            torch::nn::init::normal_(embed_proj_linear->weight, 0.0, config.initializer_range);
        }
        bert_encoder = register_module("bert_encoder", ElasticBertEncoder(config, cur_h, cur_i, cur_a));
        dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(config.hidden_dropout_prob)));
        classifier = register_module("classifier", torch::nn::Linear(torch::nn::LinearOptions(cur_h, config.num_labels)));
        torch::nn::init::normal_(classifier->weight, 0.0, config.initializer_range);
    }

    std::vector<torch::Tensor> forward(torch::Tensor input_ids) {
        torch::Tensor embedding_output = embeddings(input_ids);
        torch::Tensor hidden_states;
        if (embed_proj_linear) {
            hidden_states = embed_proj_linear->forward(embedding_output);
        } else {
            hidden_states = embedding_output;
        }
        
        std::vector<torch::Tensor> encoder_outputs = bert_encoder->forward(hidden_states);
        torch::Tensor last_hidden_state = encoder_outputs.back();
        torch::Tensor pooled_output = last_hidden_state.index({torch::indexing::Slice(), 0});
        pooled_output = dropout(pooled_output);
        torch::Tensor logits = classifier(pooled_output);
        return {logits};
    }

private:
    DynaBertConfig config_;
    double wm_;
    int cur_h, cur_i, cur_a;
    BertEmbeddings embeddings{nullptr};
    torch::nn::Linear embed_proj_linear{nullptr};
    ElasticBertEncoder bert_encoder{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear classifier{nullptr};
};

// ==========================================
// 5. Knowledge Distillation Logic (C++ Port)
// ==========================================

// Weight Extraction (Teacher to Student) (移植)
void weight_extraction(std::shared_ptr<ElasticBertForSequenceClassification> teacher_model,
                       std::shared_ptr<ElasticBertForSequenceClassification> student_model,
                       const DynaBertConfig& config) {
    
    // Pythonの state_dict() 相当の重みマップを取得
    torch::OrderedDict<std::string, torch::Tensor> teacher_params = teacher_model->named_parameters();
    torch::OrderedDict<std::string, torch::Tensor> student_params = student_model->named_parameters();
    
    // no_grad コンテキスト相当
    torch::NoGradGuard no_grad;
    
    for (auto const& item : student_params.items()) {
        const std::string& s_name = item.key();
        torch::Tensor& s_param = item.value();
        
        // Linear層の重みやバイアスであることを確認
        // 名前が ".weight" または ".bias" で終わるもの
        if (s_name.find(".weight") != std::string::npos || s_name.find(".bias") != std::string::npos) {
            
            // Teacher側にも対応するパラメータがあるか確認
            if (teacher_params.contains(s_name)) {
                torch::Tensor t_param = teacher_params[s_name];
                
                // 次元が異なる場合（width prunning）、Teacher側をスライスして合わせる
                if (!s_param.sizes().equals(t_param.sizes())) {
                    std::cout << "Extracting and slicing weight: " << s_name << " S:" << s_param.sizes() << " T:" << t_param.sizes() << std::endl;
                    
                    // 単純な前方スライス (重要度ソートは省略)
                    // Pythonの index_select(dim, indices) 相当を C++で
                    std::vector<torch::indexing::TensorIndex> slices;
                    for (size_t i = 0; i < s_param.dim(); ++i) {
                        int64_t s_dim = s_param.sizes()[i];
                        int64_t t_dim = t_param.sizes()[i];
                        if (s_dim < t_dim) {
                            // Teacherの次元の方が大きい場合、スライスする
                            slices.push_back(torch::indexing::Slice(0, s_dim));
                        } else {
                            slices.push_back(torch::indexing::Slice()); // 全て
                        }
                    }
                    // slice_backward は inplace操作ではないので、copy_(src) で代入
                    s_param.copy_(t_param.index(slices));
                } else {
                    // 次元が同じ場合はそのままコピー
                    // std::cout << "Extracting weight: " << s_name << std::endl;
                    s_param.copy_(t_param);
                }
            } else {
                 std::cout << "Warning: Parameter " << s_name << " not found in teacher." << std::endl;
            }
        }
    }
    std::cout << "Weights extracted from Teacher to Student." << std::endl;
}

// Training Step (蒸留ループ) (移植)
double train_dynabert_step(std::shared_ptr<ElasticBertForSequenceClassification> teacher,
                           std::shared_ptr<ElasticBertForSequenceClassification> student,
                           torch::optim::Optimizer& optimizer,
                           torch::Tensor input_ids,
                           torch::Tensor labels,
                           double temperature = 2.0,
                           double alpha = 0.5,
                           double beta = 0.5) {
    
    student->train();
    teacher->eval();
    
    optimizer.zero_grad();
    
    // 1. Teacher Forward (NoGradGuardはグローバル設定でも可能ですが、明示的に)
    std::vector<torch::Tensor> t_logits_vec;
    {
        torch::NoGradGuard no_grad;
        t_logits_vec = teacher->forward(input_ids);
    }
    torch::Tensor t_logits = t_logits_vec[0];
    
    // 2. Student Forward
    std::vector<torch::Tensor> s_logits_vec = student->forward(input_ids);
    torch::Tensor s_logits = s_logits_vec[0];
    
    // 3. Loss Calculation
    
    // Hard Label Loss (CE) (C++では nll_loss や cross_entropy)
    // 注意: CrossEntropyLossは C++では log_softmax を含んだものがデフォルト
    torch::Tensor loss_hard = torch::cross_entropy_loss(s_logits, labels);
    
    // Soft Label Loss (KL Divergence on logits)
    // KLDivLoss(reduction='batchmean') を使用
    // p = log_softmax(s/T), q = softmax(t/T)
    auto kl_loss = torch::nn::KLDivLoss(torch::nn::KLDivLossOptions().reduction(torch::nn::KLEmbeddingReduction::BatchMean));
    
    torch::Tensor p = torch::log_softmax(s_logits / temperature, 1);
    torch::Tensor q = torch::softmax(t_logits / temperature, 1);
    
    // KLDivLoss(log_target=False) の場合、入力は log space, ターゲットは probability space である必要がある
    torch::Tensor loss_soft = kl_loss->forward(p, q);
    
    // 総合Loss (α: Soft, β: Hard) - ハイパーパラメータは適当
    // DynaBERTでは温度の2乗を掛ける
    torch::Tensor loss = alpha * (loss_soft * (temperature * temperature)) + beta * loss_hard;
    
    loss.backward();
    optimizer.step();
    
    return loss.item<double>();
}

// ==========================================
// 6. Main Execution Flow (C++ Port)
// ==========================================

int main() {
    // Seed固定
    set_seed(42);
    
    // Configuration (移植)
    DynaBertConfig config;
    config.num_hidden_layers = 3; // 実験用にさらに短縮
    config.hidden_size = 128; // 実験用に小さく
    config.num_attention_heads = 4;
    config.intermediate_size = 512;

    // デバイス設定 (CPU)
    torch::Device device(torch::kCPU);
    std::cout << "Device: " << device.str() << std::endl;
    
    // 1. Create Teacher (width_mult=1.0) (移植)
    // C++では std::make_shared を使うのが一般的
    auto teacher = std::make_shared<ElasticBertForSequenceClassification>(config, 1.0);
    teacher->to(device);
    // (実際には事前学習重みをロードするが、今回はランダム初期化のまま)

    // 2. Create Student Candidates (移植)
    std::map<double, std::shared_ptr<ElasticBertForSequenceClassification>> students;
    // 今回サポートするもう一つの幅
    double target_wm = 0.5;
    auto student = std::make_shared<ElasticBertForSequenceClassification>(config, target_wm);
    student->to(device);
    students[target_wm] = student;
            
    // 3. Dummy Dataset (Random data for classification) (移植)
    int64_t batch_size = 8;
    int64_t num_batches = 20; // エポックあたりのバッチ数
    int64_t seq_len = 64;
    
    // オプティマイザーの定義 (代表して1つのStudent用)
    torch::optim::AdamW optimizer(student->parameters(), torch::optim::AdamWOptions(5e-5));
    
    // 4. DynaBERT Training Loop (Alternating width) (移植)
    int epochs = 2; // 実験用に短縮
    std::cout << "\nStarting DynaBERT Training for WM=" << target_wm << "..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0;
        
        // エポック内でTeacherからStudentへ重みを抽出（初期化）
        // ※注意: 本来のDynaBERTでは、各レイヤーの重要度に基づいて重みをソートし、
        //   Studentの小さな行列にマッピングする複雑な処理が必要。
        //   ここでは「前方スライスをコピーする」簡易版として実装 (weight_extraction関数)
        weight_extraction(teacher, student, config);

        for (int i = 0; i < num_batches; ++i) {
            // C++のランダムテンソル生成
            torch::Tensor dummy_input_ids = torch::randint(0, config.vocab_size, {batch_size, seq_len}, torch::kLong);
            torch::Tensor dummy_labels = torch::randint(0, config.num_labels, {batch_size}, torch::kLong);
            
            dummy_input_ids = dummy_input_ids.to(device);
            dummy_labels = dummy_labels.to(device);
            
            double loss = train_dynabert_step(teacher, student, optimizer, dummy_input_ids, dummy_labels);
            epoch_loss += loss;
            
            if (i % 5 == 0) {
                std::cout << "Epoch " << epoch+1 << "/" << epochs << " | Batch " << i+1 << "/" << num_batches 
                          << " | Loss: " << loss << std::endl;
            }
        }
                
        std::cout << "Epoch " << epoch+1 << " completed. Avg Loss: " << epoch_loss/num_batches << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    std::cout << "Training finished in " << duration.count() << " minutes." << std::endl;

    // 5. Save the final dynamic model (すべてのサブネットワークの重みを含むTeacherモデルを保存する)
    // C++での保存方法
    torch::save(teacher, "dynabert_teacher_cpp.pt");
    std::cout << "Teacher model saved to dynabert_teacher_cpp.pt" << std::endl;

    // 6. 推論時のサブネットワークの取り出し例 (移植)
    std::cout << "\nExample: Extracting WM=0.5 subnetwork for inference." << std::endl;
    auto sub_model_inf = std::make_shared<ElasticBertForSequenceClassification>(config, 0.5);
    
    // 学習済みのTeacherをロード
    auto loaded_teacher = std::make_shared<ElasticBertForSequenceClassification>(config, 1.0);
    torch::load(loaded_teacher, "dynabert_teacher_cpp.pt");
    
    // 重み抽出関数を再利用
    weight_extraction(loaded_teacher, sub_model_inf, config);
    
    sub_model_inf->to(device);
    sub_model_inf->eval();
    // これで WM=0.5 の小さなモデルとして推論できる
    std::cout << "WM=0.5 sub-model is ready for inference." << std::endl;

    return 0;
}