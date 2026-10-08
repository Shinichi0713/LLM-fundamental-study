#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <onnxruntime_cxx_api.h>

// ----------------------------------------------------------------------
// ONNX Runtime C++ 推論クラス
// ----------------------------------------------------------------------
class LLMInference {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

public:
    LLMInference(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "LLMInference"),
          session_options(),
          session(env, model_path.c_str(), session_options),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    }

    // 1ステップの Logits 出力推論
    std::vector<float> predict_next_token_logits(const std::vector<int64_t>& input_ids) {
        size_t batch_size = 1;
        size_t seq_len = input_ids.size();

        // 1. Attention Mask の作成 (すべて 1)
        std::vector<int64_t> attention_mask(seq_len, 1);

        // 2. テンソル形状の定義
        std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len)};

        // 3. Ort::Value テンソルの生成
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, const_cast<int64_t*>(input_ids.data()), input_ids.size(), input_shape.data(), input_shape.size()));
        
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size()));

        // 4. 入出力ノード名
        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {"logits"};

        // 5. 推論実行
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, input_tensors.data(), 2,
            output_names, 1
        );

        // 6. 出力テンソルの取得 [1, seq_len, vocab_size]
        float* float_data = output_tensors[0].GetTensorMutableData<float>();
        auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = tensor_info.GetShape();

        int64_t vocab_size = output_shape[2];
        
        // 最後のトークンに対応する Logits ([vocab_size]) だけを抽出
        size_t last_token_offset = (seq_len - 1) * vocab_size;
        std::vector<float> last_logits(float_data + last_token_offset, float_data + last_token_offset + vocab_size);

        return last_logits;
    }
};

// Argmax による Greedy 判定
int64_t argmax(const std::vector<float>& vec) {
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

// ----------------------------------------------------------------------
// メイン関数 (自己回帰トークン生成ループ)
// ----------------------------------------------------------------------
int main() {
    try {
        LLMInference llm("mini_llm.onnx");

        // プロンプトとなる初期トークン列 (例: トークンID: 10, 20, 30)
        std::vector<int64_t> context = {10, 20, 30};
        int generate_steps = 10;

        std::cout << "Initial Prompt Tokens: ";
        for (auto id : context) std::cout << id << " ";
        std::cout << "\nGenerating Tokens: ";

        // 自己回帰生成ループ
        for (int i = 0; i < generate_steps; ++i) {
            // 次のトークンの Logits を予測
            std::vector<float> logits = llm.predict_next_token_logits(context);
            
            // 最も確率の高いトークンを選択 (Greedy Sampling)
            int64_t next_token = argmax(logits);
            
            std::cout << next_token << " " << std::flush;
            
            // 生成されたトークンを文脈に追加して次のステップへ
            context.push_back(next_token);
        }
        std::cout << "\nGeneration complete.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}