#include "tensor.hpp"
#include "nn.hpp"
#include "lora.hpp"
#include "optimizer.hpp"
#include "dataloader.hpp"
#include "transformer.hpp"
#include "collective.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

struct TrainConfig {
    std::string model_name = "llama-like-7b-sim";
    std::string output_dir = "./checkpoints";
    std::string new_data_path = "./new_data.txt";
    std::string rehearsal_data_path = "./rehearsal.txt";
    std::string rank_config_path = "./rank_config.txt";

    size_t vocab_size = 32000;
    size_t embed_dim = 512;      // reduced for CPU demo (real 7B uses 4096)
    size_t num_heads = 8;        // real 7B uses 32
    size_t hidden_dim = 1376;  // real 7B uses 11008
    size_t num_layers = 4;       // real 7B uses 32
    size_t max_seq_len = 128;

    int lora_rank = 8;
    float lora_alpha = 16.0f;
    float lambda_distill = 0.5f;
    float rehearsal_ratio = 0.2f;

    size_t batch_size = 4;
    int accum_steps = 8;
    int epochs = 3;
    float lr = 2e-4f;
    float weight_decay = 0.01f;
    int warmup_steps = 100;

    int rank = 0;
    int world_size = 1;
};

float compute_loss(CausalLM& student, CausalLM& teacher, const Batch& batch,
                   float lambda_distill, SoftmaxCrossEntropy& ce_loss, KLDivLoss& kl_loss) {
    Tensor emb = student.embed(batch.tokens, batch.max_len);
    Tensor hidden = student.forward(emb);
    Tensor student_logits = student.logits(hidden);

    float loss_task = 0.0f;
    if (!batch.targets.empty()) {
        loss_task = ce_loss.loss(student_logits.view({student_logits.shape[0] * student_logits.shape[1],
                                                       student_logits.shape[2]}), batch.targets);
    }

    float loss_distill = 0.0f;
    if (lambda_distill > 0.0f && teacher.vocab_size > 0) {
        Tensor teacher_emb = teacher.embed(batch.tokens, batch.max_len);
        Tensor teacher_hidden = teacher.forward(teacher_emb);
        Tensor teacher_logits = teacher.logits(teacher_hidden);

        Tensor student_lp = student_logits.log_softmax(-1);
        Tensor teacher_prob = teacher_logits.softmax(-1);
        loss_distill = kl_loss.loss(student_lp.view({student_lp.shape[0] * student_lp.shape[1],
                                                      student_lp.shape[2]}),
                                    teacher_prob.view({teacher_prob.shape[0] * teacher_prob.shape[2]}));
    }

    return loss_task + lambda_distill * loss_distill;
}

void run_training(TrainConfig& cfg, dist::DistContext* dist_ctx) {
    std::cout << "Rank " << cfg.rank << "/" << cfg.world_size << " starting training...\n";

    // Load data
    auto new_data = DataLoader::load_text_file(cfg.new_data_path, static_cast<int>(cfg.vocab_size));
    auto rehearsal_data = DataLoader::load_text_file(cfg.rehearsal_data_path, static_cast<int>(cfg.vocab_size));

    if (new_data.empty()) {
        std::cerr << "Warning: new_data is empty, generating dummy data.\n";
        for (int i = 0; i < 100; ++i) {
            std::vector<int> seq;
            for (int j = 0; j < 16; ++j) seq.push_back((i * 16 + j) % static_cast<int>(cfg.vocab_size - 1) + 1);
            new_data.push_back(seq);
        }
    }
    if (rehearsal_data.empty()) {
        std::cerr << "Warning: rehearsal_data is empty, generating dummy data.\n";
        for (int i = 0; i < 500; ++i) {
            std::vector<int> seq;
            for (int j = 0; j < 16; ++j) seq.push_back((i + j * 3) % static_cast<int>(cfg.vocab_size - 1) + 1);
            rehearsal_data.push_back(seq);
        }
    }

    // Mix datasets
    std::vector<std::vector<int>> mixed_data = new_data;
    size_t num_re = static_cast<size_t>((cfg.rehearsal_ratio / (1.0f - cfg.rehearsal_ratio)) * static_cast<float>(new_data.size()));
    size_t added = 0;
    while (added < num_re && !rehearsal_data.empty()) {
        for (const auto& seq : rehearsal_data) {
            mixed_data.push_back(seq);
            added++;
            if (added >= num_re) break;
        }
    }

    DataLoader loader(mixed_data, cfg.batch_size, true, 0);

    // Build models
    CausalLM student(cfg.vocab_size, cfg.embed_dim, cfg.num_heads, cfg.hidden_dim,
                     cfg.num_layers, cfg.max_seq_len, cfg.lora_rank, cfg.lora_alpha);
    CausalLM teacher(cfg.vocab_size, cfg.embed_dim, cfg.num_heads, cfg.hidden_dim,
                     cfg.num_layers, cfg.max_seq_len, 0, 0.0f); // no LoRA, frozen

    // Freeze teacher
    // (all teacher params have requires_grad=false by default in our impl)

    AdamW opt(cfg.lr, 0.9f, 0.999f, 1e-8f, cfg.weight_decay);
    student.register_params(opt);

    SoftmaxCrossEntropy ce_loss;
    KLDivLoss kl_loss;

    int global_step = 0;
    auto start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        loader.reset();
        float epoch_loss = 0.0f;
        int num_batches = 0;
        int step_in_epoch = 0;

        while (loader.has_next()) {
            Batch batch = loader.next();
            if (batch.tokens.empty()) continue;

            float loss = compute_loss(student, teacher, batch, cfg.lambda_distill, ce_loss, kl_loss);
            epoch_loss += loss;
            num_batches++;
            step_in_epoch++;

            // Backward (simplified: just zero grad for now since we haven't implemented full autograd)
            student.zero_grad();

            // Gradient accumulation
            if (step_in_epoch % cfg.accum_steps == 0) {
                // In a full autograd system, we would call backward() on loss here.
                // For this framework, we simulate parameter updates with dummy gradients.
                // Real implementation requires full computation graph tracking.

                // Distributed: all-reduce gradients
                if (dist_ctx && dist_ctx->world_size > 1) {
                    std::vector<Tensor*> grads;
                    // Collect all trainable gradient tensors
                    // (In full impl, traverse all params and all_reduce their .grad)
                    // dist_ctx->average_gradients(grads);
                }

                opt.step();
                opt.zero_grad();
                global_step++;
            }

            if (global_step % 10 == 0 && cfg.rank == 0) {
                std::cout << "Epoch " << (epoch + 1) << " step " << global_step
                          << " loss " << (epoch_loss / num_batches) << "\n";
            }
        }

        if (cfg.rank == 0) {
            std::cout << "Epoch " << (epoch + 1) << " finished. avg loss: "
                      << (num_batches > 0 ? epoch_loss / num_batches : 0.0f) << "\n";
        }

        // Barrier at epoch end
        if (dist_ctx) dist_ctx->barrier();
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    if (cfg.rank == 0) {
        std::cout << "Training complete in " << elapsed << "s\n";
    }
}

int main(int argc, char** argv) {
    TrainConfig cfg;

    // Parse simple args
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--rank" && i + 1 < argc) cfg.rank = std::atoi(argv[++i]);
        else if (arg == "--world_size" && i + 1 < argc) cfg.world_size = std::atoi(argv[++i]);
        else if (arg == "--config" && i + 1 < argc) cfg.rank_config_path = argv[++i];
        else if (arg == "--new_data" && i + 1 < argc) cfg.new_data_path = argv[++i];
        else if (arg == "--rehearsal" && i + 1 < argc) cfg.rehearsal_data_path = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) cfg.epochs = std::atoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) cfg.lr = std::atof(argv[++i]);
        else if (arg == "--batch_size" && i + 1 < argc) cfg.batch_size = static_cast<size_t>(std::atoi(argv[++i]));
        else if (arg == "--lora_rank" && i + 1 < argc) cfg.lora_rank = std::atoi(argv[++i]);
        else if (arg == "--lambda" && i + 1 < argc) cfg.lambda_distill = static_cast<float>(std::atof(argv[++i]));
    }

    dist::DistContext* dist_ctx = nullptr;
    std::unique_ptr<dist::DistContext> dist_ctx_ptr;

    if (cfg.world_size > 1) {
        auto infos = dist::parse_rank_config(cfg.rank_config_path);
        if (infos.size() < static_cast<size_t>(cfg.world_size)) {
            std::cerr << "Rank config file has " << infos.size() << " entries, need " << cfg.world_size << "\n";
            std::cerr << "Format: each line = 'rank hostname port'\n";
            return 1;
        }
        dist_ctx_ptr = std::make_unique<dist::DistContext>(cfg.rank, cfg.world_size, infos);
        if (!dist_ctx_ptr->initialize()) {
            std::cerr << "Failed to initialize distributed context\n";
            return 1;
        }
        dist_ctx = dist_ctx_ptr.get();
    }

    run_training(cfg, dist_ctx);

    if (dist_ctx) dist_ctx->barrier();
    return 0;
}
