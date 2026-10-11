#include "dataloader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>

DataLoader::DataLoader(const std::vector<std::vector<int>>& data_, size_t batch_size_, bool shuffle_, int pad_id_)
    : data(data_), batch_size(batch_size_), shuffle(shuffle_), pad_id(pad_id_) {
    indices.resize(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    if (shuffle) shuffle_indices();
}

void DataLoader::reset() {
    current_pos = 0;
    if (shuffle) shuffle_indices();
}

bool DataLoader::has_next() const {
    return current_pos < data.size();
}

Batch DataLoader::next() {
    Batch batch;
    size_t end = std::min(current_pos + batch_size, data.size());
    for (size_t i = current_pos; i < end; ++i) {
        const auto& seq = data[indices[i]];
        if (!seq.empty()) {
            batch.tokens.push_back(seq);
            batch.targets.push_back(seq.back()); // simplistic: last token as target
            if (seq.size() > batch.max_len) batch.max_len = seq.size();
        }
    }
    current_pos = end;
    return batch;
}

void DataLoader::shuffle_indices() {
    std::mt19937 g(42);
    std::shuffle(indices.begin(), indices.end(), g);
}

std::vector<std::vector<int>> DataLoader::load_text_file(const std::string& path, int vocab_size) {
    std::vector<std::vector<int>> result;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Failed to open " << path << "\n";
        return result;
    }
    std::string line;
    std::mt19937 g(123);
    std::uniform_int_distribution<int> dist(1, vocab_size - 1);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<int> tokens;
        // Simple whitespace tokenization -> hash-based token ids
        std::istringstream iss(line);
        std::string word;
        while (iss >> word) {
            // Deterministic hash to vocab id
            size_t h = std::hash<std::string>{}(word);
            tokens.push_back(static_cast<int>(h % (vocab_size - 1)) + 1);
        }
        if (!tokens.empty()) result.push_back(tokens);
    }
    return result;
}
