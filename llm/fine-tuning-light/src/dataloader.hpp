#pragma once
#include <vector>
#include <string>
#include <cstddef>

struct Batch {
    std::vector<std::vector<int>> tokens; // variable length per sample
    std::vector<int> targets;             // next-token prediction target
    size_t max_len = 0;
};

class DataLoader {
public:
    std::vector<std::vector<int>> data;
    size_t batch_size;
    size_t current_pos = 0;
    bool shuffle;
    int pad_id;

    DataLoader(const std::vector<std::vector<int>>& data_, size_t batch_size_, bool shuffle_ = true, int pad_id_ = 0);

    void reset();
    bool has_next() const;
    Batch next();

    // Load text file: each line is a document, tokenize with simple whitespace
    static std::vector<std::vector<int>> load_text_file(const std::string& path, int vocab_size = 32000);

private:
    std::vector<size_t> indices;
    void shuffle_indices();
};
