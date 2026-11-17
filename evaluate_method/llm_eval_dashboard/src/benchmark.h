#pragma once

#include <string>

#include "types.h"

namespace llmbench {

// Loads a benchmark dataset from JSON. Throws std::runtime_error on failure.
Dataset load_dataset(const std::string& path);

// Loads an evaluation config. Relative dataset paths are resolved against the
// config file directory first. Throws std::runtime_error on failure.
EvalConfig load_config(const std::string& path);

// Resolves a possibly relative path that was referenced from config_path.
std::string resolve_relative_to(const std::string& config_path, const std::string& path);

ScoreDetail score_output(const BenchmarkItem& item, const std::string& output);

ModelSummary summarize_model(const ModelTarget& model,
                             const std::vector<RequestResult>& results,
                             double pass_threshold);

void compute_summaries(RunRecord& run, const std::vector<ModelTarget>& models);

json sanitize_config(const EvalConfig& config);
json sanitize_model(const ModelTarget& model);

} // namespace llmbench
