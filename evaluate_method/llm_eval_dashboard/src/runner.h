#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

#include "types.h"

namespace llmbench {

struct RunOptions {
  int max_items = 0; // 0 = all items
  std::uint64_t seed = 0;
  std::atomic<bool>* cancel = nullptr;
};

// Called from worker threads. Implementations must be thread-safe.
using ResultCallback = std::function<void(int model_index, const RequestResult& result)>;

std::vector<BenchmarkItem> select_items(const Dataset& dataset, int max_items);

RunRecord run_evaluation(const EvalConfig& config,
                         const Dataset& dataset,
                         const std::vector<ModelTarget>& models,
                         const RunOptions& options,
                         const ResultCallback& on_result);

} // namespace llmbench
