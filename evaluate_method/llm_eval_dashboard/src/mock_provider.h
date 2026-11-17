#pragma once

#include <cstdint>
#include <string>

#include "types.h"

namespace llmbench {

struct MockResult {
  bool ok = true;
  std::string output;
  std::string error;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  double ttft_ms = 0.0;
  double latency_ms = 0.0;
  double tokens_per_sec = 0.0;
  double cost_usd = 0.0;
};

// Deterministic, seeded simulation of an LLM response. The same model, item
// and seed always produce the same answer, latency and token counts.
MockResult generate_mock(const ModelTarget& model,
                         const BenchmarkItem& item,
                         std::uint64_t seed);

} // namespace llmbench
