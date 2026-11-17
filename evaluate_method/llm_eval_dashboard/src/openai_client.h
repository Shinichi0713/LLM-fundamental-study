#pragma once

#include <atomic>
#include <string>

#include "types.h"

namespace llmbench {

struct CallOutcome {
  bool ok = false;
  bool aborted = false;
  int status = 0;
  std::string error;
  std::string output;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  double ttft_ms = 0.0;
  double latency_ms = 0.0;
};

// Calls an OpenAI-compatible /chat/completions endpoint with streaming enabled.
CallOutcome call_openai_compatible(const ModelTarget& model,
                                   const BenchmarkItem& item,
                                   const EvalConfig& config,
                                   std::atomic<bool>* cancel);

} // namespace llmbench
