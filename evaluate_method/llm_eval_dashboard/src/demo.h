#pragma once

#include "types.h"

namespace llmbench {

// A self-contained configuration with four simulated models. Used by the
// demo command and as a fallback when no config file is present.
EvalConfig demo_config();

// Small built-in dataset used when data/benchmark_ja.json cannot be found.
Dataset builtin_dataset();

} // namespace llmbench
