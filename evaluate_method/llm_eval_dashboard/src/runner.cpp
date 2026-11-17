#include "runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <thread>

#include "benchmark.h"
#include "mock_provider.h"
#include "openai_client.h"
#include "util.h"

namespace llmbench {

namespace {

using Clock = std::chrono::steady_clock;

struct Task {
  int model_index = 0;
  int item_index = 0;
};

bool cancelled(std::atomic<bool>* flag) {
  return flag != nullptr && flag->load();
}

} // namespace

std::vector<BenchmarkItem> select_items(const Dataset& dataset, int max_items) {
  if (max_items <= 0 || static_cast<std::size_t>(max_items) >= dataset.items.size()) {
    return dataset.items;
  }

  std::vector<std::string> category_order;
  std::map<std::string, std::vector<BenchmarkItem>> by_category;
  for (const auto& item : dataset.items) {
    if (by_category.find(item.category) == by_category.end()) {
      category_order.push_back(item.category);
    }
    by_category[item.category].push_back(item);
  }

  std::vector<BenchmarkItem> selected;
  std::size_t round = 0;
  while (selected.size() < static_cast<std::size_t>(max_items)) {
    bool added = false;
    for (const auto& category : category_order) {
      const auto& items = by_category[category];
      if (round < items.size()) {
        selected.push_back(items[round]);
        added = true;
        if (selected.size() >= static_cast<std::size_t>(max_items)) {
          break;
        }
      }
    }
    if (!added) {
      break;
    }
    ++round;
  }
  return selected;
}

RunRecord run_evaluation(const EvalConfig& config,
                         const Dataset& dataset,
                         const std::vector<ModelTarget>& models,
                         const RunOptions& options,
                         const ResultCallback& on_result) {
  auto started = Clock::now();
  std::vector<BenchmarkItem> items = select_items(dataset, options.max_items);

  RunRecord run;
  run.name = config.name;
  run.created_at = iso_now();
  run.source = "live";
  run.dataset_name = dataset.name;
  run.dataset_path = config.dataset_path;
  run.pass_threshold = config.pass_threshold;
  run.config = sanitize_config(config);
  run.id = "run_" + file_stamp_now() + "_" + slugify(config.name) + "_" +
           std::to_string(options.seed % 100000);
  run.total_requests = 0;

  std::vector<Task> tasks;
  tasks.reserve(items.size() * models.size());
  for (std::size_t item_index = 0; item_index < items.size(); ++item_index) {
    for (std::size_t model_index = 0; model_index < models.size(); ++model_index) {
      tasks.push_back(Task{static_cast<int>(model_index), static_cast<int>(item_index)});
    }
  }

  std::vector<RequestResult> results(tasks.size());
  std::atomic<std::size_t> next{0};

  auto worker = [&]() {
    for (;;) {
      std::size_t index = next.fetch_add(1);
      if (index >= tasks.size() || cancelled(options.cancel)) {
        return;
      }
      const Task& task = tasks[index];
      const ModelTarget& model = models[static_cast<std::size_t>(task.model_index)];
      const BenchmarkItem& item = items[static_cast<std::size_t>(task.item_index)];

      RequestResult result;
      result.model_id = model.id;
      result.item_id = item.id;
      result.category = item.category;

      for (int attempt = 0; attempt <= config.retries; ++attempt) {
        if (cancelled(options.cancel)) {
          break;
        }

        bool ok = false;
        if (model.provider == "mock") {
          std::uint64_t seed = options.seed + static_cast<std::uint64_t>(attempt) * 7919ULL;
          MockResult mock = generate_mock(model, item, seed);
          ok = mock.ok;
          result.error = mock.error;
          result.output = mock.output;
          result.prompt_tokens = mock.prompt_tokens;
          result.completion_tokens = mock.completion_tokens;
          result.ttft_ms = mock.ttft_ms;
          result.latency_ms = mock.latency_ms;
          result.tokens_per_sec = mock.tokens_per_sec;
          result.cost_usd = mock.cost_usd;
        } else {
          CallOutcome outcome =
              call_openai_compatible(model, item, config, options.cancel);
          ok = outcome.ok;
          result.error = outcome.error;
          result.output = outcome.output;
          result.prompt_tokens = outcome.prompt_tokens;
          result.completion_tokens = outcome.completion_tokens;
          result.ttft_ms = outcome.ttft_ms;
          result.latency_ms = outcome.latency_ms;
          if (outcome.ok && outcome.latency_ms > 0.0) {
            result.tokens_per_sec =
                static_cast<double>(outcome.completion_tokens) /
                (outcome.latency_ms / 1000.0);
          }
          result.cost_usd =
              static_cast<double>(outcome.prompt_tokens) / 1e6 *
                  model.input_price_per_mtok +
              static_cast<double>(outcome.completion_tokens) / 1e6 *
                  model.output_price_per_mtok;
        }

        if (ok) {
          ScoreDetail score = score_output(item, result.output);
          result.ok = true;
          result.quality = score.quality;
          result.passed = score.quality >= config.pass_threshold;
          result.matched_keywords = score.matched_keywords;
          result.error.clear();
          break;
        }

        result.ok = false;
        result.quality = 0.0;
        result.passed = false;
        result.matched_keywords.clear();
        if (cancelled(options.cancel)) {
          break;
        }
        if (attempt < config.retries) {
          std::this_thread::sleep_for(
              std::chrono::milliseconds(350 * (attempt + 1)));
        }
      }

      if (cancelled(options.cancel)) {
        return;
      }

      results[index] = result;
      if (on_result) {
        try {
          on_result(task.model_index, result);
        } catch (...) {
          // A dashboard callback must never take down the worker.
        }
      }
    }
  };

  int thread_count = std::max(1, std::min(config.concurrency,
                                          static_cast<int>(tasks.size())));
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(thread_count));
  for (int i = 0; i < thread_count; ++i) {
    threads.emplace_back(worker);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  run.results.reserve(results.size());
  for (auto& result : results) {
    if (!result.model_id.empty()) {
      run.results.push_back(std::move(result));
    }
  }
  run.total_requests = static_cast<int>(run.results.size());
  run.duration_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  compute_summaries(run, models);
  return run;
}

} // namespace llmbench
