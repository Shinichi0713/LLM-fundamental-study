#include "benchmark.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "util.h"

namespace fs = std::filesystem;

namespace llmbench {

namespace {

const std::vector<std::string> kPalette = {
    "#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#fb7185",
    "#22d3ee", "#f472b6", "#a3e635", "#f97316", "#818cf8",
};

std::string palette_color(std::size_t index) {
  return kPalette[index % kPalette.size()];
}

std::string require_string(const json& value, const char* key, const std::string& context) {
  auto it = value.find(key);
  if (it == value.end() || !it->is_string() || it->get<std::string>().empty()) {
    throw std::runtime_error(context + ": missing string field '" + key + "'");
  }
  return it->get<std::string>();
}

std::vector<std::string> read_strings(const json& value, const char* key) {
  std::vector<std::string> result;
  auto it = value.find(key);
  if (it == value.end()) {
    return result;
  }
  if (it->is_string()) {
    result.push_back(it->get<std::string>());
    return result;
  }
  if (it->is_array()) {
    for (const auto& entry : *it) {
      if (entry.is_string()) {
        result.push_back(entry.get<std::string>());
      }
    }
  }
  return result;
}

bool keyword_matches(const std::string& normalized_output, const std::string& keyword) {
  for (const auto& alternative : split(keyword, '|')) {
    std::string needle = normalize_text(trim(alternative));
    if (!needle.empty() && normalized_output.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

} // namespace

std::string resolve_relative_to(const std::string& config_path, const std::string& path) {
  fs::path given(path);
  if (given.is_absolute()) {
    return given.lexically_normal().string();
  }
  if (!config_path.empty()) {
    fs::path base = fs::path(config_path).parent_path();
    fs::path candidate = (base / given).lexically_normal();
    std::error_code ec;
    if (fs::exists(candidate, ec)) {
      return candidate.string();
    }
  }
  std::string resolved = resolve_existing_path(path);
  if (!resolved.empty()) {
    return resolved;
  }
  if (!config_path.empty()) {
    return (fs::path(config_path).parent_path() / given).lexically_normal().string();
  }
  return given.lexically_normal().string();
}

Dataset load_dataset(const std::string& path) {
  bool ok = false;
  std::string content = read_file(path, ok);
  if (!ok) {
    throw std::runtime_error("Cannot open dataset: " + path);
  }

  json root;
  try {
    root = json::parse(content);
  } catch (const std::exception& error) {
    throw std::runtime_error("Invalid JSON in dataset '" + path + "': " + error.what());
  }

  Dataset dataset;
  json items = root;
  if (root.is_object()) {
    dataset.name = root.value("name", "Benchmark");
    dataset.version = root.value("version", "1.0");
    if (!root.contains("items") || !root["items"].is_array()) {
      throw std::runtime_error("Dataset '" + path + "' must contain an 'items' array");
    }
    items = root["items"];
  } else if (!root.is_array()) {
    throw std::runtime_error("Dataset '" + path + "' must be an object or array");
  }

  std::unordered_set<std::string> seen_ids;
  int index = 0;
  for (const auto& entry : items) {
    ++index;
    if (!entry.is_object()) {
      continue;
    }
    BenchmarkItem item;
    item.id = entry.value("id", "item_" + std::to_string(index));
    item.category = entry.value("category", "general");
    item.prompt = entry.value("prompt", "");
    item.system = entry.value("system", "");
    item.reference = entry.value("reference", "");
    item.expected_keywords = read_strings(entry, "expected_keywords");
    item.max_tokens = entry.value("max_tokens", 256);
    item.max_chars = entry.value("max_chars", 0);
    item.weight = entry.value("weight", 1.0);
    if (item.prompt.empty()) {
      throw std::runtime_error("Dataset item '" + item.id + "' has an empty prompt");
    }
    if (!seen_ids.insert(item.id).second) {
      throw std::runtime_error("Duplicate dataset item id: " + item.id);
    }
    dataset.items.push_back(std::move(item));
  }

  if (dataset.items.empty()) {
    throw std::runtime_error("Dataset '" + path + "' contains no items");
  }
  return dataset;
}

EvalConfig load_config(const std::string& path) {
  bool ok = false;
  std::string content = read_file(path, ok);
  if (!ok) {
    throw std::runtime_error("Cannot open config: " + path);
  }

  json root;
  try {
    root = json::parse(content);
  } catch (const std::exception& error) {
    throw std::runtime_error("Invalid JSON in config '" + path + "': " + error.what());
  }
  if (!root.is_object()) {
    throw std::runtime_error("Config must be a JSON object");
  }

  EvalConfig config;
  config.name = root.value("name", "LLM evaluation");
  config.dataset_path = root.value("dataset", "data/benchmark_ja.json");
  config.concurrency = root.value("concurrency", 4);
  config.timeout_ms = root.value("timeout_ms", 60000);
  config.temperature = root.value("temperature", 0.0);
  config.max_tokens = root.value("max_tokens", 256);
  config.pass_threshold = root.value("pass_threshold", 0.6);
  config.retries = root.value("retries", 1);

  config.concurrency = std::max(1, std::min(32, config.concurrency));
  config.timeout_ms = std::max(1000, std::min(600000, config.timeout_ms));
  config.pass_threshold = std::max(0.0, std::min(1.0, config.pass_threshold));
  config.retries = std::max(0, std::min(5, config.retries));

  if (!root.contains("models") || !root["models"].is_array() || root["models"].empty()) {
    throw std::runtime_error("Config must contain a non-empty 'models' array");
  }

  std::size_t model_index = 0;
  for (const auto& entry : root["models"]) {
    if (!entry.is_object()) {
      continue;
    }
    ModelTarget model;
    model.model = entry.value("model", "");
    model.id = entry.value("id", model.model.empty() ? "model_" + std::to_string(model_index + 1)
                                                     : model.model);
    model.provider = to_lower_ascii(entry.value("provider", "openai"));
    model.base_url = entry.value("base_url", "http://localhost:11434/v1");
    model.api_key_env = entry.value("api_key_env", "");
    model.mock_profile = entry.value("profile", "balanced");
    model.color = entry.value("color", palette_color(model_index));
    model.input_price_per_mtok = entry.value("input_price_per_mtok", 0.0);
    model.output_price_per_mtok = entry.value("output_price_per_mtok", 0.0);
    model.enabled = entry.value("enabled", true);

    if (model.provider != "openai" && model.provider != "mock") {
      throw std::runtime_error("Model '" + model.id +
                               "': provider must be 'openai' or 'mock'");
    }
    if (model.provider == "openai" && model.model.empty()) {
      throw std::runtime_error("Model '" + model.id + "': 'model' is required");
    }
    if (model.provider == "openai" &&
        !starts_with(model.base_url, "http://") && !starts_with(model.base_url, "https://")) {
      throw std::runtime_error("Model '" + model.id + "': base_url must start with http:// or https://");
    }
    if (model.provider == "mock" && model.model.empty()) {
      model.model = model.id;
    }
    config.models.push_back(std::move(model));
    ++model_index;
  }

  if (config.models.empty()) {
    throw std::runtime_error("Config contains no usable models");
  }

  config.dataset_path = resolve_relative_to(path, config.dataset_path);
  config.raw = sanitize_config(config);
  return config;
}

ScoreDetail score_output(const BenchmarkItem& item, const std::string& output) {
  ScoreDetail detail;
  if (trim(output).empty()) {
    detail.missed_keywords = item.expected_keywords;
    return detail;
  }

  std::string normalized_output = normalize_text(output);
  int matched = 0;
  for (const auto& keyword : item.expected_keywords) {
    if (keyword_matches(normalized_output, keyword)) {
      detail.matched_keywords.push_back(keyword);
      ++matched;
    } else {
      detail.missed_keywords.push_back(keyword);
    }
  }

  if (!item.expected_keywords.empty()) {
    detail.keyword_recall =
        static_cast<double>(matched) / static_cast<double>(item.expected_keywords.size());
  }
  if (!item.reference.empty()) {
    detail.similarity = char_bigram_f1(output, item.reference);
  }

  if (!item.expected_keywords.empty() && !item.reference.empty()) {
    detail.quality = 0.72 * detail.keyword_recall + 0.28 * detail.similarity;
  } else if (!item.expected_keywords.empty()) {
    detail.quality = detail.keyword_recall;
  } else if (!item.reference.empty()) {
    detail.quality = detail.similarity;
  } else {
    detail.quality = 1.0;
  }

  if (item.max_chars > 0) {
    auto chars = utf8_chars(output);
    if (chars.size() > static_cast<std::size_t>(item.max_chars)) {
      double overflow = static_cast<double>(chars.size()) / static_cast<double>(item.max_chars);
      detail.quality *= std::max(0.4, 1.0 - 0.15 * (overflow - 1.0));
    }
  }

  detail.quality = clamp01(detail.quality);
  return detail;
}

ModelSummary summarize_model(const ModelTarget& model,
                             const std::vector<RequestResult>& results,
                             double pass_threshold) {
  ModelSummary summary;
  summary.model_id = model.id;
  summary.provider = model.provider;
  summary.color = model.color;

  std::vector<double> latencies;
  std::vector<double> ttfts;
  double quality_sum = 0.0;
  double cost_sum = 0.0;
  long long completion_tokens = 0;
  double total_latency_seconds = 0.0;
  int success_count = 0;

  std::map<std::string, double> category_quality_sum;
  std::map<std::string, int> category_count;

  for (const auto& result : results) {
    if (result.model_id != model.id) {
      continue;
    }
    ++summary.total;
    quality_sum += result.ok ? result.quality : 0.0;
    category_quality_sum[result.category] += result.ok ? result.quality : 0.0;
    category_count[result.category] += 1;

    if (!result.ok) {
      ++summary.failures;
      continue;
    }
    ++success_count;
    if (result.passed || result.quality >= pass_threshold) {
      ++summary.passed;
    }
    if (result.latency_ms > 0.0) {
      latencies.push_back(result.latency_ms);
      total_latency_seconds += result.latency_ms / 1000.0;
    }
    if (result.ttft_ms > 0.0) {
      ttfts.push_back(result.ttft_ms);
    }
    completion_tokens += result.completion_tokens;
    cost_sum += result.cost_usd;
  }

  if (summary.total > 0) {
    summary.quality = 100.0 * quality_sum / static_cast<double>(summary.total);
    summary.success_rate =
        100.0 * static_cast<double>(success_count) / static_cast<double>(summary.total);
    summary.total_cost_usd = cost_sum;
    summary.cost_per_1k_usd =
        cost_sum / static_cast<double>(summary.total) * 1000.0;
  }

  if (!latencies.empty()) {
    double latency_sum = 0.0;
    for (double value : latencies) {
      latency_sum += value;
    }
    summary.avg_latency_ms = latency_sum / static_cast<double>(latencies.size());
    summary.p50_latency_ms = percentile(latencies, 0.50);
    summary.p95_latency_ms = percentile(latencies, 0.95);
  }
  if (!ttfts.empty()) {
    double ttft_sum = 0.0;
    for (double value : ttfts) {
      ttft_sum += value;
    }
    summary.avg_ttft_ms = ttft_sum / static_cast<double>(ttfts.size());
  }
  if (total_latency_seconds > 0.0) {
    summary.tokens_per_sec =
        static_cast<double>(completion_tokens) / total_latency_seconds;
  }

  for (const auto& entry : category_count) {
    if (entry.second > 0) {
      summary.category_scores[entry.first] =
          100.0 * category_quality_sum[entry.first] / static_cast<double>(entry.second);
    }
  }

  double speed_score = clamp01(summary.tokens_per_sec / 80.0) * 100.0;
  double latency_score =
      clamp01(1.0 - std::max(0.0, summary.p50_latency_ms - 300.0) / 3000.0) * 100.0;
  double cost_score = 100.0;
  if (summary.total_cost_usd > 0.0) {
    cost_score = clamp01(std::exp(-summary.cost_per_1k_usd / 1.5)) * 100.0;
  }
  summary.overall = 0.45 * summary.quality + 0.20 * speed_score +
                    0.15 * summary.success_rate + 0.10 * latency_score +
                    0.10 * cost_score;
  summary.overall = clamp01(summary.overall / 100.0) * 100.0;
  if (success_count == 0) {
    // Do not reward a model that produced no successful response at all.
    summary.overall = 0.0;
  }
  return summary;
}

void compute_summaries(RunRecord& run, const std::vector<ModelTarget>& models) {
  run.summaries.clear();
  for (const auto& model : models) {
    run.summaries.push_back(summarize_model(model, run.results, run.pass_threshold));
  }
}

json sanitize_model(const ModelTarget& model) {
  return json{
      {"id", model.id},
      {"provider", model.provider},
      {"base_url", model.base_url},
      {"model", model.model},
      {"api_key_env", model.api_key_env},
      {"profile", model.mock_profile},
      {"color", model.color},
      {"input_price_per_mtok", model.input_price_per_mtok},
      {"output_price_per_mtok", model.output_price_per_mtok},
      {"enabled", model.enabled},
  };
}

json sanitize_config(const EvalConfig& config) {
  json models = json::array();
  for (const auto& model : config.models) {
    models.push_back(sanitize_model(model));
  }
  return json{
      {"name", config.name},
      {"dataset", config.dataset_path},
      {"concurrency", config.concurrency},
      {"timeout_ms", config.timeout_ms},
      {"temperature", config.temperature},
      {"max_tokens", config.max_tokens},
      {"pass_threshold", config.pass_threshold},
      {"retries", config.retries},
      {"models", models},
  };
}

} // namespace llmbench
