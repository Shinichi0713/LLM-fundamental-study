#pragma once

#include <map>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace llmbench {

using json = nlohmann::json;

// A model endpoint that can be evaluated. Providers:
//   openai : any OpenAI-compatible /chat/completions endpoint
//   mock   : deterministic local simulation (no network, no API key)
struct ModelTarget {
  std::string id;
  std::string provider = "openai";
  std::string base_url = "http://localhost:11434/v1";
  std::string model;
  std::string api_key_env;
  std::string mock_profile = "balanced";
  std::string color = "#38bdf8";
  double input_price_per_mtok = 0.0;
  double output_price_per_mtok = 0.0;
  bool enabled = true;
};

struct BenchmarkItem {
  std::string id;
  std::string category;
  std::string prompt;
  std::string system;
  std::string reference;
  std::vector<std::string> expected_keywords;
  int max_tokens = 256;
  int max_chars = 0;
  double weight = 1.0;
};

struct ScoreDetail {
  double quality = 0.0;
  double keyword_recall = 0.0;
  double similarity = 0.0;
  bool passed = false;
  std::vector<std::string> matched_keywords;
  std::vector<std::string> missed_keywords;
};

struct RequestResult {
  std::string model_id;
  std::string item_id;
  std::string category;
  bool ok = false;
  std::string error;
  std::string output;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  double ttft_ms = 0.0;
  double latency_ms = 0.0;
  double tokens_per_sec = 0.0;
  double quality = 0.0;
  double cost_usd = 0.0;
  bool passed = false;
  std::vector<std::string> matched_keywords;

  json to_json() const;
  static RequestResult from_json(const json& value);
};

struct ModelSummary {
  std::string model_id;
  std::string provider;
  std::string color;
  int total = 0;
  int passed = 0;
  int failures = 0;
  double quality = 0.0;
  double success_rate = 0.0;
  double avg_latency_ms = 0.0;
  double p50_latency_ms = 0.0;
  double p95_latency_ms = 0.0;
  double avg_ttft_ms = 0.0;
  double tokens_per_sec = 0.0;
  double total_cost_usd = 0.0;
  double cost_per_1k_usd = 0.0;
  double overall = 0.0;
  std::map<std::string, double> category_scores;
  std::map<std::string, double> category_quality;

  json to_json() const;
  static ModelSummary from_json(const json& value);
};

struct RunRecord {
  std::string id;
  std::string name;
  std::string created_at;
  std::string source = "live"; // live | demo
  std::string dataset_name;
  std::string dataset_path;
  std::string app_version = "1.0.0";
  int total_requests = 0;
  double duration_ms = 0.0;
  double pass_threshold = 0.6;
  json config = json::object();
  std::vector<RequestResult> results;
  std::vector<ModelSummary> summaries;

  json to_json(bool include_results = true) const;
  static RunRecord from_json(const json& value);
};

struct EvalConfig {
  std::string name = "LLM evaluation";
  std::string dataset_path = "data/benchmark_ja.json";
  int concurrency = 4;
  int timeout_ms = 60000;
  double temperature = 0.0;
  int max_tokens = 256;
  double pass_threshold = 0.6;
  int retries = 1;
  std::vector<ModelTarget> models;
  json raw = json::object();
};

struct Dataset {
  std::string name = "Benchmark";
  std::string version = "1.0";
  std::vector<BenchmarkItem> items;
};

struct Progress {
  int completed = 0;
  int total = 0;
  int model_index = 0;
  std::string message;
};

struct ModelProgress {
  std::string id;
  std::string color;
  int completed = 0;
  int total = 0;
  double quality_sum = 0.0;
  int quality_count = 0;
  int passed = 0;
  int failures = 0;
};

struct JobSnapshot {
  std::string id;
  std::string name;
  std::string state = "idle"; // idle | running | completed | cancelled | failed
  std::string message;
  int completed = 0;
  int total = 0;
  double elapsed_ms = 0.0;
  std::string run_id;
  std::vector<ModelProgress> models;
};

} // namespace llmbench
