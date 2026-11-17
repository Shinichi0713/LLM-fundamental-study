#include "types.h"

namespace llmbench {

namespace {
template <typename T>
std::vector<T> read_vector(const json& value, const char* key) {
  std::vector<T> result;
  auto it = value.find(key);
  if (it != value.end() && it->is_array()) {
    for (const auto& entry : *it) {
      result.push_back(entry.get<T>());
    }
  }
  return result;
}
} // namespace

json RequestResult::to_json() const {
  return json{
      {"model_id", model_id},
      {"item_id", item_id},
      {"category", category},
      {"ok", ok},
      {"error", error},
      {"output", output},
      {"prompt_tokens", prompt_tokens},
      {"completion_tokens", completion_tokens},
      {"ttft_ms", ttft_ms},
      {"latency_ms", latency_ms},
      {"tokens_per_sec", tokens_per_sec},
      {"quality", quality},
      {"cost_usd", cost_usd},
      {"passed", passed},
      {"matched_keywords", matched_keywords},
  };
}

RequestResult RequestResult::from_json(const json& value) {
  RequestResult result;
  result.model_id = value.value("model_id", "");
  result.item_id = value.value("item_id", "");
  result.category = value.value("category", "");
  result.ok = value.value("ok", false);
  result.error = value.value("error", "");
  result.output = value.value("output", "");
  result.prompt_tokens = value.value("prompt_tokens", 0);
  result.completion_tokens = value.value("completion_tokens", 0);
  result.ttft_ms = value.value("ttft_ms", 0.0);
  result.latency_ms = value.value("latency_ms", 0.0);
  result.tokens_per_sec = value.value("tokens_per_sec", 0.0);
  result.quality = value.value("quality", 0.0);
  result.cost_usd = value.value("cost_usd", 0.0);
  result.passed = value.value("passed", false);
  result.matched_keywords = read_vector<std::string>(value, "matched_keywords");
  return result;
}

json ModelSummary::to_json() const {
  return json{
      {"model_id", model_id},
      {"provider", provider},
      {"color", color},
      {"total", total},
      {"passed", passed},
      {"failures", failures},
      {"quality", quality},
      {"success_rate", success_rate},
      {"avg_latency_ms", avg_latency_ms},
      {"p50_latency_ms", p50_latency_ms},
      {"p95_latency_ms", p95_latency_ms},
      {"avg_ttft_ms", avg_ttft_ms},
      {"tokens_per_sec", tokens_per_sec},
      {"total_cost_usd", total_cost_usd},
      {"cost_per_1k_usd", cost_per_1k_usd},
      {"overall", overall},
      {"category_scores", category_scores},
  };
}

ModelSummary ModelSummary::from_json(const json& value) {
  ModelSummary result;
  result.model_id = value.value("model_id", "");
  result.provider = value.value("provider", "");
  result.color = value.value("color", "");
  result.total = value.value("total", 0);
  result.passed = value.value("passed", 0);
  result.failures = value.value("failures", 0);
  result.quality = value.value("quality", 0.0);
  result.success_rate = value.value("success_rate", 0.0);
  result.avg_latency_ms = value.value("avg_latency_ms", 0.0);
  result.p50_latency_ms = value.value("p50_latency_ms", 0.0);
  result.p95_latency_ms = value.value("p95_latency_ms", 0.0);
  result.avg_ttft_ms = value.value("avg_ttft_ms", 0.0);
  result.tokens_per_sec = value.value("tokens_per_sec", 0.0);
  result.total_cost_usd = value.value("total_cost_usd", 0.0);
  result.cost_per_1k_usd = value.value("cost_per_1k_usd", 0.0);
  result.overall = value.value("overall", 0.0);
  if (value.contains("category_scores") && value["category_scores"].is_object()) {
    for (auto it = value["category_scores"].begin(); it != value["category_scores"].end(); ++it) {
      result.category_scores[it.key()] = it.value().get<double>();
    }
  }
  return result;
}

json RunRecord::to_json(bool include_results) const {
  json value{
      {"id", id},
      {"name", name},
      {"created_at", created_at},
      {"source", source},
      {"dataset_name", dataset_name},
      {"dataset_path", dataset_path},
      {"app_version", app_version},
      {"total_requests", total_requests},
      {"duration_ms", duration_ms},
      {"pass_threshold", pass_threshold},
      {"config", config},
      {"summaries", json::array()},
  };
  for (const auto& summary : summaries) {
    value["summaries"].push_back(summary.to_json());
  }
  if (include_results) {
    value["results"] = json::array();
    for (const auto& result : results) {
      value["results"].push_back(result.to_json());
    }
  }
  return value;
}

RunRecord RunRecord::from_json(const json& value) {
  RunRecord result;
  result.id = value.value("id", "");
  result.name = value.value("name", "");
  result.created_at = value.value("created_at", "");
  result.source = value.value("source", "live");
  result.dataset_name = value.value("dataset_name", "");
  result.dataset_path = value.value("dataset_path", "");
  result.app_version = value.value("app_version", "1.0.0");
  result.total_requests = value.value("total_requests", 0);
  result.duration_ms = value.value("duration_ms", 0.0);
  result.pass_threshold = value.value("pass_threshold", 0.6);
  result.config = value.value("config", json::object());
  if (value.contains("summaries") && value["summaries"].is_array()) {
    for (const auto& entry : value["summaries"]) {
      result.summaries.push_back(ModelSummary::from_json(entry));
    }
  }
  if (value.contains("results") && value["results"].is_array()) {
    for (const auto& entry : value["results"]) {
      result.results.push_back(RequestResult::from_json(entry));
    }
  }
  return result;
}

} // namespace llmbench
