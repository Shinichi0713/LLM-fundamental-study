#include "storage.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "util.h"

namespace fs = std::filesystem;

namespace llmbench {

namespace {

std::string csv_field(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  for (char c : value) {
    if (c == '"') {
      escaped += "\"\"";
    } else {
      escaped.push_back(c);
    }
  }
  return "\"" + escaped + "\"";
}

std::string model_name_for(const RunRecord& run, const std::string& model_id) {
  for (const auto& summary : run.summaries) {
    if (summary.model_id == model_id) {
      return summary.model_id;
    }
  }
  return model_id;
}

} // namespace

Storage::Storage(std::string root) : root_(std::move(root)) {
  if (root_.empty()) {
    root_ = "results";
  }
  root_ = absolute_path(root_);
  ensure_directory(root_);
}

std::string Storage::path_for(const std::string& id) const {
  std::string safe = sanitize_id(id);
  if (safe.empty()) {
    throw std::runtime_error("Invalid run id");
  }
  return (fs::path(root_) / (safe + ".json")).string();
}

std::string Storage::save_run(const RunRecord& run) {
  std::string safe = sanitize_id(run.id);
  if (safe.empty()) {
    throw std::runtime_error("Cannot save run without a valid id");
  }
  RunRecord copy = run;
  copy.id = safe;
  std::string path = path_for(safe);
  std::string temp = path + ".tmp";
  std::string content = copy.to_json(true).dump(2);
  if (!write_file(temp, content)) {
    throw std::runtime_error("Cannot write run file: " + temp);
  }
  std::error_code ec;
  fs::remove(path, ec);
  fs::rename(temp, path, ec);
  if (ec) {
    // Fall back to a direct write when rename is not possible.
    if (!write_file(path, content)) {
      throw std::runtime_error("Cannot save run file: " + path);
    }
    fs::remove(temp, ec);
  }
  return path;
}

std::vector<json> Storage::list_runs() const {
  std::vector<json> runs;
  std::error_code ec;
  if (!fs::is_directory(root_, ec)) {
    return runs;
  }
  for (const auto& entry : fs::directory_iterator(root_, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file()) {
      continue;
    }
    std::string extension = entry.path().extension().string();
    if (extension != ".json") {
      continue;
    }
    bool ok = false;
    std::string content = read_file(entry.path().string(), ok);
    if (!ok) {
      continue;
    }
    json value;
    try {
      value = json::parse(content);
    } catch (...) {
      continue;
    }
    if (!value.is_object() || !value.contains("id")) {
      continue;
    }
    json models = json::array();
    std::string best_model;
    double best_overall = -1.0;
    if (value.contains("summaries") && value["summaries"].is_array()) {
      for (const auto& summary : value["summaries"]) {
        if (!summary.is_object()) {
          continue;
        }
        double overall = summary.value("overall", 0.0);
        std::string id = summary.value("model_id", "");
        models.push_back(json{
            {"id", id},
            {"color", summary.value("color", "#38bdf8")},
            {"overall", overall},
            {"quality", summary.value("quality", 0.0)},
            {"success_rate", summary.value("success_rate", 0.0)},
            {"tokens_per_sec", summary.value("tokens_per_sec", 0.0)},
            {"p50_latency_ms", summary.value("p50_latency_ms", 0.0)},
            {"cost_per_1k_usd", summary.value("cost_per_1k_usd", 0.0)},
        });
        if (overall > best_overall) {
          best_overall = overall;
          best_model = id;
        }
      }
    }
    runs.push_back(json{
        {"id", value.value("id", "")},
        {"name", value.value("name", "")},
        {"created_at", value.value("created_at", "")},
        {"source", value.value("source", "live")},
        {"dataset_name", value.value("dataset_name", "")},
        {"request_count", value.value("total_requests", 0)},
        {"duration_ms", value.value("duration_ms", 0.0)},
        {"model_count", models.size()},
        {"best_model", best_model},
        {"best_overall", best_overall < 0.0 ? 0.0 : best_overall},
        {"models", models},
    });
  }

  std::sort(runs.begin(), runs.end(), [](const json& a, const json& b) {
    return a.value("created_at", "") > b.value("created_at", "");
  });
  return runs;
}

bool Storage::run_exists(const std::string& id) const {
  std::string safe = sanitize_id(id);
  if (safe.empty()) {
    return false;
  }
  return file_exists(path_for(safe));
}

json Storage::load_run_json(const std::string& id) const {
  std::string run_id = id;
  if (run_id.empty()) {
    run_id = latest_run_id();
  }
  if (run_id.empty()) {
    throw std::runtime_error("No runs available");
  }
  std::string path = path_for(run_id);
  bool ok = false;
  std::string content = read_file(path, ok);
  if (!ok) {
    throw std::runtime_error("Run not found: " + run_id);
  }
  try {
    return json::parse(content);
  } catch (const std::exception& error) {
    throw std::runtime_error("Invalid run file '" + run_id + "': " + error.what());
  }
}

RunRecord Storage::load_run(const std::string& id) const {
  return RunRecord::from_json(load_run_json(id));
}

std::string Storage::latest_run_id() const {
  auto runs = list_runs();
  if (runs.empty()) {
    return {};
  }
  return runs.front().value("id", "");
}

std::string Storage::export_csv(const std::string& id) const {
  RunRecord run = load_run(id);
  std::string csv = "\xEF\xBB\xBF";
  csv += "model,item_id,category,ok,passed,quality,ttft_ms,latency_ms,"
         "tokens_per_sec,prompt_tokens,completion_tokens,cost_usd,error,output\r\n";
  for (const auto& result : run.results) {
    csv += csv_field(model_name_for(run, result.model_id));
    csv += "," + csv_field(result.item_id);
    csv += "," + csv_field(result.category);
    csv += "," + csv_field(result.ok ? "true" : "false");
    csv += "," + csv_field(result.passed ? "true" : "false");
    csv += "," + format_double(result.quality, 4);
    csv += "," + format_double(result.ttft_ms, 1);
    csv += "," + format_double(result.latency_ms, 1);
    csv += "," + format_double(result.tokens_per_sec, 2);
    csv += "," + std::to_string(result.prompt_tokens);
    csv += "," + std::to_string(result.completion_tokens);
    csv += "," + format_double(result.cost_usd, 6);
    csv += "," + csv_field(result.error);
    csv += "," + csv_field(result.output);
    csv += "\r\n";
  }
  return csv;
}

} // namespace llmbench
