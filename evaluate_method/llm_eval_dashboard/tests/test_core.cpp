// Minimal dependency-free unit tests for the llmbench core library.
//
// Build with -DLLMBENCH_BUILD_TESTS=ON (the default) and run ctest or the
// llmbench_tests executable directly.

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark.h"
#include "demo.h"
#include "mock_provider.h"
#include "runner.h"
#include "storage.h"
#include "types.h"
#include "util.h"

namespace fs = std::filesystem;

namespace {

int g_checks = 0;
int g_failures = 0;

void report_failure(const char* file, int line, const std::string& message) {
  ++g_failures;
  std::cerr << "FAIL " << file << ":" << line << "  " << message << "\n";
}

#define CHECK(condition)                                                    \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(condition)) {                                                     \
      report_failure(__FILE__, __LINE__, "CHECK(" #condition ")");          \
    }                                                                       \
  } while (false)

#define CHECK_NEAR(actual, expected, tolerance)                             \
  do {                                                                      \
    ++g_checks;                                                             \
    double actual_value = static_cast<double>(actual);                      \
    double expected_value = static_cast<double>(expected);                  \
    if (std::fabs(actual_value - expected_value) > (tolerance)) {           \
      report_failure(__FILE__, __LINE__,                                    \
                     std::string(#actual " = ") +                           \
                         std::to_string(actual_value) + " (expected " +     \
                         std::to_string(expected_value) + ")");             \
    }                                                                       \
  } while (false)

#define CHECK_EQ(actual, expected)                                          \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!((actual) == (expected))) {                                        \
      report_failure(__FILE__, __LINE__, "CHECK_EQ(" #actual ", " #expected ")"); \
    }                                                                       \
  } while (false)

using namespace llmbench;

void test_strings() {
  CHECK_EQ(trim("  hello \n"), std::string("hello"));
  CHECK_EQ(to_lower_ascii("HeLLo"), std::string("hello"));
  CHECK(starts_with("benchmark", "bench"));
  CHECK(ends_with("benchmark", "mark"));
  CHECK_EQ(replace_all("a-b-c", "-", "_"), std::string("a_b_c"));
  CHECK_EQ(normalize_text("  Hello, World!  "), std::string("helloworld"));

  auto parts = split("a,b,,c", ',');
  CHECK_EQ(parts.size(), std::size_t(4));
  CHECK_EQ(parts[1], std::string("b"));
  CHECK_EQ(parts[2], std::string(""));

  CHECK_EQ(slugify("LLM Eval Dashboard!"), std::string("llm-eval-dashboard"));
  CHECK_EQ(sanitize_id("run_2026/01/01"), std::string("run_20260101"));
  CHECK_EQ(escape_csv("a\"b"), std::string("a\"\"b"));
}

void test_numbers() {
  CHECK_NEAR(clamp01(-1.0), 0.0, 1e-9);
  CHECK_NEAR(clamp01(0.4), 0.4, 1e-9);
  CHECK_NEAR(clamp01(2.0), 1.0, 1e-9);

  std::vector<double> values{10.0, 20.0, 30.0, 40.0};
  CHECK_NEAR(percentile(values, 0.5), 25.0, 1e-9);
  CHECK_NEAR(percentile(values, 0.0), 10.0, 1e-9);
  CHECK_NEAR(percentile(values, 1.0), 40.0, 1e-9);

  CHECK(fnv1a64("abc") != fnv1a64("abd"));
  CHECK_EQ(fnv1a64("abc"), fnv1a64("abc"));
}

void test_text_similarity() {
  CHECK_NEAR(char_bigram_f1("hello world", "hello world"), 1.0, 1e-9);
  CHECK_NEAR(char_bigram_f1("", ""), 0.0, 1e-9);
  CHECK(char_bigram_f1("hello", "goodbye") < 0.2);
  CHECK(estimate_tokens("") >= 0);
  CHECK(estimate_tokens("hello world") > 0);
}

void test_scoring() {
  BenchmarkItem item;
  item.id = "q1";
  item.category = "math";
  item.expected_keywords = {"12", "答え"};
  item.reference = "答えは 12 です。";

  ScoreDetail full = score_output(item, "答えは 12 です。");
  CHECK(full.matched_keywords.size() == 2);
  CHECK(full.missed_keywords.empty());
  CHECK_NEAR(full.keyword_recall, 1.0, 1e-9);
  CHECK_NEAR(full.quality, 1.0, 1e-9);

  ScoreDetail partial = score_output(item, "12");
  CHECK_EQ(partial.matched_keywords.size(), std::size_t(1));
  CHECK(partial.quality > 0.0);
  CHECK(partial.quality < 1.0);

  ScoreDetail empty = score_output(item, "   ");
  CHECK_NEAR(empty.quality, 0.0, 1e-9);
  CHECK_EQ(empty.missed_keywords.size(), std::size_t(2));

  // Alternative keywords separated by '|'.
  BenchmarkItem alt;
  alt.expected_keywords = {"東京|とうきょう", "日本"};
  ScoreDetail alt_score = score_output(alt, "とうきょうは日本の首都です");
  CHECK_EQ(alt_score.matched_keywords.size(), std::size_t(2));
  CHECK_NEAR(alt_score.quality, 1.0, 1e-9);
}

void test_summary() {
  ModelTarget model;
  model.id = "model-a";
  model.provider = "mock";
  model.color = "#123456";

  std::vector<RequestResult> results;
  for (int i = 0; i < 4; ++i) {
    RequestResult result;
    result.model_id = "model-a";
    result.item_id = "item-" + std::to_string(i);
    result.category = i < 2 ? "math" : "coding";
    result.ok = i != 3;
    result.quality = i == 0 ? 1.0 : 0.5;
    result.passed = result.quality >= 0.6;
    result.latency_ms = 100.0 * (i + 1);
    result.ttft_ms = 20.0;
    result.completion_tokens = 50;
    result.cost_usd = 0.0001;
    results.push_back(result);
  }

  ModelSummary summary = summarize_model(model, results, 0.6);
  CHECK_EQ(summary.total, 4);
  CHECK_EQ(summary.failures, 1);
  CHECK_NEAR(summary.success_rate, 75.0, 1e-9);
  CHECK_NEAR(summary.quality, 50.0, 1e-9); // (1 + .5 + .5 + 0) / 4 * 100
  CHECK_NEAR(summary.p50_latency_ms, 200.0, 1e-9);
  CHECK(summary.tokens_per_sec > 0.0);
  CHECK(summary.overall > 0.0);
  CHECK(summary.overall <= 100.0);
  CHECK_NEAR(summary.category_scores["math"], 75.0, 1e-9);

  // A model with no successful requests must not receive a positive score.
  std::vector<RequestResult> failures(2);
  for (auto& result : failures) {
    result.model_id = "model-a";
    result.category = "math";
    result.ok = false;
  }
  ModelSummary failed = summarize_model(model, failures, 0.6);
  CHECK_EQ(failed.success_rate, 0.0);
  CHECK_EQ(failed.overall, 0.0);
}

void test_select_items() {
  Dataset dataset;
  dataset.items = {
      {"a1", "alpha", "p", "", "", {}, 32, 0, 1.0},
      {"a2", "alpha", "p", "", "", {}, 32, 0, 1.0},
      {"a3", "alpha", "p", "", "", {}, 32, 0, 1.0},
      {"b1", "beta", "p", "", "", {}, 32, 0, 1.0},
      {"b2", "beta", "p", "", "", {}, 32, 0, 1.0},
  };

  CHECK_EQ(select_items(dataset, 0).size(), std::size_t(5));
  CHECK_EQ(select_items(dataset, 99).size(), std::size_t(5));

  auto four = select_items(dataset, 4);
  CHECK_EQ(four.size(), std::size_t(4));
  int alpha = 0;
  int beta = 0;
  for (const auto& item : four) {
    alpha += item.category == "alpha" ? 1 : 0;
    beta += item.category == "beta" ? 1 : 0;
  }
  CHECK_EQ(alpha, 2);
  CHECK_EQ(beta, 2);
}

void test_mock_determinism() {
  ModelTarget model;
  model.id = "mock-1";
  model.provider = "mock";
  model.mock_profile = "balanced";

  BenchmarkItem item;
  item.id = "item-1";
  item.category = "reasoning";
  item.prompt = "Explain why the sky is blue.";
  item.expected_keywords = {"scatter", "blue"};
  item.max_tokens = 128;

  MockResult first = generate_mock(model, item, 42);
  MockResult second = generate_mock(model, item, 42);
  CHECK(first.ok);
  CHECK_EQ(first.output, second.output);
  CHECK_NEAR(first.latency_ms, second.latency_ms, 1e-9);
  CHECK_EQ(first.completion_tokens, second.completion_tokens);

  MockResult other = generate_mock(model, item, 43);
  CHECK(first.output != other.output || first.latency_ms != other.latency_ms);
}

void test_run_roundtrip() {
  RunRecord run;
  run.id = "run_test_roundtrip";
  run.name = "Round trip";
  run.dataset_name = "unit";
  run.total_requests = 1;

  RequestResult result;
  result.model_id = "m";
  result.item_id = "i";
  result.category = "c";
  result.ok = true;
  result.quality = 0.75;
  result.output = "hello";
  run.results.push_back(result);

  json encoded = run.to_json(true);
  RunRecord decoded = RunRecord::from_json(encoded);
  CHECK_EQ(decoded.id, run.id);
  CHECK_EQ(decoded.name, run.name);
  CHECK_EQ(decoded.results.size(), std::size_t(1));
  CHECK_EQ(decoded.results[0].output, std::string("hello"));
  CHECK_NEAR(decoded.results[0].quality, 0.75, 1e-9);
}

void test_storage() {
  fs::path root = fs::temp_directory_path() / "llmbench_tests_storage";
  std::error_code ec;
  fs::remove_all(root, ec);

  Storage storage(root.string());
  CHECK(directory_exists(storage.root()));

  RunRecord run;
  run.id = "run_test_storage";
  run.name = "Storage test";
  run.created_at = iso_now();
  run.dataset_name = "unit";
  run.total_requests = 1;

  RequestResult result;
  result.model_id = "model-x";
  result.item_id = "item-1";
  result.category = "math";
  result.ok = true;
  result.output = "answer, with \"quotes\"";
  result.quality = 1.0;
  run.results.push_back(result);

  std::string path = storage.save_run(run);
  CHECK(file_exists(path));
  CHECK(storage.run_exists(run.id));

  auto listed = storage.list_runs();
  CHECK_EQ(listed.size(), std::size_t(1));
  CHECK_EQ(listed[0].value("id", ""), run.id);
  CHECK_EQ(storage.latest_run_id(), run.id);

  RunRecord loaded = storage.load_run(run.id);
  CHECK_EQ(loaded.results.size(), std::size_t(1));
  CHECK_EQ(loaded.results[0].output, result.output);

  std::string csv = storage.export_csv(run.id);
  CHECK(csv.find("model-x") != std::string::npos);
  CHECK(csv.find("\"\"quotes\"\"") != std::string::npos);

  fs::remove_all(root, ec);
}

void test_demo_dataset() {
  Dataset dataset = builtin_dataset();
  CHECK(!dataset.items.empty());
  for (const auto& item : dataset.items) {
    CHECK(!item.id.empty());
    CHECK(!item.prompt.empty());
    CHECK(!item.category.empty());
  }

  EvalConfig config = demo_config();
  CHECK(!config.models.empty());
  for (const auto& model : config.models) {
    CHECK_EQ(model.provider, std::string("mock"));
  }
}

} // namespace

int main() {
  test_strings();
  test_numbers();
  test_text_similarity();
  test_scoring();
  test_summary();
  test_select_items();
  test_mock_determinism();
  test_run_roundtrip();
  test_storage();
  test_demo_dataset();

  std::cout << (g_failures == 0 ? "OK" : "FAILED") << ": " << g_checks
            << " checks, " << g_failures << " failure(s)\n";
  return g_failures == 0 ? 0 : 1;
}
