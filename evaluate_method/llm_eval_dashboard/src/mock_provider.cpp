#include "mock_provider.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "util.h"

namespace llmbench {

namespace {

struct Profile {
  double quality;
  double tokens_per_sec;
  double ttft_ms;
  int failure_permille;
  std::map<std::string, double> category_bonus;
};

Profile profile_for(const std::string& name) {
  if (name == "fast") {
    return Profile{0.70, 118.0, 130.0, 8,
                   {{"reasoning", 0.00},
                    {"coding", -0.11},
                    {"math", -0.05},
                    {"japanese", 0.03},
                    {"summarization", 0.02},
                    {"instruction", -0.04}}};
  }
  if (name == "quality") {
    return Profile{0.93, 34.0, 470.0, 2,
                   {{"reasoning", 0.02},
                    {"coding", 0.03},
                    {"math", 0.03},
                    {"japanese", 0.01},
                    {"summarization", 0.01},
                    {"instruction", 0.02}}};
  }
  if (name == "local") {
    return Profile{0.77, 25.0, 360.0, 14,
                   {{"reasoning", -0.03},
                    {"coding", -0.06},
                    {"math", -0.05},
                    {"japanese", 0.04},
                    {"summarization", 0.03},
                    {"instruction", -0.02}}};
  }
  return Profile{0.84, 60.0, 230.0, 4,
                 {{"reasoning", 0.00},
                  {"coding", -0.01},
                  {"math", 0.00},
                  {"japanese", 0.01},
                  {"summarization", 0.00},
                  {"instruction", 0.00}}};
}

std::string join(const std::vector<std::string>& values, const std::string& separator) {
  std::string result;
  for (const auto& value : values) {
    if (!result.empty()) {
      result += separator;
    }
    result += value;
  }
  return result;
}

std::string reference_snippet(const std::string& reference, double quality) {
  if (reference.empty()) {
    return {};
  }
  auto chars = utf8_chars(reference);
  double fraction = 0.15 + 0.80 * quality;
  std::size_t keep = std::max<std::size_t>(
      12, static_cast<std::size_t>(std::llround(static_cast<double>(chars.size()) * fraction)));
  keep = std::min(keep, chars.size());
  std::string result;
  for (std::size_t i = 0; i < keep; ++i) {
    result += chars[i];
  }
  return result;
}

std::string filter_code(const std::string& reference,
                        const std::vector<std::string>& matched,
                        const std::vector<std::string>& missed) {
  std::vector<std::string> lines = split(reference, '\n');
  std::vector<std::string> kept;
  for (const auto& line : lines) {
    bool drop = false;
    for (const auto& keyword : missed) {
      for (const auto& alternative : split(keyword, '|')) {
        std::string needle = trim(alternative);
        if (!needle.empty() && line.find(needle) != std::string::npos) {
          drop = true;
          break;
        }
      }
      if (drop) {
        break;
      }
    }
    if (!drop) {
      kept.push_back(line);
    }
  }
  if (kept.empty()) {
    kept.push_back("def solve():");
    kept.push_back("    pass");
  }
  std::string code = join(kept, "\n");
  if (!matched.empty()) {
    code += "\n# 確認した要素: " + join(matched, ", ");
  }
  return code;
}

std::string instruction_output(const BenchmarkItem& item,
                               const std::vector<std::string>& matched,
                               const std::vector<std::string>& missed,
                               double quality) {
  auto has = [&](const std::string& value) {
    return std::find(matched.begin(), matched.end(), value) != matched.end();
  };

  if (item.id == "inst_json") {
    std::string output = "{";
    bool first = true;
    if (has("\"name\"")) {
      output += "\"name\":\"山田太郎\"";
      first = false;
    }
    if (has("\"age\"")) {
      if (!first) output += ",";
      output += "\"age\":34";
      first = false;
    }
    if (has("\"department\"")) {
      if (!first) output += ",";
      output += "\"department\":\"開発部\"";
      first = false;
    }
    output += "}";
    if (!missed.empty()) {
      output += "\n補足: 一部のキーを出力できませんでした。";
    }
    return output;
  }

  if (item.id == "inst_refuse") {
    std::string output = "申し訳ありませんが、個人情報を無断で収集する方法はお教えできません。";
    if (!matched.empty()) {
      output += "\n" + join(matched, "、") + "を尊重する必要があります。";
    }
    if (quality < 0.5) {
      output += "\n（安全上の配慮が不十分な回答です）";
    }
    return output;
  }

  std::string output;
  int number = 1;
  for (const auto& keyword : matched) {
    output += std::to_string(number++) + ". " + keyword + "\n";
  }
  if (matched.empty()) {
    output = "回答を生成できませんでした。\n";
  }
  std::string snippet = reference_snippet(item.reference, quality);
  if (!snippet.empty()) {
    output += "\n" + snippet;
  }
  return output;
}

} // namespace

MockResult generate_mock(const ModelTarget& model,
                         const BenchmarkItem& item,
                         std::uint64_t seed) {
  MockResult result;
  Profile profile = profile_for(model.mock_profile);

  std::uint64_t state = fnv1a64(model.id + "|" + item.id + "|" + std::to_string(seed));
  auto next = [&]() {
    state = mix64(state);
    return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
  };

  double category_bonus = 0.0;
  auto it = profile.category_bonus.find(item.category);
  if (it != profile.category_bonus.end()) {
    category_bonus = it->second;
  }
  double quality = profile.quality + category_bonus + (next() - 0.5) * 0.11;
  quality = std::max(0.18, std::min(0.99, quality));

  result.prompt_tokens = estimate_tokens(item.system + item.prompt) + 12;
  result.ttft_ms = profile.ttft_ms + next() * 180.0;

  int roll = static_cast<int>(next() * 1000.0);
  if (roll < profile.failure_permille) {
    result.ok = false;
    result.error = (roll % 2 == 0)
                       ? "HTTP 429: rate limit exceeded (simulated)"
                       : "HTTP 500: upstream timeout (simulated)";
    result.latency_ms = result.ttft_ms + 180.0 + next() * 900.0;
    result.tokens_per_sec = 0.0;
    result.completion_tokens = 0;
    return result;
  }

  // Decide which keywords this simulated model manages to cover.
  std::vector<std::string> keywords = item.expected_keywords;
  std::sort(keywords.begin(), keywords.end(),
            [&](const std::string& a, const std::string& b) {
              return fnv1a64(a + std::to_string(seed)) < fnv1a64(b + std::to_string(seed));
            });
  std::size_t target_matches = static_cast<std::size_t>(
      std::llround(static_cast<double>(keywords.size()) * quality));
  std::vector<std::string> matched(keywords.begin(),
                                   keywords.begin() + std::min(target_matches, keywords.size()));
  std::vector<std::string> missed(keywords.begin() + std::min(target_matches, keywords.size()),
                                  keywords.end());

  std::string joined = join(matched, "、");
  std::string snippet = reference_snippet(item.reference, quality);

  if (item.category == "coding") {
    result.output = "```python\n" + filter_code(item.reference, matched, missed) + "\n```";
  } else if (item.category == "instruction") {
    result.output = instruction_output(item, matched, missed, quality);
  } else if (item.category == "reasoning") {
    result.output = "考え方:\n1. 与えられた条件を整理します。\n2. 計算と検証を行います。\n";
    if (!joined.empty()) {
      result.output += "結論: " + joined + "。\n";
    }
    result.output += snippet;
  } else if (item.category == "math") {
    result.output = "計算します。\n";
    if (!joined.empty()) {
      result.output += joined + "。\n";
    }
    result.output += snippet;
  } else if (item.category == "japanese") {
    result.output = "承知いたしました。";
    if (!joined.empty()) {
      result.output += joined + "。";
    }
    result.output += "\n" + snippet;
  } else if (item.category == "summarization") {
    result.output = "要約: ";
    if (!joined.empty()) {
      result.output += joined + "。";
    }
    result.output += "\n" + snippet;
  } else {
    result.output = joined;
    if (!snippet.empty()) {
      result.output += "\n" + snippet;
    }
  }

  result.completion_tokens = std::max(10, estimate_tokens(result.output));
  double tps = profile.tokens_per_sec * (0.86 + next() * 0.28);
  result.tokens_per_sec = tps;
  double generation_ms = static_cast<double>(result.completion_tokens) / tps * 1000.0;
  result.latency_ms = result.ttft_ms + generation_ms * (0.94 + next() * 0.12);
  result.cost_usd = static_cast<double>(result.prompt_tokens) / 1e6 * model.input_price_per_mtok +
                    static_cast<double>(result.completion_tokens) / 1e6 *
                        model.output_price_per_mtok;
  return result;
}

} // namespace llmbench
