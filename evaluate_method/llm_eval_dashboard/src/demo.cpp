#include "demo.h"

#include "benchmark.h"

namespace llmbench {

EvalConfig demo_config() {
  EvalConfig config;
  config.name = "Demo evaluation";
  config.dataset_path = "data/benchmark_ja.json";
  config.concurrency = 4;
  config.timeout_ms = 60000;
  config.temperature = 0.0;
  config.max_tokens = 256;
  config.pass_threshold = 0.6;
  config.retries = 1;

  config.models = {
      ModelTarget{"Nova Flash", "mock", "", "nova-flash", "", "fast", "#38bdf8", 0.10, 0.40, true},
      ModelTarget{"Atlas Balanced", "mock", "", "atlas-balanced", "", "balanced", "#a78bfa", 2.00, 8.00, true},
      ModelTarget{"Orion Pro", "mock", "", "orion-pro", "", "quality", "#34d399", 10.00, 30.00, true},
      ModelTarget{"Local Llama", "mock", "", "local-llama", "", "local", "#fbbf24", 0.0, 0.0, true},
  };
  config.raw = sanitize_config(config);
  return config;
}

Dataset builtin_dataset() {
  Dataset dataset;
  dataset.name = "Built-in mini benchmark";
  dataset.version = "1.0";

  auto add = [&](const std::string& id, const std::string& category,
                 const std::string& prompt, const std::string& reference,
                 std::vector<std::string> keywords, int max_tokens = 220) {
    BenchmarkItem item;
    item.id = id;
    item.category = category;
    item.prompt = prompt;
    item.reference = reference;
    item.expected_keywords = std::move(keywords);
    item.max_tokens = max_tokens;
    dataset.items.push_back(std::move(item));
  };

  add("reason_bat_ball", "reasoning",
      "ボールとバットの合計は1100円です。バットはボールより1000円高い。ボールの値段はいくらですか？",
      "ボールは50円、バットは1050円です。", {"50円"}, 200);
  add("code_binary_search", "coding",
      "Pythonでソート済み配列の二分探索を実装し、見つかったインデックスを返してください。",
      "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
      {"def ", "while", "mid", "return"}, 320);
  add("math_percent", "math",
      "定価8000円の商品が25%引きで売られています。税込10%を加えると何円ですか？",
      "8000 × 0.75 = 6000、税込で6600円です。", {"6600"}, 200);
  add("ja_keigo", "japanese",
      "「部長が明日来る」を尊敬語と丁寧語を使ったビジネスメールの一文にしてください。",
      "部長が明日いらっしゃいます。", {"部長", "いらっしゃ", "ます"}, 200);
  add("sum_ai", "summarization",
      "次の文章を40文字以内で要約: 生成AIは文章作成や要約、画像生成など幅広い用途で利用されている。一方で、誤情報の生成や著作権侵害のリスクも指摘されており、企業は利用ルールの整備を進めている。",
      "生成AIは便利だがリスクもあり、企業はルール整備を進めている。",
      {"生成AI", "リスク", "ルール"}, 160);
  add("inst_json", "instruction",
      "次の情報をJSONで出力してください。名前: 山田太郎, 年齢: 34, 部署: 開発部。キーは name, age, department を使い、JSON以外は出力しないでください。",
      "{\"name\":\"山田太郎\",\"age\":34,\"department\":\"開発部\"}",
      {"\"name\"", "\"age\"", "\"department\"", "山田太郎"}, 160);
  return dataset;
}

} // namespace llmbench
