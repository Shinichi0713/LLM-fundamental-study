#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "app.h"
#include "benchmark.h"
#include "demo.h"
#include "runner.h"
#include "storage.h"
#include "util.h"

namespace {

using namespace llmbench;

std::atomic<bool> g_cancel{false};

extern "C" void handle_signal(int) {
  g_cancel.store(true);
}

struct Options {
  std::string command = "serve";
  std::string config;
  std::string results;
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string web;
  std::string name;
  std::string run_id;
  std::string out;
  int max_items = 0;
  int concurrency = 0;
  int count = 3;
  bool open = false;
  bool demo = true;
  bool help = false;
  bool version = false;
};

void print_help() {
  std::cout
      << "LLM Eval Dashboard " << "1.0.0" << "\n\n"
      << "C++17 benchmark runner and self-contained dashboard for\n"
      << "OpenAI-compatible LLM endpoints.\n\n"
      << "Usage:\n"
      << "  llmbench [serve] [options]   Start the dashboard (default)\n"
      << "  llmbench run [options]       Run an evaluation once\n"
      << "  llmbench demo [options]      Generate demo runs\n"
      << "  llmbench list [options]      List saved runs\n"
      << "  llmbench export [options]    Export a run as CSV\n\n"
      << "Options:\n"
      << "  -c, --config PATH       Config JSON (default: configs/example.json)\n"
      << "  -r, --results DIR       Results directory (default: ./results)\n"
      << "  -H, --host ADDR         Bind address (default: 127.0.0.1)\n"
      << "  -p, --port N            HTTP port (default: 8080)\n"
      << "  -w, --web DIR           Dashboard assets directory\n"
      << "      --open              Open the dashboard in a browser\n"
      << "      --no-demo           Do not seed demo runs on an empty results dir\n"
      << "      --max-items N       Evaluate only N prompts (0 = all)\n"
      << "      --concurrency N     Override worker count (1-12)\n"
      << "      --name NAME         Run name\n"
      << "      --id RUN_ID         Run id for export (default: latest)\n"
      << "  -o, --out PATH          CSV output path for export\n"
      << "      --count N           Number of demo runs to generate (default: 3)\n"
      << "  -h, --help              Show this help\n"
      << "      --version           Show version\n\n"
      << "Examples:\n"
      << "  llmbench serve --open\n"
      << "  llmbench run --config configs/example.json --max-items 12\n"
      << "  llmbench export --id run_20260101_120000_demo_1 --out report.csv\n\n"
      << "Environment:\n"
      << "  API keys are read from the env var named by each model's\n"
      << "  \"api_key_env\" field (for example OPENAI_API_KEY).\n";
}

bool parse_options(int argc, char** argv, Options& options) {
  int index = 1;
  if (argc > 1) {
    std::string first = argv[1];
    if (!first.empty() && first[0] != '-') {
      options.command = first;
      index = 2;
    }
  }

  auto require_value = [&](int& i, const std::string& flag) -> std::string {
    if (i + 1 >= argc) {
      std::cerr << "Missing value for " << flag << "\n";
      return {};
    }
    return argv[++i];
  };

  for (; index < argc; ++index) {
    std::string arg = argv[index];
    std::string value;
    auto equals = arg.find('=');
    if (equals != std::string::npos) {
      value = arg.substr(equals + 1);
      arg = arg.substr(0, equals);
    }

    auto take = [&](const std::string& flag) {
      if (value.empty()) {
        value = require_value(index, flag);
      }
      return value;
    };

    if (arg == "-h" || arg == "--help") {
      options.help = true;
    } else if (arg == "--version") {
      options.version = true;
    } else if (arg == "-c" || arg == "--config") {
      options.config = take(arg);
    } else if (arg == "-r" || arg == "--results") {
      options.results = take(arg);
    } else if (arg == "-H" || arg == "--host") {
      options.host = take(arg);
    } else if (arg == "-p" || arg == "--port") {
      try {
        options.port = std::stoi(take(arg));
      } catch (...) {
        std::cerr << "Invalid port\n";
        return false;
      }
    } else if (arg == "-w" || arg == "--web") {
      options.web = take(arg);
    } else if (arg == "--open") {
      options.open = true;
    } else if (arg == "--no-demo") {
      options.demo = false;
    } else if (arg == "--max-items") {
      try {
        options.max_items = std::stoi(take(arg));
      } catch (...) {
        std::cerr << "Invalid --max-items\n";
        return false;
      }
    } else if (arg == "--concurrency") {
      try {
        options.concurrency = std::stoi(take(arg));
      } catch (...) {
        std::cerr << "Invalid --concurrency\n";
        return false;
      }
    } else if (arg == "--name") {
      options.name = take(arg);
    } else if (arg == "--id") {
      options.run_id = take(arg);
    } else if (arg == "-o" || arg == "--out") {
      options.out = take(arg);
    } else if (arg == "--count") {
      try {
        options.count = std::stoi(take(arg));
      } catch (...) {
        std::cerr << "Invalid --count\n";
        return false;
      }
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      return false;
    }
  }

  if (options.port < 1 || options.port > 65535) {
    std::cerr << "Port must be between 1 and 65535\n";
    return false;
  }
  options.max_items = std::max(0, options.max_items);
  options.concurrency = std::max(0, std::min(12, options.concurrency));
  options.count = std::max(1, std::min(10, options.count));
  return true;
}

std::string find_file(const std::string& path) {
  if (path.empty()) {
    return {};
  }
  if (file_exists(path)) {
    return absolute_path(path);
  }
  std::string resolved = resolve_existing_path(path);
  return resolved;
}

EvalConfig load_config_or_demo(const std::string& requested, bool& used_demo) {
  std::string path = find_file(requested);
  if (path.empty() && requested.empty()) {
    path = find_file("configs/example.json");
  }
  if (path.empty()) {
    used_demo = true;
    return demo_config();
  }
  used_demo = false;
  return load_config(path);
}

Dataset load_dataset_or_builtin(const std::string& requested, bool& used_builtin) {
  std::string path = find_file(requested);
  if (path.empty()) {
    used_builtin = true;
    return builtin_dataset();
  }
  used_builtin = false;
  return load_dataset(path);
}

std::string results_dir(const std::string& requested) {
  if (!requested.empty()) {
    return absolute_path(requested);
  }
  std::string existing = resolve_existing_path("results");
  if (!existing.empty()) {
    return existing;
  }
  return absolute_path("results");
}

void print_summary(const RunRecord& run) {
  std::cout << "\n"
            << std::left << std::setw(24) << "Model" << std::right
            << std::setw(9) << "Overall" << std::setw(9) << "Quality"
            << std::setw(9) << "Success" << std::setw(9) << "tok/s"
            << std::setw(10) << "p50 ms" << std::setw(11) << "Cost/1k"
            << "\n";
  std::cout << std::string(81, '-') << "\n";
  for (const auto& summary : run.summaries) {
    std::cout << std::left << std::setw(24) << summary.model_id << std::right
              << std::setw(9) << format_double(summary.overall, 1)
              << std::setw(9) << format_double(summary.quality, 1)
              << std::setw(8) << format_double(summary.success_rate, 1) << "%"
              << std::setw(9) << format_double(summary.tokens_per_sec, 1)
              << std::setw(10) << format_double(summary.p50_latency_ms, 0)
              << std::setw(11) << ("$" + format_double(summary.cost_per_1k_usd, 3))
              << "\n";
  }
  std::cout << "\n";
}

int command_serve(const Options& options) {
  bool config_is_demo = false;
  bool dataset_is_builtin = false;
  EvalConfig config = load_config_or_demo(options.config, config_is_demo);
  Dataset dataset = load_dataset_or_builtin(config.dataset_path, dataset_is_builtin);

  if (options.concurrency > 0) {
    config.concurrency = options.concurrency;
  }
  if (!options.name.empty()) {
    config.name = options.name;
  }

  Storage storage(results_dir(options.results));
  if (options.demo && storage.list_runs().empty()) {
    std::cout << "No saved runs found. Generating demo runs...\n";
    generate_demo_runs(storage, dataset, options.count);
  }

  if (config_is_demo) {
    std::cout << "Using built-in demo configuration (no config file found).\n";
  }
  if (dataset_is_builtin) {
    std::cout << "Using built-in mini dataset (data/benchmark_ja.json not found).\n";
  }

  AppOptions app_options;
  app_options.host = options.host;
  app_options.port = options.port;
  app_options.web_dir = options.web;
  app_options.open_browser = options.open;

  App app(app_options, std::move(config), std::move(dataset), storage);
  return app.run() ? 0 : 1;
}

int command_run(const Options& options) {
  bool config_is_demo = false;
  bool dataset_is_builtin = false;
  EvalConfig config = load_config_or_demo(options.config, config_is_demo);
  Dataset dataset = load_dataset_or_builtin(config.dataset_path, dataset_is_builtin);

  if (options.concurrency > 0) {
    config.concurrency = options.concurrency;
  }
  if (!options.name.empty()) {
    config.name = options.name;
  }

  config.models.erase(
      std::remove_if(config.models.begin(), config.models.end(),
                     [](const ModelTarget& model) { return !model.enabled; }),
      config.models.end());
  if (config.models.empty()) {
    std::cerr << "No enabled models in config\n";
    return 1;
  }

  std::vector<BenchmarkItem> items = select_items(dataset, options.max_items);
  int total = static_cast<int>(items.size() * config.models.size());

  if (config_is_demo) {
    std::cout << "Using built-in demo configuration.\n";
  }
  if (dataset_is_builtin) {
    std::cout << "Using built-in mini dataset.\n";
  }
  std::cout << "Evaluating " << config.models.size() << " model(s) x "
            << items.size() << " prompt(s) = " << total << " requests\n\n";

  std::signal(SIGINT, handle_signal);
#ifdef SIGTERM
  std::signal(SIGTERM, handle_signal);
#endif
  g_cancel.store(false);

  std::atomic<int> completed{0};
  std::mutex output_mutex;
  RunOptions run_options;
  run_options.max_items = options.max_items;
  run_options.seed = static_cast<std::uint64_t>(
      std::chrono::system_clock::now().time_since_epoch().count());
  run_options.cancel = &g_cancel;

  RunRecord run = run_evaluation(
      config, dataset, config.models, run_options,
      [&](int, const RequestResult& result) {
        int done = completed.fetch_add(1) + 1;
        int percent = total > 0 ? static_cast<int>(100.0 * done / total) : 0;
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << "\r[" << std::setw(3) << percent << "%] " << std::setw(3)
                  << done << "/" << total << "  " << std::left << std::setw(24)
                  << result.model_id.substr(0, 24) << std::setw(18)
                  << result.item_id.substr(0, 18)
                  << (result.ok ? "OK" : "FAILED") << std::flush;
      });
  std::cout << "\n";

  if (run.results.empty()) {
    std::cerr << "No results were collected.\n";
    return 1;
  }

  Storage storage(results_dir(options.results));
  std::string path = storage.save_run(run);
  print_summary(run);

  std::cout << "Saved: " << path << "\n";
  std::cout << "Requests: " << run.total_requests << "   Duration: "
            << format_duration(run.duration_ms) << "\n";
  if (g_cancel.load()) {
    std::cout << "Evaluation was cancelled; partial results were saved.\n";
  }
  std::cout << "Run `llmbench serve --open` to explore the dashboard.\n";
  return 0;
}

int command_demo(const Options& options) {
  bool dataset_is_builtin = false;
  Dataset dataset = load_dataset_or_builtin("data/benchmark_ja.json", dataset_is_builtin);
  if (dataset_is_builtin) {
    std::cout << "Using built-in mini dataset.\n";
  }
  Storage storage(results_dir(options.results));
  std::cout << "Generating " << options.count << " demo run(s)...\n";
  generate_demo_runs(storage, dataset, options.count);
  std::cout << "Done. Results: " << storage.root() << "\n";
  std::cout << "Run `llmbench serve --open` to explore the dashboard.\n";
  return 0;
}

int command_list(const Options& options) {
  Storage storage(results_dir(options.results));
  auto runs = storage.list_runs();
  if (runs.empty()) {
    std::cout << "No runs in " << storage.root() << "\n";
    return 0;
  }
  std::cout << std::left << std::setw(38) << "Run id" << std::setw(12) << "Source"
            << std::setw(22) << "Created" << std::setw(10) << "Requests"
            << std::setw(28) << "Best model" << "Overall\n";
  std::cout << std::string(120, '-') << "\n";
  for (const auto& run : runs) {
    std::cout << std::left << std::setw(38) << run.value("id", "")
              << std::setw(12) << run.value("source", "")
              << std::setw(22) << run.value("created_at", "")
              << std::setw(10) << run.value("request_count", 0)
              << std::setw(28) << run.value("best_model", "")
              << format_double(run.value("best_overall", 0.0), 1) << "\n";
  }
  return 0;
}

int command_export(const Options& options) {
  Storage storage(results_dir(options.results));
  std::string id = options.run_id;
  if (id.empty()) {
    id = storage.latest_run_id();
  }
  if (id.empty()) {
    std::cerr << "No runs available to export\n";
    return 1;
  }
  std::string csv = storage.export_csv(id);
  std::string out = options.out.empty() ? id + ".csv" : options.out;
  if (!write_file(out, csv)) {
    std::cerr << "Could not write " << out << "\n";
    return 1;
  }
  std::cout << "Exported " << id << " to " << absolute_path(out) << "\n";
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, options)) {
    return 2;
  }
  if (options.help) {
    print_help();
    return 0;
  }
  if (options.version) {
    std::cout << "llmbench 1.0.0\n";
    return 0;
  }

  try {
    if (options.command == "serve" || options.command == "dashboard") {
      return command_serve(options);
    }
    if (options.command == "run") {
      return command_run(options);
    }
    if (options.command == "demo") {
      return command_demo(options);
    }
    if (options.command == "list" || options.command == "ls") {
      return command_list(options);
    }
    if (options.command == "export") {
      return command_export(options);
    }
    std::cerr << "Unknown command: " << options.command << "\n\n";
    print_help();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
