#include "app.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <sstream>

#include "benchmark.h"
#include "demo.h"
#include "embedded_assets.h"
#include "httplib.h"
#include "runner.h"
#include "util.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <shellapi.h>
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace llmbench {

namespace {

using Clock = std::chrono::steady_clock;

void send_json(httplib::Response& res, const json& value, int status = 200) {
  res.status = status;
  res.set_header("Cache-Control", "no-store");
  res.set_content(value.dump(), "application/json; charset=utf-8");
}

void send_error(httplib::Response& res, int status, const std::string& message) {
  send_json(res, json{{"error", message}}, status);
}

std::string content_type_for(const std::string& name) {
  if (name == "index.html") return "text/html; charset=utf-8";
  if (name == "styles.css") return "text/css; charset=utf-8";
  if (name == "app.js") return "application/javascript; charset=utf-8";
  return "application/octet-stream";
}

std::string pick_color(std::size_t index) {
  static const char* colors[] = {"#38bdf8", "#a78bfa", "#34d399", "#fbbf24",
                                 "#fb7185", "#22d3ee", "#f472b6", "#a3e635",
                                 "#f97316", "#818cf8"};
  return colors[index % (sizeof(colors) / sizeof(colors[0]))];
}

bool parse_custom_model(const json& entry,
                        std::size_t index,
                        ModelTarget& model,
                        std::string& error) {
  if (!entry.is_object()) {
    error = "custom_models entries must be objects";
    return false;
  }
  model.model = entry.value("model", "");
  model.id = entry.value("id", model.model.empty() ? "custom_" + std::to_string(index + 1)
                                                   : model.model);
  model.provider = to_lower_ascii(entry.value("provider", "openai"));
  model.base_url = entry.value("base_url", "http://localhost:11434/v1");
  model.api_key_env = entry.value("api_key_env", "");
  model.mock_profile = entry.value("profile", "balanced");
  model.color = entry.value("color", pick_color(index));
  model.input_price_per_mtok = entry.value("input_price_per_mtok", 0.0);
  model.output_price_per_mtok = entry.value("output_price_per_mtok", 0.0);
  model.enabled = entry.value("enabled", true);

  if (model.provider != "openai" && model.provider != "mock") {
    error = "model '" + model.id + "': provider must be openai or mock";
    return false;
  }
  if (model.provider == "openai" && model.model.empty()) {
    error = "model '" + model.id + "': model name is required";
    return false;
  }
  if (model.provider == "openai" && !starts_with(model.base_url, "http://") &&
      !starts_with(model.base_url, "https://")) {
    error = "model '" + model.id + "': base_url must start with http:// or https://";
    return false;
  }
  if (model.provider == "mock" && model.model.empty()) {
    model.model = model.id;
  }
  return true;
}

json dataset_items_json(const Dataset& dataset) {
  json items = json::array();
  for (const auto& item : dataset.items) {
    items.push_back(json{{"id", item.id},
                         {"category", item.category},
                         {"prompt", item.prompt},
                         {"system", item.system},
                         {"expected_keywords", item.expected_keywords},
                         {"max_tokens", item.max_tokens}});
  }
  return items;
}

} // namespace

// ---------------------------------------------------------------------------
// JobManager
// ---------------------------------------------------------------------------

struct JobManager::Job {
  std::string id;
  std::string name;
  std::string source = "live";
  std::string state = "running";
  std::string message = "Starting...";
  std::string run_id;
  int total = 0;
  int completed = 0;
  Clock::time_point started;
  std::atomic<bool> cancel{false};
  std::atomic<bool> active{true};
  std::thread thread;
  std::vector<ModelProgress> models;
  mutable std::mutex mutex;
  std::shared_ptr<RunRecord> run;
};

JobManager::JobManager(Storage& storage, Dataset dataset, EvalConfig config)
    : storage_(storage), default_dataset_(std::move(dataset)),
      default_config_(std::move(config)) {}

JobManager::~JobManager() {
  cancel();
  wait();
}

bool JobManager::start(const EvalConfig& config,
                       const Dataset& dataset,
                       int max_items,
                       const std::string& source,
                       std::string& error) {
  std::shared_ptr<Job> previous;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (job_ && job_->active.load()) {
      error = "An evaluation is already running";
      return false;
    }
    previous = job_;
  }
  if (previous && previous->thread.joinable()) {
    previous->thread.join();
  }

  auto job = std::make_shared<Job>();
  job->id = "job_" + file_stamp_now();
  job->name = config.name;
  job->source = source;
  job->started = Clock::now();

  std::vector<BenchmarkItem> items = select_items(dataset, max_items);
  job->total = static_cast<int>(items.size() * config.models.size());
  for (const auto& model : config.models) {
    ModelProgress progress;
    progress.id = model.id;
    progress.color = model.color;
    progress.total = static_cast<int>(items.size());
    job->models.push_back(std::move(progress));
  }

  const std::uint64_t seed = static_cast<std::uint64_t>(
      std::chrono::system_clock::now().time_since_epoch().count());

  {
    std::lock_guard<std::mutex> lock(mutex_);
    job->thread = std::thread([this, job, config, dataset, max_items, source, seed]() {
      try {
        RunOptions options;
        options.max_items = max_items;
        options.seed = seed;
        options.cancel = &job->cancel;

        RunRecord run = run_evaluation(
            config, dataset, config.models, options,
            [job](int model_index, const RequestResult& result) {
              std::lock_guard<std::mutex> lock(job->mutex);
              job->completed += 1;
              if (model_index >= 0 &&
                  model_index < static_cast<int>(job->models.size())) {
                auto& progress = job->models[static_cast<std::size_t>(model_index)];
                progress.completed += 1;
                if (result.ok) {
                  progress.quality_sum += result.quality;
                  progress.quality_count += 1;
                  if (result.passed) {
                    progress.passed += 1;
                  }
                } else {
                  progress.failures += 1;
                }
              }
              job->message = result.model_id + " / " + result.item_id +
                             (result.ok ? "  OK" : "  FAILED");
            });

        run.source = source;
        run.name = config.name;
        run.id = "run_" + file_stamp_now() + "_" + slugify(config.name) + "_" +
                 std::to_string(seed % 10000);
        run.config["name"] = run.name;

        if (run.total_requests > 0) {
          try {
            storage_.save_run(run);
          } catch (const std::exception& save_error) {
            std::lock_guard<std::mutex> lock(job->mutex);
            job->message = std::string("Could not save run: ") + save_error.what();
          }
        }

        std::lock_guard<std::mutex> lock(job->mutex);
        job->run = std::make_shared<RunRecord>(run);
        job->run_id = run.id;
        if (job->cancel.load()) {
          job->state = "cancelled";
          job->message = "Cancelled after " + std::to_string(job->completed) +
                         " requests";
        } else {
          job->state = "completed";
          job->message = "Completed " + std::to_string(job->completed) +
                         " requests";
        }
        job->active.store(false);
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(job->mutex);
        job->state = "failed";
        job->message = error.what();
        job->active.store(false);
      }
    });
    job_ = job;
  }
  return true;
}

bool JobManager::cancel() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!job_ || !job_->active.load()) {
    return false;
  }
  job_->cancel.store(true);
  {
    std::lock_guard<std::mutex> job_lock(job_->mutex);
    job_->message = "Cancelling...";
  }
  return true;
}

bool JobManager::running() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return job_ && job_->active.load();
}

JobSnapshot JobManager::snapshot() const {
  JobSnapshot snapshot;
  std::shared_ptr<Job> job;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    job = job_;
  }
  if (!job) {
    snapshot.state = "idle";
    return snapshot;
  }
  {
    std::lock_guard<std::mutex> lock(job->mutex);
    snapshot.id = job->id;
    snapshot.name = job->name;
    snapshot.state = job->state;
    snapshot.message = job->message;
    snapshot.completed = job->completed;
    snapshot.total = job->total;
    snapshot.run_id = job->run_id;
    snapshot.models = job->models;
    snapshot.elapsed_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - job->started).count();
  }
  if (!job->active.load() && snapshot.state == "running") {
    snapshot.state = "completed";
  }
  return snapshot;
}

json JobManager::snapshot_json() const {
  JobSnapshot snapshot = this->snapshot();
  json models = json::array();
  for (const auto& model : snapshot.models) {
    double quality = model.quality_count > 0
                         ? 100.0 * model.quality_sum / model.quality_count
                         : 0.0;
    models.push_back(json{
        {"id", model.id},
        {"color", model.color},
        {"completed", model.completed},
        {"total", model.total},
        {"quality", quality},
        {"passed", model.passed},
        {"failures", model.failures},
    });
  }
  double percent = snapshot.total > 0
                       ? 100.0 * snapshot.completed / snapshot.total
                       : 0.0;
  return json{
      {"id", snapshot.id},
      {"name", snapshot.name},
      {"state", snapshot.state},
      {"message", snapshot.message},
      {"completed", snapshot.completed},
      {"total", snapshot.total},
      {"percent", percent},
      {"elapsed_ms", snapshot.elapsed_ms},
      {"run_id", snapshot.run_id},
      {"models", models},
  };
}

void JobManager::wait() {
  std::shared_ptr<Job> job;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    job = job_;
  }
  if (job && job->thread.joinable()) {
    job->thread.join();
  }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

App::App(AppOptions options, EvalConfig config, Dataset dataset, Storage& storage)
    : options_(std::move(options)), config_(std::move(config)),
      dataset_(std::move(dataset)), storage_(storage),
      jobs_(storage, dataset_, config_), server_(new httplib::Server()) {
  std::string candidate;
  if (!options_.web_dir.empty()) {
    candidate = absolute_path(options_.web_dir);
  } else {
    candidate = resolve_existing_path("web");
  }
  if (!candidate.empty() && file_exists((fs::path(candidate) / "index.html").string())) {
    web_dir_ = candidate;
  }
  setup_routes();
}

App::~App() = default;

std::string App::asset(const std::string& name) const {
  if (!web_dir_.empty()) {
    bool ok = false;
    std::string content = read_file((fs::path(web_dir_) / name).string(), ok);
    if (ok) {
      return content;
    }
  }
  if (name == "index.html") {
    return embedded::index_html();
  }
  if (name == "styles.css") {
    return embedded::styles_css();
  }
  if (name == "app.js") {
    return embedded::app_js();
  }
  return {};
}

void App::setup_routes() {
  server_->set_payload_max_length(2 * 1024 * 1024);
  server_->set_keep_alive_max_count(50);

  server_->Get("/", [this](const httplib::Request&, httplib::Response& res) {
    res.set_content(asset("index.html"), content_type_for("index.html"));
    res.set_header("Cache-Control", "no-cache");
  });
  server_->Get("/styles.css", [this](const httplib::Request&, httplib::Response& res) {
    res.set_content(asset("styles.css"), content_type_for("styles.css"));
    res.set_header("Cache-Control", "no-cache");
  });
  server_->Get("/app.js", [this](const httplib::Request&, httplib::Response& res) {
    res.set_content(asset("app.js"), content_type_for("app.js"));
    res.set_header("Cache-Control", "no-cache");
  });

  server_->Get("/api/health", [this](const httplib::Request&, httplib::Response& res) {
    send_json(res, json{{"ok", true},
                        {"app", "LLM Eval Dashboard"},
                        {"version", "1.0.0"},
                        {"uptime_ms",
                         std::chrono::duration<double, std::milli>(
                             Clock::now() - started_)
                             .count()}});
  });

  server_->Get("/api/bootstrap", [this](const httplib::Request&, httplib::Response& res) {
    try {
      json runs = storage_.list_runs();
      json latest = nullptr;
      if (!runs.empty()) {
        latest = storage_.load_run_json(runs.front().value("id", ""));
      }
      std::vector<std::string> categories;
      for (const auto& item : dataset_.items) {
        if (std::find(categories.begin(), categories.end(), item.category) ==
            categories.end()) {
          categories.push_back(item.category);
        }
      }
      send_json(res, json{
                         {"app", {{"name", "LLM Eval Dashboard"}, {"version", "1.0.0"}}},
                         {"config", sanitize_config(config_)},
                         {"dataset",
                          {{"name", dataset_.name},
                           {"count", dataset_.items.size()},
                           {"categories", categories},
                           {"items", dataset_items_json(dataset_)}}},
                         {"runs", runs},
                         {"latest", latest},
                         {"job", jobs_.snapshot_json()},
                     });
    } catch (const std::exception& error) {
      send_error(res, 500, error.what());
    }
  });

  server_->Get("/api/config", [this](const httplib::Request&, httplib::Response& res) {
    send_json(res, sanitize_config(config_));
  });

  server_->Get("/api/dataset", [this](const httplib::Request&, httplib::Response& res) {
    send_json(res, json{{"name", dataset_.name},
                        {"count", dataset_.items.size()},
                        {"items", dataset_items_json(dataset_)}});
  });

  server_->Get("/api/runs", [this](const httplib::Request&, httplib::Response& res) {
    send_json(res, json{{"runs", storage_.list_runs()}});
  });

  server_->Get(R"(/api/runs/([A-Za-z0-9_\-]+)/csv)",
               [this](const httplib::Request& req, httplib::Response& res) {
                 try {
                   std::string id = req.matches[1];
                   std::string csv = storage_.export_csv(id);
                   res.set_header("Content-Disposition",
                                  "attachment; filename=\"" + id + ".csv\"");
                   res.set_content(csv, "text/csv; charset=utf-8");
                 } catch (const std::exception& error) {
                   send_error(res, 404, error.what());
                 }
               });

  server_->Get(R"(/api/runs/([A-Za-z0-9_\-]+))",
               [this](const httplib::Request& req, httplib::Response& res) {
                 try {
                   send_json(res, storage_.load_run_json(req.matches[1]));
                 } catch (const std::exception& error) {
                   send_error(res, 404, error.what());
                 }
               });

  server_->Get("/api/job", [this](const httplib::Request&, httplib::Response& res) {
    send_json(res, jobs_.snapshot_json());
  });

  server_->Post("/api/job/cancel", [this](const httplib::Request&, httplib::Response& res) {
    bool cancelled = jobs_.cancel();
    send_json(res, json{{"ok", true}, {"cancelled", cancelled}});
  });

  server_->Post("/api/run", [this](const httplib::Request& req, httplib::Response& res) {
    json body = json::object();
    if (!req.body.empty()) {
      try {
        body = json::parse(req.body);
      } catch (const std::exception& error) {
        send_error(res, 400, std::string("Invalid JSON: ") + error.what());
        return;
      }
    }
    if (!body.is_object()) {
      send_error(res, 400, "Request body must be an object");
      return;
    }

    EvalConfig run_config = config_;
    std::vector<ModelTarget> models;
    std::set<std::string> wanted;
    bool explicit_selection = false;

    if (body.contains("model_ids") && body["model_ids"].is_array() &&
        !body["model_ids"].empty()) {
      explicit_selection = true;
      for (const auto& id : body["model_ids"]) {
        if (id.is_string()) {
          wanted.insert(id.get<std::string>());
        }
      }
      for (const auto& model : config_.models) {
        if (wanted.count(model.id) > 0) {
          models.push_back(model);
        }
      }
    } else {
      models = config_.models;
    }

    if (body.contains("custom_models") && body["custom_models"].is_array()) {
      std::size_t index = 0;
      for (const auto& entry : body["custom_models"]) {
        ModelTarget model;
        std::string error;
        if (!parse_custom_model(entry, models.size() + index, model, error)) {
          send_error(res, 400, error);
          return;
        }
        models.push_back(std::move(model));
        ++index;
      }
    }

    // Explicitly selected models may be disabled in the config file; only
    // apply the "enabled" filter when the caller did not choose models.
    if (!explicit_selection) {
      models.erase(std::remove_if(models.begin(), models.end(),
                                  [](const ModelTarget& model) {
                                    return !model.enabled;
                                  }),
                   models.end());
    }
    if (models.empty()) {
      send_error(res, 400, "No models selected");
      return;
    }

    run_config.models = models;
    run_config.name = body.value("name", config_.name);
    if (run_config.name.empty()) {
      run_config.name = "LLM evaluation";
    }
    run_config.concurrency = std::max(
        1, std::min(12, body.value("concurrency", config_.concurrency)));
    int max_items = body.value("max_items", 0);
    max_items = std::max(0, std::min(1000, max_items));
    run_config.raw = sanitize_config(run_config);

    std::string error;
    if (!jobs_.start(run_config, dataset_, max_items, "live", error)) {
      send_error(res, 409, error);
      return;
    }
    send_json(res, json{{"ok", true}, {"job", jobs_.snapshot_json()}});
  });

  server_->Post("/api/demo", [this](const httplib::Request&, httplib::Response& res) {
    EvalConfig demo = demo_config();
    demo.dataset_path = config_.dataset_path;
    demo.name = "Demo evaluation";
    std::string error;
    if (!jobs_.start(demo, dataset_, 0, "demo", error)) {
      send_error(res, 409, error);
      return;
    }
    send_json(res, json{{"ok", true}, {"job", jobs_.snapshot_json()}});
  });

  server_->set_logger([](const httplib::Request& req, const httplib::Response& res) {
    if (res.status >= 400 && req.path.rfind("/api/job", 0) != 0) {
      std::cerr << "[http] " << res.status << " " << req.method << " " << req.path
                << "\n";
    }
  });

  server_->set_exception_handler(
      [](const httplib::Request&, httplib::Response& res, std::exception_ptr error) {
        std::string message = "Internal server error";
        if (error) {
          try {
            std::rethrow_exception(error);
          } catch (const std::exception& exception) {
            message = exception.what();
          } catch (...) {
            message = "Unknown error";
          }
        }
        send_error(res, 500, message);
      });

  server_->set_error_handler([](const httplib::Request& req, httplib::Response& res) {
    if (req.path.rfind("/api/", 0) == 0) {
      send_error(res, res.status, "Not found: " + req.path);
    } else {
      res.set_content("<!doctype html><html><body><h1>404</h1>"
                      "<p>Not found. Open <a href=\"/\">the dashboard</a>.</p>"
                      "</body></html>",
                      "text/html; charset=utf-8");
    }
  });
}

bool App::run() {
  std::string display_host = options_.host == "0.0.0.0" ? "127.0.0.1" : options_.host;
  std::string url = "http://" + display_host + ":" + std::to_string(options_.port);
  std::cout << "\n  LLM Eval Dashboard\n";
  std::cout << "  " << url << "\n";
  std::cout << "  web assets: "
            << (web_dir_.empty() ? "embedded" : web_dir_) << "\n";
  std::cout << "  results:    " << storage_.root() << "\n\n";
  std::cout << "  Press Ctrl+C to stop.\n" << std::flush;

  started_ = Clock::now();
  if (options_.open_browser) {
    open_in_browser(url);
  }
  bool ok = server_->listen(options_.host, options_.port);
  if (!ok) {
    std::cerr << "\n[error] Could not listen on " << options_.host << ":"
              << options_.port << " (is the port already in use?)\n";
  }
  return ok;
}

void generate_demo_runs(Storage& storage, const Dataset& dataset, int count) {
  if (count <= 0 || dataset.items.empty()) {
    return;
  }
  EvalConfig config = demo_config();
  const auto now = std::chrono::system_clock::now();
  for (int i = 0; i < count; ++i) {
    RunOptions options;
    options.max_items = 0;
    options.seed = 1000 + static_cast<std::uint64_t>(i) * 7919;
    RunRecord run = run_evaluation(config, dataset, config.models, options, nullptr);
    run.source = "demo";
    if (i == count - 1) {
      run.name = "Demo current";
    } else if (i == 0) {
      run.name = "Demo baseline";
    } else {
      run.name = "Demo iteration " + std::to_string(i + 1);
    }
    run.created_at = iso_from_time(now - std::chrono::hours(24 * 3 * (count - 1 - i)));
    run.id = "run_" + file_stamp_now() + "_demo_" + std::to_string(i + 1);
    run.config["name"] = run.name;
    storage.save_run(run);
  }
}

void open_in_browser(const std::string& url) {
#ifdef _WIN32
  std::wstring wide(url.begin(), url.end());
  ShellExecuteW(nullptr, L"open", wide.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
  std::system(("open '" + url + "' >/dev/null 2>&1 &").c_str());
#else
  std::system(("xdg-open '" + url + "' >/dev/null 2>&1 &").c_str());
#endif
}

} // namespace llmbench
