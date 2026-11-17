#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "benchmark.h"
#include "storage.h"
#include "types.h"

namespace httplib {
class Server;
}

namespace llmbench {

struct AppOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string web_dir;
  bool open_browser = false;
};

class JobManager {
 public:
  JobManager(Storage& storage, Dataset dataset, EvalConfig config);
  ~JobManager();

  JobManager(const JobManager&) = delete;
  JobManager& operator=(const JobManager&) = delete;

  bool start(const EvalConfig& config,
             const Dataset& dataset,
             int max_items,
             const std::string& source,
             std::string& error);

  bool cancel();
  bool running() const;
  JobSnapshot snapshot() const;
  json snapshot_json() const;
  void wait();

 private:
  struct Job;

  Storage& storage_;
  Dataset default_dataset_;
  EvalConfig default_config_;

  mutable std::mutex mutex_;
  std::shared_ptr<Job> job_;
};

class App {
 public:
  App(AppOptions options, EvalConfig config, Dataset dataset, Storage& storage);
  ~App();

  // Blocks until the HTTP server stops.
  bool run();

 private:
  void setup_routes();
  std::string asset(const std::string& name) const;

  AppOptions options_;
  EvalConfig config_;
  Dataset dataset_;
  Storage& storage_;
  JobManager jobs_;
  std::unique_ptr<httplib::Server> server_;
  std::string web_dir_;
  std::chrono::steady_clock::time_point started_;
};

// Creates a few backdated demo runs so the dashboard has history on first use.
void generate_demo_runs(Storage& storage, const Dataset& dataset, int count = 3);

void open_in_browser(const std::string& url);

} // namespace llmbench
