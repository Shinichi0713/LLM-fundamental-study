#pragma once

#include <string>
#include <vector>

#include "types.h"

namespace llmbench {

class Storage {
 public:
  explicit Storage(std::string root);

  const std::string& root() const { return root_; }

  // Persists a run and returns the file path. The run id must already be set.
  std::string save_run(const RunRecord& run);

  // Lightweight metadata for the dashboard, newest first.
  std::vector<json> list_runs() const;

  bool run_exists(const std::string& id) const;

  // Throws std::runtime_error when the run cannot be read.
  json load_run_json(const std::string& id) const;
  RunRecord load_run(const std::string& id) const;

  std::string latest_run_id() const;

  // Returns a UTF-8 CSV document (with BOM for Excel).
  std::string export_csv(const std::string& id) const;

 private:
  std::string path_for(const std::string& id) const;

  std::string root_;
};

} // namespace llmbench
