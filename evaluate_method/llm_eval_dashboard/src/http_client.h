#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace llmbench {

struct HttpRequest {
  std::string url;
  std::vector<std::pair<std::string, std::string>> headers;
  std::string body;
  int timeout_ms = 60000;
};

struct HttpResponse {
  int status = 0;
  std::string body;
  std::string error;
  bool aborted = false;
};

using StreamCallback = std::function<bool(const char* data, std::size_t length)>;

// POSTs a request and forwards the response body in chunks. Returns after the
// whole body has been received. Implemented with WinHTTP on Windows (TLS
// included) and cpp-httplib elsewhere.
HttpResponse http_post_stream(const HttpRequest& request,
                              const StreamCallback& on_chunk,
                              std::atomic<bool>* cancel = nullptr);

} // namespace llmbench
