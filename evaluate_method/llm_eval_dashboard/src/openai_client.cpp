#include "openai_client.h"

#include <chrono>
#include <cstdlib>
#include <string>

#include "http_client.h"
#include "util.h"

namespace llmbench {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string join_url(std::string base, const std::string& suffix) {
  while (!base.empty() && base.back() == '/') {
    base.pop_back();
  }
  return base + suffix;
}

void read_usage(const json& value, int& prompt_tokens, int& completion_tokens) {
  auto it = value.find("usage");
  if (it == value.end() || !it->is_object()) {
    return;
  }
  prompt_tokens = it->value("prompt_tokens", prompt_tokens);
  completion_tokens = it->value("completion_tokens", completion_tokens);
}

std::string extract_error_message(const std::string& body) {
  try {
    json value = json::parse(body);
    if (value.contains("error")) {
      if (value["error"].is_string()) {
        return value["error"].get<std::string>();
      }
      if (value["error"].is_object()) {
        return value["error"].value("message", value["error"].dump());
      }
    }
    if (value.contains("message") && value["message"].is_string()) {
      return value["message"].get<std::string>();
    }
  } catch (...) {
    // Fall through to the raw body.
  }
  std::string text = trim(body);
  if (text.size() > 400) {
    text = truncate_utf8(text, 400) + "...";
  }
  return text;
}

struct StreamState {
  std::string pending;
  std::string body;
  std::string content;
  std::string error;
  bool saw_sse = false;
  bool done = false;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  double ttft_ms = 0.0;
  Clock::time_point start;
};

void handle_json_event(const std::string& payload, StreamState& state) {
  json value;
  try {
    value = json::parse(payload);
  } catch (...) {
    return;
  }
  if (value.contains("error")) {
    state.error = extract_error_message(payload);
    return;
  }
  read_usage(value, state.prompt_tokens, state.completion_tokens);

  if (!value.contains("choices") || !value["choices"].is_array() ||
      value["choices"].empty()) {
    return;
  }
  const auto& choice = value["choices"][0];
  std::string delta;
  if (choice.contains("delta") && choice["delta"].is_object()) {
    const auto& part = choice["delta"];
    if (part.contains("content") && part["content"].is_string()) {
      delta = part["content"].get<std::string>();
    }
  }
  if (delta.empty() && choice.contains("message") && choice["message"].is_object()) {
    const auto& message = choice["message"];
    if (message.contains("content") && message["content"].is_string()) {
      delta = message["content"].get<std::string>();
    }
  }
  if (delta.empty() && choice.contains("text") && choice["text"].is_string()) {
    delta = choice["text"].get<std::string>();
  }
  if (!delta.empty()) {
    if (state.ttft_ms == 0.0) {
      state.ttft_ms = elapsed_ms(state.start);
    }
    state.content += delta;
  }
}

void process_lines(StreamState& state) {
  for (;;) {
    auto newline = state.pending.find('\n');
    if (newline == std::string::npos) {
      return;
    }
    std::string line = state.pending.substr(0, newline);
    state.pending.erase(0, newline + 1);
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (starts_with(line, "data:")) {
      state.saw_sse = true;
      std::string payload = trim(line.substr(5));
      if (payload == "[DONE]") {
        state.done = true;
        continue;
      }
      if (!payload.empty()) {
        handle_json_event(payload, state);
      }
    }
  }
}

void flush_pending(StreamState& state) {
  if (state.pending.empty()) {
    return;
  }
  // A provider may end the stream without a trailing newline.
  state.pending.push_back('\n');
  process_lines(state);
}

void parse_non_stream_body(StreamState& state) {
  if (state.body.empty()) {
    return;
  }
  try {
    json value = json::parse(state.body);
    if (value.contains("error")) {
      state.error = extract_error_message(state.body);
      return;
    }
    read_usage(value, state.prompt_tokens, state.completion_tokens);
    if (value.contains("choices") && value["choices"].is_array() &&
        !value["choices"].empty()) {
      const auto& choice = value["choices"][0];
      if (choice.contains("message") && choice["message"].is_object() &&
          choice["message"].contains("content") &&
          choice["message"]["content"].is_string()) {
        state.content = choice["message"]["content"].get<std::string>();
        state.ttft_ms = elapsed_ms(state.start);
      } else if (choice.contains("text") && choice["text"].is_string()) {
        state.content = choice["text"].get<std::string>();
        state.ttft_ms = elapsed_ms(state.start);
      }
    }
  } catch (...) {
    // Not JSON; the caller reports the HTTP/parse error.
  }
}

struct AttemptResult {
  HttpResponse response;
  StreamState state;
};

AttemptResult attempt(const ModelTarget& model,
                      const BenchmarkItem& item,
                      const EvalConfig& config,
                      bool include_usage,
                      std::atomic<bool>* cancel) {
  AttemptResult result;
  result.state.start = Clock::now();

  json messages = json::array();
  if (!item.system.empty()) {
    messages.push_back({{"role", "system"}, {"content", item.system}});
  }
  messages.push_back({{"role", "user"}, {"content", item.prompt}});

  json body{
      {"model", model.model},
      {"messages", messages},
      {"temperature", config.temperature},
      {"max_tokens", item.max_tokens > 0 ? item.max_tokens : config.max_tokens},
      {"stream", true},
  };
  if (include_usage) {
    body["stream_options"] = {{"include_usage", true}};
  }

  HttpRequest request;
  request.url = join_url(model.base_url, "/chat/completions");
  request.timeout_ms = config.timeout_ms;
  request.body = body.dump();
  request.headers = {
      {"Content-Type", "application/json"},
      {"Accept", "text/event-stream"},
      {"User-Agent", "llm-eval-dashboard/1.0"},
  };

  if (!model.api_key_env.empty()) {
    const char* key = std::getenv(model.api_key_env.c_str());
    if (key == nullptr || *key == '\0') {
      result.response.error = "Environment variable " + model.api_key_env +
                              " is not set";
      return result;
    }
    request.headers.emplace_back("Authorization", std::string("Bearer ") + key);
  }

  result.response = http_post_stream(
      request,
      [&](const char* data, std::size_t length) {
        if (result.state.body.size() < 4 * 1024 * 1024) {
          result.state.body.append(data, length);
        }
        result.state.pending.append(data, length);
        process_lines(result.state);
        return true;
      },
      cancel);
  if (result.state.saw_sse) {
    flush_pending(result.state);
  }
  return result;
}

} // namespace

CallOutcome call_openai_compatible(const ModelTarget& model,
                                   const BenchmarkItem& item,
                                   const EvalConfig& config,
                                   std::atomic<bool>* cancel) {
  CallOutcome outcome;
  auto start = Clock::now();

  AttemptResult result = attempt(model, item, config, true, cancel);
  if (!result.response.error.empty() && result.response.status == 0) {
    outcome.aborted = result.response.aborted;
    outcome.error = result.response.error;
    outcome.latency_ms = elapsed_ms(start);
    return outcome;
  }

  // Some providers reject stream_options; retry once without it.
  if ((result.response.status == 400 || result.response.status == 422) &&
      result.state.content.empty()) {
    AttemptResult retry = attempt(model, item, config, false, cancel);
    if (retry.response.status >= 200 && retry.response.status < 300) {
      result = std::move(retry);
    } else if (!retry.response.error.empty()) {
      result.response.error = retry.response.error;
      result.response.status = retry.response.status;
    }
  }

  outcome.status = result.response.status;
  outcome.latency_ms = elapsed_ms(start);
  outcome.aborted = result.response.aborted;

  if (result.response.status < 200 || result.response.status >= 300) {
    if (outcome.aborted) {
      outcome.error = "cancelled";
    } else if (!result.response.error.empty()) {
      outcome.error = result.response.error;
    } else {
      outcome.error = "HTTP " + std::to_string(result.response.status) + ": " +
                      extract_error_message(result.state.body);
    }
    return outcome;
  }

  if (result.state.content.empty()) {
    // The endpoint may have ignored "stream" and returned a normal JSON body.
    parse_non_stream_body(result.state);
  }
  if (!result.state.error.empty()) {
    outcome.error = result.state.error;
    return outcome;
  }
  if (result.state.content.empty()) {
    outcome.error = "Empty response from model";
    return outcome;
  }

  outcome.ok = true;
  outcome.output = result.state.content;
  outcome.ttft_ms = result.state.ttft_ms > 0.0 ? result.state.ttft_ms : outcome.latency_ms;
  outcome.prompt_tokens = result.state.prompt_tokens;
  outcome.completion_tokens = result.state.completion_tokens;
  if (outcome.prompt_tokens <= 0) {
    outcome.prompt_tokens = estimate_tokens(item.system + item.prompt) + 12;
  }
  if (outcome.completion_tokens <= 0) {
    outcome.completion_tokens = estimate_tokens(outcome.output);
  }
  return outcome;
}

} // namespace llmbench
