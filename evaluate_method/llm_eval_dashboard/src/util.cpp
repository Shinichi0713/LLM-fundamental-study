#include "util.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace llmbench {

namespace {

constexpr char kHex[] = "0123456789abcdef";

bool is_ascii_space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

std::string local_stamp(const char* format) {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char buffer[64];
  std::strftime(buffer, sizeof(buffer), format, &tm);
  return buffer;
}

// Iterates UTF-8 code points and returns the byte offset of each one.
std::vector<std::size_t> codepoint_offsets(const std::string& value) {
  std::vector<std::size_t> offsets;
  std::size_t i = 0;
  while (i < value.size()) {
    offsets.push_back(i);
    unsigned char c = static_cast<unsigned char>(value[i]);
    std::size_t length = 1;
    if ((c & 0x80) == 0) {
      length = 1;
    } else if ((c & 0xE0) == 0xC0) {
      length = 2;
    } else if ((c & 0xF0) == 0xE0) {
      length = 3;
    } else if ((c & 0xF8) == 0xF0) {
      length = 4;
    }
    i += std::min(length, value.size() - i);
  }
  return offsets;
}

} // namespace

std::string iso_now() {
  return iso_from_time(std::chrono::system_clock::now());
}

std::string iso_from_time(std::chrono::system_clock::time_point tp) {
  std::time_t t = std::chrono::system_clock::to_time_t(tp);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buffer[64];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buffer;
}

std::string file_stamp_now() {
  return local_stamp("%Y%m%d_%H%M%S");
}

std::string to_lower_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string trim(std::string value) {
  std::size_t begin = 0;
  while (begin < value.size() && is_ascii_space(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin && is_ascii_space(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

bool starts_with(const std::string& value, const std::string& prefix) {
  return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> split(const std::string& value, char delimiter) {
  std::vector<std::string> parts;
  std::string current;
  std::istringstream stream(value);
  while (std::getline(stream, current, delimiter)) {
    parts.push_back(current);
  }
  if (!value.empty() && value.back() == delimiter) {
    parts.emplace_back();
  }
  return parts;
}

std::string replace_all(std::string value, const std::string& from, const std::string& to) {
  if (from.empty()) {
    return value;
  }
  std::size_t position = 0;
  while ((position = value.find(from, position)) != std::string::npos) {
    value.replace(position, from.size(), to);
    position += to.size();
  }
  return value;
}

std::string slugify(const std::string& value) {
  std::string result;
  bool last_dash = false;
  for (unsigned char c : value) {
    if (std::isalnum(c)) {
      result.push_back(static_cast<char>(std::tolower(c)));
      last_dash = false;
    } else if (!last_dash && !result.empty()) {
      result.push_back('-');
      last_dash = true;
    }
  }
  while (!result.empty() && result.back() == '-') {
    result.pop_back();
  }
  if (result.empty()) {
    result = "run";
  }
  return result;
}

std::string sanitize_id(const std::string& value) {
  std::string result;
  for (unsigned char c : value) {
    if (std::isalnum(c) || c == '-' || c == '_') {
      result.push_back(static_cast<char>(c));
    }
  }
  return result;
}

std::string escape_csv(const std::string& value) {
  std::string result;
  result.reserve(value.size() + 8);
  for (char c : value) {
    if (c == '"') {
      result += "\"\"";
    } else if (c == '\r') {
      // normalised below
    } else {
      result.push_back(c);
    }
  }
  return result;
}

std::string truncate_utf8(const std::string& value, std::size_t max_chars) {
  auto offsets = codepoint_offsets(value);
  if (offsets.size() <= max_chars) {
    return value;
  }
  std::size_t end = max_chars < offsets.size() ? offsets[max_chars] : value.size();
  return value.substr(0, end);
}

std::string format_double(double value, int precision) {
  if (!std::isfinite(value)) {
    return "0";
  }
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
  return buffer;
}

std::string format_duration(double milliseconds) {
  if (milliseconds < 1000.0) {
    return format_double(milliseconds, 0) + " ms";
  }
  if (milliseconds < 60000.0) {
    return format_double(milliseconds / 1000.0, 2) + " s";
  }
  return format_double(milliseconds / 60000.0, 2) + " min";
}

std::string normalize_text(const std::string& value) {
  std::string result;
  result.reserve(value.size());
  for (unsigned char c : value) {
    if (c < 0x80) {
      if (std::isalnum(c)) {
        result.push_back(static_cast<char>(std::tolower(c)));
      }
      continue;
    }
    // Keep multi-byte UTF-8 sequences; drop only ASCII whitespace/punctuation.
    result.push_back(static_cast<char>(c));
  }
  return result;
}

std::vector<std::string> utf8_chars(const std::string& value) {
  std::vector<std::string> chars;
  auto offsets = codepoint_offsets(value);
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    std::size_t begin = offsets[i];
    std::size_t end = (i + 1 < offsets.size()) ? offsets[i + 1] : value.size();
    chars.push_back(value.substr(begin, end - begin));
  }
  return chars;
}

double char_bigram_f1(const std::string& candidate, const std::string& reference) {
  std::string a = normalize_text(candidate);
  std::string b = normalize_text(reference);
  if (a.empty() || b.empty()) {
    return 0.0;
  }
  auto a_chars = utf8_chars(a);
  auto b_chars = utf8_chars(b);
  if (a_chars.size() < 2 || b_chars.size() < 2) {
    return a == b ? 1.0 : 0.0;
  }
  std::unordered_map<std::string, int> counts;
  for (std::size_t i = 0; i + 1 < b_chars.size(); ++i) {
    counts[b_chars[i] + b_chars[i + 1]] += 1;
  }
  int overlap = 0;
  for (std::size_t i = 0; i + 1 < a_chars.size(); ++i) {
    auto it = counts.find(a_chars[i] + a_chars[i + 1]);
    if (it != counts.end() && it->second > 0) {
      --it->second;
      ++overlap;
    }
  }
  double precision = static_cast<double>(overlap) / static_cast<double>(a_chars.size() - 1);
  double recall = static_cast<double>(overlap) / static_cast<double>(b_chars.size() - 1);
  if (precision + recall <= 0.0) {
    return 0.0;
  }
  return 2.0 * precision * recall / (precision + recall);
}

int estimate_tokens(const std::string& text) {
  if (text.empty()) {
    return 0;
  }
  std::size_t ascii = 0;
  std::size_t multibyte = 0;
  for (unsigned char c : text) {
    if (c < 0x80) {
      ++ascii;
    } else if ((c & 0xC0) != 0x80) {
      ++multibyte;
    }
  }
  // Rough but stable heuristic: ~4 ASCII characters or ~1.5 CJK characters
  // per token.
  double tokens = static_cast<double>(ascii) / 4.0 +
                  static_cast<double>(multibyte) / 1.5;
  return std::max(1, static_cast<int>(std::lround(tokens)));
}

double clamp01(double value) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  return std::max(0.0, std::min(1.0, value));
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  p = std::max(0.0, std::min(1.0, p));
  double position = p * static_cast<double>(values.size() - 1);
  std::size_t lower = static_cast<std::size_t>(std::floor(position));
  std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  if (lower == upper) {
    return values[lower];
  }
  double weight = position - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

uint64_t fnv1a64(const std::string& value) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char c : value) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ULL;
  }
  return hash;
}

uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

std::string read_file(const std::string& path, bool& ok) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    ok = false;
    return {};
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  ok = true;
  return buffer.str();
}

bool write_file(const std::string& path, const std::string& content) {
  fs::path parent = fs::path(path).parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    fs::create_directories(parent, ec);
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return false;
  }
  output.write(content.data(), static_cast<std::streamsize>(content.size()));
  return static_cast<bool>(output);
}

bool file_exists(const std::string& path) {
  std::error_code ec;
  return fs::is_regular_file(fs::path(path), ec);
}

bool directory_exists(const std::string& path) {
  std::error_code ec;
  return fs::is_directory(fs::path(path), ec);
}

bool ensure_directory(const std::string& path) {
  std::error_code ec;
  fs::create_directories(fs::path(path), ec);
  return !ec;
}

std::string executable_dir() {
#ifdef _WIN32
  std::wstring buffer(MAX_PATH, L'\0');
  for (;;) {
    DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
                                      static_cast<DWORD>(buffer.size()));
    if (length == 0) {
      return fs::current_path().string();
    }
    if (length < buffer.size()) {
      buffer.resize(length);
      break;
    }
    buffer.resize(buffer.size() * 2);
  }
  return fs::path(buffer).parent_path().string();
#else
  std::array<char, 4096> buffer{};
  ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length <= 0) {
    return fs::current_path().string();
  }
  buffer[static_cast<std::size_t>(length)] = '\0';
  return fs::path(buffer.data()).parent_path().string();
#endif
}

std::string absolute_path(const std::string& path) {
  std::error_code ec;
  fs::path result = fs::absolute(fs::path(path), ec);
  if (ec) {
    return path;
  }
  return result.lexically_normal().string();
}

std::string resolve_existing_path(const std::string& path) {
  std::error_code ec;
  fs::path given(path);
  if (given.is_absolute()) {
    return fs::exists(given, ec) ? given.string() : std::string();
  }
  fs::path cwd_candidate = fs::current_path(ec) / given;
  if (!ec && fs::exists(cwd_candidate, ec)) {
    return cwd_candidate.lexically_normal().string();
  }
  fs::path exe_candidate = fs::path(executable_dir()) / given;
  if (fs::exists(exe_candidate, ec)) {
    return exe_candidate.lexically_normal().string();
  }
  return {};
}

} // namespace llmbench
