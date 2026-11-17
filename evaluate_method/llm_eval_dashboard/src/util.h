#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace llmbench {

// ---- time ---------------------------------------------------------------
std::string iso_now();
std::string iso_from_time(std::chrono::system_clock::time_point tp);
std::string file_stamp_now();

// ---- strings ------------------------------------------------------------
std::string to_lower_ascii(std::string value);
std::string trim(std::string value);
bool starts_with(const std::string& value, const std::string& prefix);
bool ends_with(const std::string& value, const std::string& suffix);
std::vector<std::string> split(const std::string& value, char delimiter);
std::string replace_all(std::string value, const std::string& from, const std::string& to);
std::string slugify(const std::string& value);
std::string sanitize_id(const std::string& value);
std::string escape_csv(const std::string& value);
std::string truncate_utf8(const std::string& value, std::size_t max_chars);
std::string format_double(double value, int precision = 2);
std::string format_duration(double milliseconds);

// ---- text analysis ------------------------------------------------------
// Removes whitespace and common punctuation, lowercases ASCII characters.
std::string normalize_text(const std::string& value);
std::vector<std::string> utf8_chars(const std::string& value);
double char_bigram_f1(const std::string& candidate, const std::string& reference);
int estimate_tokens(const std::string& text);

// ---- numbers ------------------------------------------------------------
double clamp01(double value);
double percentile(std::vector<double> values, double p);
uint64_t fnv1a64(const std::string& value);
uint64_t mix64(uint64_t x);

// ---- files --------------------------------------------------------------
std::string read_file(const std::string& path, bool& ok);
bool write_file(const std::string& path, const std::string& content);
bool file_exists(const std::string& path);
bool directory_exists(const std::string& path);
bool ensure_directory(const std::string& path);
std::string executable_dir();
std::string absolute_path(const std::string& path);
// Resolves a relative path against the current directory first, then the
// executable directory. Returns an empty string when neither exists.
std::string resolve_existing_path(const std::string& path);

} // namespace llmbench
