#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <numeric>

// ------------------------------------------------------------------
// LLM Quality Evaluator in C++17
// Metrics: Exact Match, BLEU (simplified), ROUGE-L, Token F1, Levenshtein
// ------------------------------------------------------------------

namespace eval {

// -------------------- Utility --------------------

static inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string tok;
    while (iss >> tok) {
        // simple lowercase normalization
        std::transform(tok.begin(), tok.end(), tok.begin(), ::tolower);
        tokens.push_back(tok);
    }
    return tokens;
}

static inline std::vector<std::string> get_ngrams(const std::vector<std::string>& tokens, size_t n) {
    std::vector<std::string> ngrams;
    if (tokens.size() < n) return ngrams;
    for (size_t i = 0; i <= tokens.size() - n; ++i) {
        std::string gram = tokens[i];
        for (size_t j = 1; j < n; ++j) {
            gram += " " + tokens[i + j];
        }
        ngrams.push_back(gram);
    }
    return ngrams;
}

// -------------------- Exact Match --------------------

bool exact_match(const std::string& reference, const std::string& hypothesis) {
    return reference == hypothesis;
}

// -------------------- Levenshtein Distance --------------------

size_t levenshtein(const std::string& s1, const std::string& s2) {
    const size_t len1 = s1.size(), len2 = s2.size();
    std::vector<size_t> prev(len2 + 1), curr(len2 + 1);
    for (size_t j = 0; j <= len2; ++j) prev[j] = j;
    for (size_t i = 1; i <= len1; ++i) {
        curr[0] = i;
        for (size_t j = 1; j <= len2; ++j) {
            size_t cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
        }
        std::swap(prev, curr);
    }
    return prev[len2];
}

// -------------------- Token F1 --------------------

double token_f1(const std::string& reference, const std::string& hypothesis) {
    auto ref_tokens = tokenize(reference);
    auto hyp_tokens = tokenize(hypothesis);

    std::unordered_map<std::string, size_t> ref_counts, hyp_counts;
    for (const auto& t : ref_tokens) ref_counts[t]++;
    for (const auto& t : hyp_tokens) hyp_counts[t]++;

    size_t intersection = 0;
    for (const auto& kv : hyp_counts) {
        auto it = ref_counts.find(kv.first);
        if (it != ref_counts.end()) {
            intersection += std::min(kv.second, it->second);
        }
    }

    if (ref_tokens.empty() && hyp_tokens.empty()) return 1.0;
    if (ref_tokens.empty() || hyp_tokens.empty()) return 0.0;

    double precision = static_cast<double>(intersection) / hyp_tokens.size();
    double recall    = static_cast<double>(intersection) / ref_tokens.size();
    if (precision + recall == 0.0) return 0.0;
    return 2.0 * precision * recall / (precision + recall);
}

// -------------------- BLEU (simplified, up to 4-gram) --------------------

double bleu(const std::string& reference, const std::string& hypothesis, size_t max_n = 4) {
    auto ref_tokens = tokenize(reference);
    auto hyp_tokens = tokenize(hypothesis);

    if (hyp_tokens.empty()) return 0.0;
    if (ref_tokens.empty()) return (hyp_tokens.empty() ? 1.0 : 0.0);

    double bp = (hyp_tokens.size() > ref_tokens.size())
                    ? 1.0
                    : std::exp(1.0 - static_cast<double>(ref_tokens.size()) / hyp_tokens.size());

    double log_sum = 0.0;
    size_t valid_n = 0;

    for (size_t n = 1; n <= max_n; ++n) {
        auto ref_ngrams = get_ngrams(ref_tokens, n);
        auto hyp_ngrams = get_ngrams(hyp_tokens, n);

        if (hyp_ngrams.empty()) continue;

        std::unordered_map<std::string, size_t> ref_counts;
        for (const auto& g : ref_ngrams) ref_counts[g]++;

        size_t clip_count = 0;
        std::unordered_map<std::string, size_t> hyp_counts;
        for (const auto& g : hyp_ngrams) hyp_counts[g]++;

        for (const auto& kv : hyp_counts) {
            auto it = ref_counts.find(kv.first);
            if (it != ref_counts.end()) {
                clip_count += std::min(kv.second, it->second);
            }
        }

        if (!hyp_ngrams.empty() && clip_count > 0) {
            log_sum += std::log(static_cast<double>(clip_count) / hyp_ngrams.size());
            valid_n++;
        }
    }

    if (valid_n == 0) return 0.0;
    double avg_log = log_sum / valid_n;
    return bp * std::exp(avg_log);
}

// -------------------- ROUGE-L (LCS-based F-measure) --------------------

double rouge_l(const std::string& reference, const std::string& hypothesis) {
    auto ref = tokenize(reference);
    auto hyp = tokenize(hypothesis);

    size_t m = ref.size(), n = hyp.size();
    if (m == 0 && n == 0) return 1.0;
    if (m == 0 || n == 0) return 0.0;

    std::vector<std::vector<size_t>> dp(m + 1, std::vector<size_t>(n + 1, 0));
    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            if (ref[i - 1] == hyp[j - 1])
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else
                dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
        }
    }

    double lcs_len = static_cast<double>(dp[m][n]);
    double precision = lcs_len / n;
    double recall    = lcs_len / m;
    if (precision + recall == 0.0) return 0.0;
    return 2.0 * precision * recall / (precision + recall);
}

// -------------------- Result Structure --------------------

struct Metrics {
    double exact_match   = 0.0; // 0 or 1
    double bleu          = 0.0; // 0 ~ 1
    double rouge_l       = 0.0; // 0 ~ 1
    double token_f1      = 0.0; // 0 ~ 1
    size_t levenshtein   = 0;   // character distance
};

Metrics evaluate(const std::string& reference, const std::string& hypothesis) {
    Metrics m;
    m.exact_match = exact_match(reference, hypothesis) ? 1.0 : 0.0;
    m.bleu        = bleu(reference, hypothesis);
    m.rouge_l     = rouge_l(reference, hypothesis);
    m.token_f1    = token_f1(reference, hypothesis);
    m.levenshtein = levenshtein(reference, hypothesis);
    return m;
}

void print_metrics(const Metrics& m) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Exact Match : " << m.exact_match << "\n";
    std::cout << "BLEU        : " << m.bleu << "\n";
    std::cout << "ROUGE-L     : " << m.rouge_l << "\n";
    std::cout << "Token F1    : " << m.token_f1 << "\n";
    std::cout << "Levenshtein : " << m.levenshtein << "\n";
}

} // namespace eval

// ------------------------------------------------------------------
// Example: Evaluate multiple LLM outputs against reference answers
// ------------------------------------------------------------------

int main() {
    // reference (ground truth) -> hypothesis (LLM output)
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {
            "The capital of Japan is Tokyo.",
            "The capital of Japan is Tokyo."
        },
        {
            "The quick brown fox jumps over the lazy dog.",
            "A quick brown fox jumps over a lazy dog."
        },
        {
            "Machine learning is a subset of artificial intelligence.",
            "Machine learning is part of AI."
        },
        {
            "Rust is a systems programming language that runs blazingly fast.",
            "Python is a scripting language."
        }
    };

    std::cout << "===== LLM Quality Evaluation =====\n\n";

    for (size_t i = 0; i < test_cases.size(); ++i) {
        const auto& ref = test_cases[i].first;
        const auto& hyp = test_cases[i].second;

        std::cout << "[Test Case " << (i + 1) << "]\n";
        std::cout << "Reference : " << ref << "\n";
        std::cout << "Hypothesis: " << hyp << "\n";
        std::cout << "---\n";

        auto m = eval::evaluate(ref, hyp);
        eval::print_metrics(m);
        std::cout << "\n";
    }

    // Aggregate average scores across dataset
    std::cout << "===== Average Metrics =====\n";
    double sum_em = 0, sum_bleu = 0, sum_rl = 0, sum_f1 = 0;
    for (const auto& tc : test_cases) {
        auto m = eval::evaluate(tc.first, tc.second);
        sum_em   += m.exact_match;
        sum_bleu += m.bleu;
        sum_rl   += m.rouge_l;
        sum_f1   += m.token_f1;
    }
    size_t n = test_cases.size();
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Avg Exact Match: " << (sum_em / n) << "\n";
    std::cout << "Avg BLEU       : " << (sum_bleu / n) << "\n";
    std::cout << "Avg ROUGE-L    : " << (sum_rl / n) << "\n";
    std::cout << "Avg Token F1   : " << (sum_f1 / n) << "\n";

    return 0;
}
