/*
============================================================
Simple AI Agent in C++17 (Standard Library Only)
============================================================
ReAct (Reasoning + Acting) パターンを用いた最小構成のAIエージェント。
外部ライブラリ不要。C++17以上でコンパイルしてください。

コンパイル例:
    g++ -std=c++17 -O2 -o simple_ai_agent simple_ai_agent.cpp

使い方:
    ./simple_ai_agent
*/

#include <algorithm>
#include <cctype>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
// 簡易JSONパーサー（ReActのthought/action/action_inputを抽出）
// ============================================================
struct ReActResult {
    std::string thought;
    std::string action;
    std::string action_input;
};

ReActResult parseReActJson(const std::string& json_str) {
    ReActResult result;
    // "thought": "..." を抽出
    std::regex thought_re("\"thought\"\\s*:\\s*\"([^\"]*)\"");
    std::regex action_re("\"action\"\\s*:\\s*\"([^\"]*)\"");
    std::regex input_re("\"action_input\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(json_str, match, thought_re)) result.thought = match[1];
    if (std::regex_search(json_str, match, action_re)) result.action = match[1];
    if (std::regex_search(json_str, match, input_re)) result.action_input = match[1];
    return result;
}

// ============================================================
// 文字列ユーティリティ
// ============================================================
std::string toLower(const std::string& s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return r;
}

std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

// ============================================================
// ツール実装
// ============================================================
std::string tool_calculator(const std::string& expression) {
    std::string expr = trim(expression);
    if (expr.empty()) return "エラー: 式が空です";
    // 許可文字のチェック
    for (char c : expr) {
        bool ok = std::isdigit(static_cast<unsigned char>(c)) ||
                  c == '+' || c == '-' || c == '*' || c == '/' ||
                  c == '.' || c == '(' || c == ')' || c == ' ';
        if (!ok) return "エラー: 使用可能な文字は 0-9 + - * / . ( ) スペース のみです";
    }

    // 簡易式評価（+ と - のみ対応、スペース区切で数字と演算子）
    std::istringstream iss(expr);
    double result = 0.0;
    double num;
    char op = '+';
    bool first = true;

    while (iss >> num || (iss >> op && iss >> num)) {
        if (first) {
            result = num;
            first = false;
        } else {
            if (op == '+') result += num;
            else if (op == '-') result -= num;
            else if (op == '*') result *= num;
            else if (op == '/') {
                if (num == 0.0) return "エラー: ゼロ除算";
                result /= num;
            }
        }
        // 次の演算子を読む試行
        while (iss.peek() == ' ') iss.get();
        if (iss.peek() == '+' || iss.peek() == '-' || iss.peek() == '*' || iss.peek() == '/') {
            op = static_cast<char>(iss.get());
        } else {
            break;
        }
        while (iss.peek() == ' ') iss.get();
    }

    // 整数なら整数表示
    if (result == static_cast<long long>(result)) {
        return std::to_string(static_cast<long long>(result));
    }
    return std::to_string(result);
}

std::string tool_read_file(const std::string& filepath) {
    std::string path = trim(filepath);
    if (path.empty()) return "エラー: ファイルパスが空です";
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "エラー: ファイル '" + path + "' が見つかりません";
    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
    if (content.size() > 500) {
        content = content.substr(0, 500) + "\n... (以下省略)";
    }
    return content.empty() ? "(ファイルは空です)" : content;
}

std::string tool_get_time(const std::string& /*unused*/) {
    std::time_t now = std::time(nullptr);
    std::tm* local = std::localtime(&now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", local);
    return std::string(buf);
}

// ============================================================
// エージェントクラス
// ============================================================
using ToolFunc = std::function<std::string(const std::string&)>;

class SimpleAgent {
public:
    explicit SimpleAgent(int max_steps = 10) : max_steps_(max_steps) {}

    std::string think(const std::string& goal) {
        std::string goal_lower = toLower(goal);
        std::string memory_text;
        for (const auto& m : memory_) memory_text += m + "\n";

        // ステップ1: ファイル読み込みが必要か？
        std::vector<std::string> file_keywords = {"ファイル", "読む", "read", "内容", "中身"};
        bool needs_file = std::any_of(file_keywords.begin(), file_keywords.end(),
            [&](const std::string& kw){ return goal_lower.find(kw) != std::string::npos; });
        if (needs_file && memory_text.find("read_file") == std::string::npos) {
            std::regex fname_re(R"([\w\-\.]+\.(txt|md|csv|json|py))");
            std::smatch m;
            std::string fname = "sample.txt";
            if (std::regex_search(goal, m, fname_re)) fname = m[0];
            return "{\"thought\": \"ファイルの内容を確認する必要があります\", "
                   "\"action\": \"read_file\", "
                   "\"action_input\": \"" + fname + "\"}";
        }

        // ステップ2: 計算が必要か？
        std::vector<std::string> calc_keywords = {"計算", "足す", "引く", "掛ける", "割る",
                                                   "合計", "平均", "+ ", "- ", "* ", "/ "};
        bool needs_calc = std::any_of(calc_keywords.begin(), calc_keywords.end(),
            [&](const std::string& kw){ return goal_lower.find(kw) != std::string::npos; });
        if (needs_calc && memory_text.find("calculator") == std::string::npos) {
            std::regex num_re("\\d+");
            std::sregex_iterator it(goal.begin(), goal.end(), num_re);
            std::sregex_iterator end;
            std::vector<std::string> numbers;
            for (; it != end && numbers.size() < 2; ++it) numbers.push_back(it->str());
            std::string expr = (numbers.size() >= 2) ? (numbers[0] + " + " + numbers[1]) : "1 + 1";
            return "{\"thought\": \"計算が必要です\", "
                   "\"action\": \"calculator\", "
                   "\"action_input\": \"" + expr + "\"}";
        }

        // ステップ3: 時刻が必要か？
        std::vector<std::string> time_keywords = {"時間", "時刻", "日時", "今何時",
                                                   "today", "now", "time"};
        bool needs_time = std::any_of(time_keywords.begin(), time_keywords.end(),
            [&](const std::string& kw){ return goal_lower.find(kw) != std::string::npos; });
        if (needs_time && memory_text.find("get_time") == std::string::npos) {
            return "{\"thought\": \"現在時刻を確認します\", "
                   "\"action\": \"get_time\", "
                   "\"action_input\": \"\"}";
        }

        // ステップ4: 終了
        return "{\"thought\": \"必要な情報が揃いました。タスクを完了します\", "
               "\"action\": \"finish\", "
               "\"action_input\": \"\"}";
    }

    std::string run(const std::string& goal) {
        memory_.clear();
        std::cout << "============================================================" << std::endl;
        std::cout << " 目標: " << goal << std::endl;
        std::cout << "============================================================" << std::endl;

        for (int step = 1; step <= max_steps_; ++step) {
            std::cout << std::endl << "--- ステップ " << step << " ---" << std::endl;

            std::string thought_json = think(goal);
            ReActResult parsed = parseReActJson(thought_json);

            std::cout << " 思考: " << parsed.thought << std::endl;
            std::cout << " 行動: " << parsed.action << "(" << parsed.action_input << ")" << std::endl;

            if (parsed.action == "finish") {
                std::cout << std::endl << " タスク完了" << std::endl;
                break;
            }

            std::string observation;
            auto it = tools_.find(parsed.action);
            if (it != tools_.end()) {
                observation = it->second(parsed.action_input);
            } else {
                observation = "未知のツール: " + parsed.action;
            }

            std::cout << " 観察: " << observation << std::endl;
            memory_.push_back("Step" + std::to_string(step) + ": " + parsed.action +
                              "(" + parsed.action_input + ") -> " + observation);
        }

        std::cout << std::endl << "============================================================" << std::endl;
        std::cout << " 実行ログ:" << std::endl;
        if (memory_.empty()) {
            std::cout << " (ログなし)" << std::endl;
        } else {
            for (const auto& m : memory_) std::cout << "  " << m << std::endl;
        }
        std::cout << "============================================================" << std::endl;

        std::string result;
        for (const auto& m : memory_) result += m + "\n";
        return result;
    }

    void registerTool(const std::string& name, ToolFunc func) {
        tools_[name] = func;
    }

private:
    int max_steps_;
    std::vector<std::string> memory_;
    std::map<std::string, ToolFunc> tools_;
};

// ============================================================
// メイン
// ============================================================
int main() {
    SimpleAgent agent(10);

    // ツール登録
    agent.registerTool("calculator", tool_calculator);
    agent.registerTool("read_file", tool_read_file);
    agent.registerTool("get_time", tool_get_time);

    // サンプルファイル作成
    {
        std::ofstream ofs("sample.txt");
        if (ofs.is_open()) {
            ofs << "これはサンプルファイルです。\nAIエージェントのテスト用データ。\n";
        }
    }

    std::vector<std::string> goals = {
        "sample.txt のファイル内容を読んで、現在の時刻も教えて",
        "23 + 47 を計算して",
        "今の日時を教えて",
    };

    for (const auto& goal : goals) {
        agent.run(goal);
        std::cout << std::endl << std::endl;
    }

    return 0;
}
