/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Conservative synthesis repetition guard
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/synthesis_output_guard.h"

#include <cctype>
#include <string>
#include <unordered_map>
#include <vector>

namespace Thoth {
namespace {

std::string normalizeWs(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool prev_space = true;
    for (unsigned char ch : text) {
        if (std::isspace(ch)) {
            if (!prev_space) {
                out.push_back(' ');
                prev_space = true;
            }
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(ch)));
        prev_space = false;
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

std::vector<std::string> tokenizeWords(const std::string& normalized) {
    std::vector<std::string> words;
    std::string current;
    for (unsigned char ch : normalized) {
        if (std::isalnum(ch)) {
            current.push_back(static_cast<char>(ch));
        } else if (!current.empty()) {
            words.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        words.push_back(current);
    }
    return words;
}

bool lowNgramDiversity(const std::vector<std::string>& words, std::size_t n) {
    if (words.size() < n + 60) {
        return false;
    }
    const std::size_t total = words.size() - n + 1;
    if (total < 80) {
        return false;
    }

    std::unordered_map<std::string, int> counts;
    counts.reserve(total);
    int max_count = 0;
    for (std::size_t i = 0; i < total; ++i) {
        std::string key = words[i];
        for (std::size_t j = 1; j < n; ++j) {
            key.push_back(' ');
            key += words[i + j];
        }
        const int c = ++counts[key];
        if (c > max_count) {
            max_count = c;
        }
    }

    // Require a clear dominant repeat and overall low n-gram diversity.
    // Instruction-loop outputs cluster well below ~0.40 unique ratio; varied
    // diagnostic lists that only change indices stay much higher.
    if (max_count < 6) {
        return false;
    }
    const double unique_ratio =
        static_cast<double>(counts.size()) / static_cast<double>(total);
    return unique_ratio <= 0.42;
}

bool exactDuplicateLineDominance(const std::string& text) {
    std::unordered_map<std::string, int> line_counts;
    int nonempty = 0;
    std::string line;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        const bool at_end = i == text.size();
        const char ch = at_end ? '\n' : text[i];
        if (ch == '\n' || at_end) {
            const std::string norm = normalizeWs(line);
            line.clear();
            if (norm.size() < 24) {
                continue;
            }
            ++nonempty;
            ++line_counts[norm];
            continue;
        }
        line.push_back(ch);
    }
    if (nonempty < 10) {
        return false;
    }
    int max_dup = 0;
    for (const auto& kv : line_counts) {
        if (kv.second > max_dup) {
            max_dup = kv.second;
        }
    }
    return max_dup >= 5 &&
           (static_cast<double>(max_dup) / static_cast<double>(nonempty)) >= 0.40;
}

} // namespace

bool synthesisOutputLooksUnusableRepetition(const std::string& text) {
    if (text.size() < 400) {
        return false;
    }
    const std::string normalized = normalizeWs(text);
    const auto words = tokenizeWords(normalized);
    if (words.size() < 80) {
        return false;
    }
    if (lowNgramDiversity(words, 5) || lowNgramDiversity(words, 6)) {
        return true;
    }
    return exactDuplicateLineDominance(text);
}

} // namespace Thoth
