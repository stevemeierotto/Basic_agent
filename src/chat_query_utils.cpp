/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Chat query intent helpers (C2 Phase 3 tool gating)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/chat_query_utils.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace Thoth {

namespace {

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string trimCopy(const std::string& value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c);
    });
    if (first == value.end()) {
        return "";
    }
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c);
    }).base();
    return std::string(first, last);
}

std::string normalizeGreetingCandidate(const std::string& query) {
    std::string normalized = lowerCopy(trimCopy(query));
    while (!normalized.empty() &&
           (normalized.back() == '!' || normalized.back() == '?' || normalized.back() == '.')) {
        normalized.pop_back();
    }
    return trimCopy(normalized);
}

bool containsInterrogativeToken(const std::string& query) {
    static const std::unordered_set<std::string> kInterrogatives = {
        "explain", "what", "how", "why",
    };

    std::string normalized;
    normalized.reserve(query.size());
    for (unsigned char c : query) {
        if (std::isalnum(c)) {
            normalized.push_back(static_cast<char>(std::tolower(c)));
        } else {
            normalized.push_back(' ');
        }
    }

    std::istringstream stream(normalized);
    std::string token;
    while (stream >> token) {
        if (kInterrogatives.count(token) > 0) {
            return true;
        }
    }
    return false;
}

} // namespace

bool looksLikeToolIntent(const std::string& query) {
    const std::string q = lowerCopy(query);

    static const std::vector<const char*> kTriggers = {
        "run test",
        "run_tests",
        "execute tool",
        "use tool",
        "call tool",
        "tool_call",
        "store_fact",
        "web_scrape",
        "code_modify",
        "gmail",
        "summarize_text",
        "project_analyze",
        "self_correct",
        "apply diff",
        "build project",
    };

    for (const char* trigger : kTriggers) {
        if (q.find(trigger) != std::string::npos) {
            return true;
        }
    }

    return false;
}

bool isGreetingSkipQuery(const std::string& query) {
    // Plan M G2: exact-match is the primary rule. The interrogative check is only
    // defensive for future phrase additions; current locked greetings already
    // reject inputs like "thanks why?" by exact-match.
    if (containsInterrogativeToken(query)) {
        return false;
    }

    static const std::unordered_set<std::string> kGreetingPhrases = {
        "hello",
        "hi",
        "hey",
        "howdy",
        "yo",
        "thanks",
        "thank you",
        "thx",
        "good morning",
        "good afternoon",
        "good evening",
    };

    return kGreetingPhrases.count(normalizeGreetingCandidate(query)) > 0;
}

} // namespace Thoth
