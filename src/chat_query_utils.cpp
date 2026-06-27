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
#include <string>
#include <vector>

namespace Thoth {

namespace {

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
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

} // namespace Thoth
