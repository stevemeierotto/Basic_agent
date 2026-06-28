/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C5 robustness harness scripted LLM responses
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/robustness_mock_responses.h"
#include <mutex>
#include <queue>

namespace Thoth {

namespace {
std::mutex g_mutex;
std::queue<std::string> g_responses;
} // namespace

void RobustnessMockResponses::reset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    while (!g_responses.empty()) {
        g_responses.pop();
    }
}

void RobustnessMockResponses::push(std::string response) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_responses.push(std::move(response));
}

void RobustnessMockResponses::pushAll(const std::vector<std::string>& responses) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto& r : responses) {
        g_responses.push(r);
    }
}

std::optional<std::string> RobustnessMockResponses::pop() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_responses.empty()) {
        return std::nullopt;
    }
    std::string front = std::move(g_responses.front());
    g_responses.pop();
    return front;
}

std::size_t RobustnessMockResponses::size() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_responses.size();
}

} // namespace Thoth
