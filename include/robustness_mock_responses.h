/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C5 robustness harness scripted LLM responses
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ROBUSTNESS_MOCK_RESPONSES_H
#define THOTH_ROBUSTNESS_MOCK_RESPONSES_H

#include <optional>
#include <string>
#include <vector>

namespace Thoth {

/** Thread-safe FIFO queue consumed by LLMInterface::query() during robustness tests. */
class RobustnessMockResponses {
public:
    static void reset();
    static void push(std::string response);
    static void pushAll(const std::vector<std::string>& responses);
    static std::optional<std::string> pop();
    static std::size_t size();
};

} // namespace Thoth

#endif // THOTH_ROBUSTNESS_MOCK_RESPONSES_H
