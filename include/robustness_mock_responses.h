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

/** Thread-safe FIFO queue consumed by LLMInterface during robustness / Plan N tests. */
class RobustnessMockResponses {
public:
    static void reset();
    static void push(std::string response);
    static void pushAll(const std::vector<std::string>& responses);
    /** Plan N N6 — next queryDetailed returns ok=false with this error (Class A). */
    static void pushFailure(std::string error);
    static std::optional<std::string> pop();
    static std::optional<std::string> popFailure();
    static std::size_t size();
};

} // namespace Thoth

#endif // THOTH_ROBUSTNESS_MOCK_RESPONSES_H
