/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Shared text-generation timeout policy
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "plan.h"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>

namespace Thoth::LlmTimeoutPolicy {

inline constexpr long kDefaultSeconds = 900;

inline long timeoutSeconds() {
    if (const char* raw = std::getenv("THOTH_LLM_TIMEOUT_SECONDS")) {
        try {
            const std::string value(raw);
            size_t consumed = 0;
            const long parsed = std::stol(value, &consumed);
            // Keep the shared budget representable in the step's millisecond field.
            if (consumed == value.size() && parsed > 0 &&
                parsed <= std::numeric_limits<int>::max() / 1000) {
                return parsed;
            }
        } catch (...) {
            // Invalid or out-of-range overrides use the default.
        }
    }
    return kDefaultSeconds;
}

inline int stepTimeoutMs(StepType type, int requestedMs) {
    if (type == StepType::LLM) {
        // This bounds the whole step, including retries, as before.
        return std::max(requestedMs, static_cast<int>(timeoutSeconds() * 1000));
    }
    return requestedMs > 0 ? requestedMs : 30000;
}

} // namespace Thoth::LlmTimeoutPolicy
