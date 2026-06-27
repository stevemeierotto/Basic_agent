/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C3 reflection policy helpers
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/reflection_utils.h"
#include <cctype>

namespace Thoth {

namespace {

bool errorLooksLikeTimeout(const std::string& error) {
    if (error.empty()) {
        return false;
    }
    const std::string lower = [&error]() {
        std::string value = error;
        for (char& c : value) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return value;
    }();
    return lower.find("timed out") != std::string::npos ||
           lower.find("timeout") != std::string::npos;
}

} // namespace

bool trajectoryHasTimeoutFailure(const Trajectory& trajectory) {
    for (const auto& step : trajectory.steps) {
        if (errorLooksLikeTimeout(step.error)) {
            return true;
        }
    }
    return false;
}

std::string reflectionSkipReason(bool reflectionEnabled,
                                 int reflectionCount,
                                 int maxReflections,
                                 float score,
                                 float scoreThreshold,
                                 bool timeoutFailure) {
    if (!reflectionEnabled || maxReflections <= 0) {
        return "reflection_disabled";
    }
    if (score >= scoreThreshold) {
        return "";
    }
    if (timeoutFailure) {
        return "timeout_failure";
    }
    if (reflectionCount >= maxReflections) {
        return "max_reflections_exhausted";
    }
    return "";
}

} // namespace Thoth
