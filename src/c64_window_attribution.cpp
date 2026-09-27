/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C6.4 prospective window attribution
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "../include/c64_window_attribution.h"

#include "../include/json.hpp"

#include <cstdlib>
#include <fstream>

namespace Thoth {

namespace {

std::string envOrEmpty(const char* key) {
    const char* value = std::getenv(key);
    return (value && *value) ? std::string(value) : std::string();
}

} // namespace

bool c64GoalStartedInWindow(std::int64_t windowStartMs,
                            std::int64_t windowEndMs,
                            std::int64_t goalStartedMs) {
    return goalStartedMs >= windowStartMs && goalStartedMs <= windowEndMs;
}

C64WindowAssignment resolveC64WindowAssignment(std::int64_t goalStartedMs) {
    C64WindowAssignment none;
    const std::string path = envOrEmpty("THOTH_C64_WINDOW_FILE");
    const std::string current = envOrEmpty("THOTH_C64_CURRENT_FINGERPRINT");
    if (path.empty() || current.empty()) {
        return none;
    }
    std::ifstream in(path);
    if (!in) {
        return none;
    }
    nlohmann::json window;
    try {
        in >> window;
    } catch (...) {
        return none;
    }
    if (!window.is_object() || window.value("status", "") != "open") {
        return none;
    }
    if (window.value("evaluation_tier", "") != "authoritative" ||
        window.value("environment_schema_version", "") != "c64-env-1" ||
        window.value("protocol_version", "") != "C6.4 v1.0") {
        return none;
    }
    const std::string windowId = window.value("window_id", "");
    const std::string frozen = window.value("c64_cohort_fingerprint", "");
    if (windowId.empty() || frozen.empty() || frozen != current) {
        return none;
    }
    const auto start = window.value("window_start_ms", static_cast<std::int64_t>(0));
    const auto end = window.value("window_end_ms", static_cast<std::int64_t>(0));
    if (!c64GoalStartedInWindow(start, end, goalStartedMs)) {
        return none;
    }
    C64WindowAssignment assigned;
    assigned.assigned = true;
    assigned.window_id = windowId;
    assigned.protocol_version = window.value("protocol_version", "");
    assigned.metric_schema_version = window.value("metric_schema_version", "");
    assigned.environment_schema_version = window.value("environment_schema_version", "");
    assigned.evaluation_tier = window.value("evaluation_tier", "");
    assigned.c64_cohort_fingerprint = frozen;
    return assigned;
}

} // namespace Thoth
