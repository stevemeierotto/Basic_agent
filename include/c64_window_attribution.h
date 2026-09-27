/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C6.4 prospective window attribution
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_C64_WINDOW_ATTRIBUTION_H
#define THOTH_C64_WINDOW_ATTRIBUTION_H

#include <cstdint>
#include <string>

namespace Thoth {

/** Inclusive bounds from C6.4 v1.0: goal_started_at_ms ∈ [start, end]. */
inline constexpr std::int64_t kC64WindowSpanMs = 28LL * 86'400'000LL;

bool c64GoalStartedInWindow(std::int64_t windowStartMs,
                            std::int64_t windowEndMs,
                            std::int64_t goalStartedMs);

struct C64WindowAssignment {
    bool assigned = false;
    std::string window_id;
    std::string protocol_version;
    std::string metric_schema_version;
    std::string environment_schema_version;
    std::string evaluation_tier;
    std::string c64_cohort_fingerprint;
};

/**
 * Read THOTH_C64_WINDOW_FILE. Attribute only when the file is an open
 * authoritative c64-env-1 window, the goal start is inside its bounds,
 * and THOTH_C64_CURRENT_FINGERPRINT equals the frozen fingerprint.
 * Does not modify the window file and does not open a window.
 */
C64WindowAssignment resolveC64WindowAssignment(std::int64_t goalStartedMs);

} // namespace Thoth

#endif
