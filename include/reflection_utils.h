/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C3 reflection policy helpers
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_REFLECTION_UTILS_H
#define THOTH_REFLECTION_UTILS_H

#include "trajectory.h"
#include <string>

namespace Thoth {

/** True when any recorded step error indicates a wall-clock timeout (not low-quality score). */
bool trajectoryHasTimeoutFailure(const Trajectory& trajectory);

/** Human-readable skip reason when reflection is suppressed; empty if reflection may proceed. */
std::string reflectionSkipReason(bool reflectionEnabled,
                                 int reflectionCount,
                                 int maxReflections,
                                 float score,
                                 float scoreThreshold,
                                 bool timeoutFailure);

} // namespace Thoth

#endif // THOTH_REFLECTION_UTILS_H
