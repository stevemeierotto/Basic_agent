/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — experience provenance and learning eligibility
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_EXPERIENCE_PROVENANCE_H
#define THOTH_EXPERIENCE_PROVENANCE_H

#include <string>

namespace Thoth {

inline constexpr const char* kPlanOriginModel = "model";
inline constexpr const char* kPlanOriginFallback = "fallback";
inline constexpr const char* kPlanOriginUnknown = "unknown";

/** model and fallback pass through. Every other value, including empty, is unknown. */
inline std::string normalizePlanOrigin(const std::string& raw) {
    if (raw == kPlanOriginModel || raw == kPlanOriginFallback) {
        return raw;
    }
    return kPlanOriginUnknown;
}

/** Only model-origin experience may influence later planning. */
inline bool learningEligibleOrigin(const std::string& raw) {
    return normalizePlanOrigin(raw) == kPlanOriginModel;
}

} // namespace Thoth

#endif // THOTH_EXPERIENCE_PROVENANCE_H
