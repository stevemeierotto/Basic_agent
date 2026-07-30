/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP1 feature flags (Attachment Lifecycle Protocol)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ALP_FEATURE_FLAGS_H
#define THOTH_ALP_FEATURE_FLAGS_H

#include <cstdlib>
#include <string>

namespace Thoth {
namespace AlpFeatureFlags {

inline bool truthyEnv(const char* value) {
    if (!value || !*value) {
        return false;
    }
    const std::string flag(value);
    return flag == "1" || flag == "true" || flag == "TRUE" || flag == "yes" || flag == "YES";
}

/** Full ALP write path (UUID ids, attachments namespace, no suffix). Brownfield: after migration. */
inline bool alpEnabled() {
    return truthyEnv(std::getenv("THOTH_ALP_ENABLED"));
}

/** Transactional candidate → validate → commit indexing (ALP-B). */
inline bool transactionalIndexingEnabled() {
    return truthyEnv(std::getenv("THOTH_ALP_TX_INDEX"));
}

/** New GUI picker/reconcile (ALP-E). */
inline bool alpGuiEnabled() {
    return truthyEnv(std::getenv("THOTH_ALP_GUI"));
}

/** Skip migration on empty workspace install. */
inline bool alpGreenfield() {
    return truthyEnv(std::getenv("THOTH_ALP_GREENFIELD"));
}

/** ALP-C — ENABLED without TX_INDEX is misconfigured. */
inline bool alpMisconfigured() {
    return alpEnabled() && !transactionalIndexingEnabled();
}

/** ALP-C — operator create path requires both flags. */
inline bool alpCreateAllowed() {
    return alpEnabled() && transactionalIndexingEnabled();
}

} // namespace AlpFeatureFlags
} // namespace Thoth

#endif // THOTH_ALP_FEATURE_FLAGS_H
