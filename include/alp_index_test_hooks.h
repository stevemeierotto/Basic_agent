/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP-B test hooks (env-gated; unit tests only)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ALP_INDEX_TEST_HOOKS_H
#define THOTH_ALP_INDEX_TEST_HOOKS_H

#include "alp_feature_flags.h"

#include <cstdlib>
#include <string>

namespace Thoth {
namespace AlpIndexTestHooks {

inline bool forceSaveIndexFailure() {
    return AlpFeatureFlags::truthyEnv(std::getenv("THOTH_ALP_FORCE_SAVE_INDEX_FAIL"));
}

/** Reset per-indexFile embed attempt counter (call at start of each indexFile). */
inline void resetEmbedAttemptCounter() {
    static thread_local size_t counter = 0;
    counter = 0;
}

/**
 * Returns true when embed should be forced to fail for this attempt.
 * THOTH_ALP_TEST_EMBED_FAIL_AFTER=N — fail attempts with index >= N (0 = fail all).
 */
inline bool shouldForceEmbedFailure() {
    const char* env = std::getenv("THOTH_ALP_TEST_EMBED_FAIL_AFTER");
    if (!env || !*env) {
        return false;
    }
    static thread_local size_t counter = 0;
    const size_t threshold = static_cast<size_t>(std::stoul(env));
    const size_t attempt = counter++;
    return attempt >= threshold;
}

} // namespace AlpIndexTestHooks
} // namespace Thoth

#endif // THOTH_ALP_INDEX_TEST_HOOKS_H
