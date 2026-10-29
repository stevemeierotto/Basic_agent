/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — CM-1 idle discovery model priority
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include <atomic>

namespace Thoth {

/**
 * Accepted goals that have not yet left their execution loop.
 * Discovery reads this without taking the controller mutex.
 * An early return before the loop starts drops the same count the loop would have dropped.
 * A newer goal increments before it joins the previous loop, so the count stays above zero
 * across that overlap.
 */
inline std::atomic<int>& cm1ForegroundGoalCount() {
    static std::atomic<int> count{0};
    return count;
}

inline bool cm1ForegroundWorkActive() {
    return cm1ForegroundGoalCount().load(std::memory_order_acquire) > 0;
}

/** Set only around a discovery model call. Goal planning does not set it. */
inline thread_local int cm1DiscoveryYieldDepth = 0;

struct Cm1DiscoveryYieldScope {
    Cm1DiscoveryYieldScope() { ++cm1DiscoveryYieldDepth; }
    ~Cm1DiscoveryYieldScope() { --cm1DiscoveryYieldDepth; }
    Cm1DiscoveryYieldScope(const Cm1DiscoveryYieldScope&) = delete;
    Cm1DiscoveryYieldScope& operator=(const Cm1DiscoveryYieldScope&) = delete;
};

inline bool cm1DiscoveryYieldArmed() {
    return cm1DiscoveryYieldDepth > 0;
}

/** Returned by a discovery model call that gave way before HTTP. The writer stores nothing. */
inline constexpr const char* kCm1DiscoveryYieldError = "cm1_discovery_yield";

/**
 * Test seam. Null in production.
 * The idle worker invokes it on the discovery thread immediately before the
 * yield check, and before any model call.
 */
inline std::atomic<void (*)()>& cm1DiscoveryBeforeCallForTests() {
    static std::atomic<void (*)()> probe{nullptr};
    return probe;
}

} // namespace Thoth
