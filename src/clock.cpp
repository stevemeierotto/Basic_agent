/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — clock implementations
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/clock.h"

namespace Thoth {

namespace {

int64_t systemNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

constexpr int64_t kMsPerDay = 86'400'000LL;

} // namespace

int64_t SystemClock::nowMs() const {
    return systemNowMs();
}

FakeClock::FakeClock(int64_t startMs)
    : now_ms_(startMs > 0 ? startMs : systemNowMs()) {}

int64_t FakeClock::nowMs() const {
    return now_ms_;
}

void FakeClock::setNowMs(int64_t ms) {
    now_ms_ = ms;
}

void FakeClock::advanceMs(int64_t deltaMs) {
    now_ms_ += deltaMs;
}

void FakeClock::advanceDays(int days) {
    now_ms_ += days * kMsPerDay;
}

std::shared_ptr<Clock> makeSystemClock() {
    return std::make_shared<SystemClock>();
}

} // namespace Thoth
