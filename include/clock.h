/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — injectable clock for consolidation policy (M2)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CLOCK_H
#define THOTH_CLOCK_H

#include <chrono>
#include <cstdint>
#include <memory>

namespace Thoth {

class Clock {
public:
    virtual ~Clock() = default;
    virtual int64_t nowMs() const = 0;
};

class SystemClock : public Clock {
public:
    int64_t nowMs() const override;
};

/** Test-only clock; advanceMs() moves time forward deterministically. */
class FakeClock : public Clock {
public:
    explicit FakeClock(int64_t startMs = 0);

    int64_t nowMs() const override;
    void setNowMs(int64_t ms);
    void advanceMs(int64_t deltaMs);
    void advanceDays(int days);

private:
    int64_t now_ms_;
};

std::shared_ptr<Clock> makeSystemClock();

} // namespace Thoth

#endif // THOTH_CLOCK_H
