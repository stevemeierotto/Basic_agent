/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — consolidation timing storage
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/memory_consolidation_metrics.h"

namespace Thoth {

namespace {
ConsolidationTiming g_lastTiming;
}

const ConsolidationTiming& lastConsolidationTiming() {
    return g_lastTiming;
}

void resetConsolidationTimingForTest() {
    g_lastTiming = {};
}

void recordConsolidationTiming(const ConsolidationTiming& timing) {
    g_lastTiming = timing;
}

} // namespace Thoth
