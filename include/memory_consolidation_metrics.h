/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — last consolidation timing (observability + M1.5 tests)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_MEMORY_CONSOLIDATION_METRICS_H
#define THOTH_MEMORY_CONSOLIDATION_METRICS_H

#include <cstdint>

namespace Thoth {

struct ConsolidationTiming {
    int64_t summary_ms = 0;
    int64_t embed_ms = 0;
    int64_t transaction_ms = 0;
    int64_t consolidation_ms = 0;
};

/** Updated after each successful or attempted consolidation (for tests and diagnostics). */
const ConsolidationTiming& lastConsolidationTiming();

void resetConsolidationTimingForTest();

/** Called by MemoryPruner after each consolidation attempt. */
void recordConsolidationTiming(const ConsolidationTiming& timing);

} // namespace Thoth

#endif
