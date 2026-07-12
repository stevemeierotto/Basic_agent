/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — shared memory pruning tier limits (SQLite hot tier + GUI chat_sessions.json)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_MEMORY_PRUNING_CONFIG_H
#define THOTH_MEMORY_PRUNING_CONFIG_H

#include <cstddef>

namespace Thoth::MemoryPruning {

/** Hot-tier capacity: raw turns kept in SQLite `messages` and chat_sessions.json. */
constexpr std::size_t kMaxHotMessages = 50;

/** Turns moved to cold tier (`archived_turns`) when hot tier exceeds capacity. */
constexpr std::size_t kPruneBatchSize = 10;

/** Max consolidation batches per single invocation (M2). */
constexpr std::size_t kMaxBatchesPerInvocation = 5;

/**
 * Circuit breaker: after this many consecutive automatic consolidation
 * attempts that make no forward progress (0 turns archived while the policy
 * still wants consolidation), stop retrying on every message. This prevents a
 * persistent failure (e.g. a poisoned DB transaction, an unavailable embedding
 * backend, or a stalled LLM) from re-running expensive, multi-minute
 * consolidation work on the worker thread for every new turn — the failure
 * mode that could freeze the control panel. The backoff is cleared on a
 * successful archive, an explicit (manual) consolidation, or session
 * (re)activation.
 */
constexpr int kMaxConsecutiveNoProgress = 3;

} // namespace Thoth::MemoryPruning

#endif // THOTH_MEMORY_PRUNING_CONFIG_H
