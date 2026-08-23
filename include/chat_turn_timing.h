/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Phase 0 chat turn worker-queue timing (thread-local, telemetry only)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include <cstdint>
#include <optional>

namespace Thoth {
namespace ChatTurnTiming {

/** Set by EngineRuntime worker before plugin turn processing; consumed once in processQuery. */
struct WorkerContext {
    std::int64_t enqueued_at_ms = 0;
    std::int64_t worker_started_at_ms = 0;
};

void setActiveWorkerContext(const WorkerContext& context);
void clearActiveWorkerContext();
/** Returns context if present; clears thread-local storage (single consume per turn). */
std::optional<WorkerContext> consumeWorkerContext();

} // namespace ChatTurnTiming
} // namespace Thoth
