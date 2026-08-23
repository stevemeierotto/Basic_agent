/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Phase 0 chat turn worker-queue timing
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/chat_turn_timing.h"

namespace Thoth {
namespace ChatTurnTiming {

namespace {

thread_local WorkerContext g_active_context{};
thread_local bool g_has_active_context = false;

} // namespace

void setActiveWorkerContext(const WorkerContext& context) {
    g_active_context = context;
    g_has_active_context = true;
}

void clearActiveWorkerContext() {
    g_has_active_context = false;
    g_active_context = {};
}

std::optional<WorkerContext> consumeWorkerContext() {
    if (!g_has_active_context) {
        return std::nullopt;
    }
    const WorkerContext out = g_active_context;
    clearActiveWorkerContext();
    return out;
}

} // namespace ChatTurnTiming
} // namespace Thoth
