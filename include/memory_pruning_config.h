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

} // namespace Thoth::MemoryPruning

#endif // THOTH_MEMORY_PRUNING_CONFIG_H
