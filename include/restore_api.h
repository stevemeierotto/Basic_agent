/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — memory range restore public API (M4)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_RESTORE_API_H
#define THOTH_RESTORE_API_H

#include "memory_repository.h"
#include <cstdint>
#include <string>
#include <vector>

namespace Thoth {

enum class RestoreMode : uint8_t {
    REPLAY,
    REHYDRATE,
};

struct RestoreRequest {
    RestoreMode mode = RestoreMode::REPLAY;
    RestoreRange range;
    bool allow_during_goal = false;
    std::string requested_by;
};

struct RestoreResult {
    RestoreMode mode = RestoreMode::REPLAY;
    int matched = 0;
    int restored = 0;
    int skipped_dup = 0;
    bool blocked = false;
    std::string block_reason;
    std::vector<MemoryRepository::ArchivedTurnRecord> turns;
};

std::string restoreModeToString(RestoreMode mode);

} // namespace Thoth

#endif // THOTH_RESTORE_API_H
