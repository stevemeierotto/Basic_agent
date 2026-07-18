/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — memory range restore public API helpers (M4)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/restore_api.h"

namespace Thoth {

std::string restoreModeToString(RestoreMode mode) {
    switch (mode) {
        case RestoreMode::REPLAY: return "REPLAY";
        case RestoreMode::REHYDRATE: return "REHYDRATE";
        default: return "UNKNOWN";
    }
}

} // namespace Thoth
