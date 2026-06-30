/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — memory retrieval/storage scope
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_MEMORY_SCOPE_H
#define THOTH_MEMORY_SCOPE_H

namespace Thoth {

enum class MemoryScope : int {
    SESSION = 0,
    PROJECT = 1,
    USER    = 2,
    GLOBAL  = 3
};

} // namespace Thoth

#endif
