/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Chat query intent helpers (C2 Phase 3 tool gating)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CHAT_QUERY_UTILS_H
#define THOTH_CHAT_QUERY_UTILS_H

#include <string>

namespace Thoth {

/** True when the user message likely requests tool execution (not plain Q&A). */
bool looksLikeToolIntent(const std::string& query);

} // namespace Thoth

#endif // THOTH_CHAT_QUERY_UTILS_H
