/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E1 Ollama snapshot helper (Checkpoint B)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_OLLAMA_SNAPSHOT_H
#define THOTH_OLLAMA_SNAPSHOT_H

#include "benchmark_environment.h"

#include <optional>
#include <string>

namespace Thoth {

struct OllamaFetchOptions {
    std::string base_url;
    long timeout_ms = 2000;
};

/** Best-effort HTTP fetch; empty optional when unreachable or parse fails. */
std::optional<OllamaSnapshot> fetchOllamaSnapshot(const OllamaFetchOptions& options = {});

/** True when /api/tags responds successfully within timeout. */
bool isOllamaReachable(const OllamaFetchOptions& options = {});

} // namespace Thoth

#endif // THOTH_OLLAMA_SNAPSHOT_H
