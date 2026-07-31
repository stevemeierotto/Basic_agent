/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — runtime bootstrap and startup diagnostics (Plan E)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "inference_endpoint.h"
#include "json.hpp"

#include <string>
#include <vector>

class Config;

namespace Thoth {

/** Load .env once (idempotent). Shell-exported variables are not overwritten. */
void bootstrapRuntimeEnvironment();

/** Print resolved workspace/log/inference paths when diagnostics are enabled. */
void logResolvedRuntimeConfig(const Config* config = nullptr);

struct InferenceMisconfigWarning {
    std::string code;
    std::string message;
};

/**
 * Detect backend/URL mismatches (e.g. ollama backend with llama-server URLs).
 * Pure function — safe for unit tests.
 */
std::vector<InferenceMisconfigWarning> detectInferenceBackendMisconfigs(
    const std::string& backend_name,
    const InferenceEndpointConfig& endpoints);

/** Log misconfiguration warnings to stderr (always, one line each). */
void logInferenceBackendMisconfigWarnings(const Config* config = nullptr);

struct EmbeddingProbeSnapshot {
    std::string status;  // ok | failed | skipped | unknown
    std::string backend;
    std::string embed_base_url;
    std::string model;
    int dimension = 0;
    std::string error;
};

/** Run embedding probe once; caches result for /ready. */
EmbeddingProbeSnapshot probeEmbeddingBackend(const Config* config = nullptr);

/** Last probe result from startup (status unknown if never probed). */
EmbeddingProbeSnapshot getLastEmbeddingProbeSnapshot();

/** JSON object for /ready embedding block. */
nlohmann::json embeddingProbeJson(const EmbeddingProbeSnapshot& snapshot);

/**
 * Probe the configured embedding endpoint once at startup.
 * Always logs success or failure; verbose detail when diagnostics are enabled.
 */
void logEmbeddingStartupProbe(const Config* config = nullptr);

/** Returns true when THOTH_LOG_CONFIG=1 or config verbosity >= 2. */
bool runtimeConfigDiagnosticsEnabled(const Config* config = nullptr);

/**
 * Defensive RAII guard — declare as the first member of runtime-owning classes
 * so bootstrap runs before other members that read THOTH_* environment variables.
 */
struct RuntimeBootstrapGuard {
    RuntimeBootstrapGuard() { bootstrapRuntimeEnvironment(); }
};

} // namespace Thoth
