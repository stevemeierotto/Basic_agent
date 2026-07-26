/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — runtime bootstrap and startup diagnostics (Plan E)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

class Config;

namespace Thoth {

/** Load .env once (idempotent). Shell-exported variables are not overwritten. */
void bootstrapRuntimeEnvironment();

/** Print resolved workspace/log/inference paths when diagnostics are enabled. */
void logResolvedRuntimeConfig(const Config* config = nullptr);

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
