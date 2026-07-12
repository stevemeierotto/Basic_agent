/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — runtime bootstrap and startup diagnostics (Plan E)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "runtime_bootstrap.h"

#include "config.h"
#include "env_loader.h"
#include "file_handler.h"
#include "inference_endpoint.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>

namespace Thoth {
namespace {

std::once_flag g_bootstrapOnce;

bool truthyEnvFlag(const char* value) {
    if (!value || !*value) {
        return false;
    }
    const std::string flag(value);
    return flag == "1" || flag == "true" || flag == "TRUE" || flag == "yes" || flag == "YES";
}

} // namespace

void bootstrapRuntimeEnvironment() {
    std::call_once(g_bootstrapOnce, []() {
        FileHandler fileHandler;
        const std::string envPath = fileHandler.getEnvPath();
        if (!std::filesystem::exists(envPath)) {
            return;
        }
        EnvLoader::loadEnvFileIfUnset(envPath);
    });
}

bool runtimeConfigDiagnosticsEnabled(const Config* config) {
    if (truthyEnvFlag(std::getenv("THOTH_LOG_CONFIG"))) {
        return true;
    }
    return config != nullptr && config->verbosity >= 2;
}

void logResolvedRuntimeConfig(const Config* config) {
    if (!runtimeConfigDiagnosticsEnabled(config)) {
        return;
    }

    FileHandler fileHandler;
    const auto endpoints = config != nullptr ? resolveInferenceEndpoints(*config)
                                           : resolveInferenceEndpoints();

    std::cerr << "[Thoth] project_root=" << fileHandler.getProjectRoot() << '\n'
              << "[Thoth] workspace=" << fileHandler.getAgentWorkspacePath() << '\n'
              << "[Thoth] logs=" << fileHandler.getLogsPath() << '\n'
              << "[Thoth] inference_base=" << endpoints.base_url << '\n'
              << "[Thoth] embed_base=" << endpoints.embed_base_url << '\n'
              << "[Thoth] database="
              << (std::filesystem::path(fileHandler.getAgentWorkspacePath()) / "memory.db").string()
              << '\n'
              << "[Thoth] config=" << fileHandler.getAgentWorkspacePath("config.json") << '\n';
}

} // namespace Thoth
