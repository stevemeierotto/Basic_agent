/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — runtime bootstrap and startup diagnostics (Plan E)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "runtime_bootstrap.h"

#include "config.h"
#include "embedding_engine.h"
#include "env_loader.h"
#include "file_handler.h"
#include "inference_client.h"
#include "inference_endpoint.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>

namespace Thoth {
namespace {

std::once_flag g_bootstrapOnce;

constexpr const char* kEmbedProbeInput = "thoth-embedding-probe";

bool truthyEnvFlag(const char* value) {
    if (!value || !*value) {
        return false;
    }
    const std::string flag(value);
    return flag == "1" || flag == "true" || flag == "TRUE" || flag == "yes" || flag == "YES";
}

bool testSuiteDevTierEnabled() {
    const char* dev = std::getenv("THOTH_TEST_SUITE_DEV");
    return dev && (std::string(dev) == "1" || std::string(dev) == "true");
}

std::string resolveEmbeddingModelName(const Config* config) {
    if (config && !config->embedding_model.empty()) {
        return config->embedding_model;
    }
    return "nomic-embed-text:v1.5";
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

    std::string backend_error;
    const auto backend = tryResolveInferenceBackend(backend_error);
    const std::string backend_name =
        backend ? inferenceBackendName(*backend)
                : (backend_error.empty() ? "unknown" : "invalid");

    std::cerr << "[Thoth] project_root=" << fileHandler.getProjectRoot() << '\n'
              << "[Thoth] workspace=" << fileHandler.getAgentWorkspacePath() << '\n'
              << "[Thoth] logs=" << fileHandler.getLogsPath() << '\n'
              << "[Thoth] inference_backend=" << backend_name << '\n'
              << "[Thoth] inference_base=" << endpoints.base_url << '\n'
              << "[Thoth] embed_base=" << endpoints.embed_base_url << '\n';
    if (config) {
        std::cerr << "[Thoth] llm_model=" << config->llm_model << '\n'
                  << "[Thoth] embedding_model=" << config->embedding_model << '\n';
    }
    std::cerr << "[Thoth] database="
              << (std::filesystem::path(fileHandler.getAgentWorkspacePath()) / "memory.db").string()
              << '\n'
              << "[Thoth] config=" << fileHandler.getAgentWorkspacePath("config.json") << '\n';
}

void logEmbeddingStartupProbe(const Config* config) {
    if (testSuiteDevTierEnabled()) {
        if (runtimeConfigDiagnosticsEnabled(config)) {
            std::cerr << "[Thoth] embed_probe=skipped (THOTH_TEST_SUITE_DEV TfIdf tier)\n";
        }
        return;
    }

    const auto endpoints = config != nullptr ? resolveInferenceEndpoints(*config)
                                           : resolveInferenceEndpoints();
    const std::string model = resolveEmbeddingModelName(config);
    const int expected_dim = EmbeddingEngine(EmbeddingEngine::Method::External, nullptr).getDimension();

    std::string backend_error;
    const auto backend = tryResolveInferenceBackend(backend_error);
    const std::string backend_name =
        backend ? inferenceBackendName(*backend)
                : (backend_error.empty() ? "unknown" : "invalid");

    if (!backend) {
        std::cerr << "[Thoth] embed_probe=failed backend=" << backend_name
                  << " error=" << backend_error << '\n';
        return;
    }

    try {
        const auto client = createInferenceClient(endpoints, config);
        InferenceEmbedRequest request;
        request.model = model;
        request.inputs = {kEmbedProbeInput};

        const auto result = client->embed(request);
        if (!result.ok || result.embeddings.empty()) {
            std::cerr << "[Thoth] embed_probe=failed backend=" << client->backendName()
                      << " embed_base=" << endpoints.embed_base_url
                      << " model=" << model;
            if (!result.error.empty()) {
                std::cerr << " error=" << result.error;
            }
            std::cerr << '\n';
            return;
        }

        const std::size_t dim = result.embeddings.front().size();
        if (static_cast<int>(dim) != expected_dim) {
            std::cerr << "[Thoth] embed_probe=failed backend=" << client->backendName()
                      << " embed_base=" << endpoints.embed_base_url
                      << " model=" << model << " dimension=" << dim
                      << " expected=" << expected_dim << '\n';
            return;
        }

        std::cerr << "[Thoth] embed_probe=ok backend=" << client->backendName()
                  << " embed_base=" << endpoints.embed_base_url
                  << " model=" << model << " dimension=" << dim << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "[Thoth] embed_probe=failed backend=" << backend_name
                  << " embed_base=" << endpoints.embed_base_url
                  << " model=" << model << " error=" << ex.what() << '\n';
    }
}

} // namespace Thoth
