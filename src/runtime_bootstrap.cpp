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
std::mutex g_probeMutex;
EmbeddingProbeSnapshot g_lastProbeSnapshot;
bool g_probeCached = false;

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

bool urlHintsLlamaCppService(const std::string& url) {
    return url.find("llama-server") != std::string::npos
        || url.find("llama-embed") != std::string::npos
        || url.find(":8080") != std::string::npos
        || url.find(":8081") != std::string::npos;
}

bool urlHintsOllamaService(const std::string& url) {
    return url.find(":11434") != std::string::npos
        || url.find("ollama") != std::string::npos;
}

void cacheProbeSnapshot(EmbeddingProbeSnapshot snapshot) {
    std::lock_guard<std::mutex> lock(g_probeMutex);
    g_lastProbeSnapshot = std::move(snapshot);
    g_probeCached = true;
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

std::vector<InferenceMisconfigWarning> detectInferenceBackendMisconfigs(
    const std::string& backend_name,
    const InferenceEndpointConfig& endpoints) {
    std::vector<InferenceMisconfigWarning> warnings;

    if (backend_name == "ollama") {
        if (urlHintsLlamaCppService(endpoints.base_url)) {
            warnings.push_back({
                "backend_url_mismatch",
                "THOTH_INFERENCE_BACKEND=ollama but inference_base looks like llama.cpp "
                    "(set THOTH_INFERENCE_BACKEND=llama_cpp for Docker Compose)",
            });
        }
        if (urlHintsLlamaCppService(endpoints.embed_base_url)) {
            warnings.push_back({
                "embed_backend_url_mismatch",
                "THOTH_INFERENCE_BACKEND=ollama but embed_base looks like llama-embed-server "
                    "(set THOTH_INFERENCE_BACKEND=llama_cpp and keep THOTH_EMBED_BASE_URL)",
            });
        }
    } else if (backend_name == "llama_cpp") {
        if (urlHintsOllamaService(endpoints.base_url)) {
            warnings.push_back({
                "backend_url_mismatch",
                "THOTH_INFERENCE_BACKEND=llama_cpp but inference_base looks like Ollama "
                    "(use http://llama-server:8080 in Docker or http://127.0.0.1:8080 on host)",
            });
        }
        if (urlHintsOllamaService(endpoints.embed_base_url)) {
            warnings.push_back({
                "embed_backend_url_mismatch",
                "THOTH_INFERENCE_BACKEND=llama_cpp but embed_base looks like Ollama "
                    "(use http://llama-embed-server:8081 in Docker or http://127.0.0.1:8081 on host)",
            });
        }
    }

    return warnings;
}

void logInferenceBackendMisconfigWarnings(const Config* config) {
    const auto endpoints = config != nullptr ? resolveInferenceEndpoints(*config)
                                             : resolveInferenceEndpoints();

    std::string backend_error;
    const auto backend = tryResolveInferenceBackend(backend_error);
    const std::string backend_name =
        backend ? inferenceBackendName(*backend)
                : (backend_error.empty() ? "unknown" : "invalid");

    for (const auto& warning : detectInferenceBackendMisconfigs(backend_name, endpoints)) {
        std::cerr << "[Thoth] config_warning code=" << warning.code << ' '
                  << warning.message << '\n';
    }
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

EmbeddingProbeSnapshot probeEmbeddingBackend(const Config* config) {
    EmbeddingProbeSnapshot snapshot;
    snapshot.status = "unknown";
    snapshot.model = resolveEmbeddingModelName(config);

    if (testSuiteDevTierEnabled()) {
        snapshot.status = "skipped";
        cacheProbeSnapshot(snapshot);
        return snapshot;
    }

    const auto endpoints = config != nullptr ? resolveInferenceEndpoints(*config)
                                           : resolveInferenceEndpoints();
    snapshot.embed_base_url = endpoints.embed_base_url;

    const int expected_dim =
        EmbeddingEngine(EmbeddingEngine::Method::External, nullptr).getDimension();

    std::string backend_error;
    const auto backend = tryResolveInferenceBackend(backend_error);
    snapshot.backend =
        backend ? inferenceBackendName(*backend)
                : (backend_error.empty() ? "unknown" : "invalid");

    if (!backend) {
        snapshot.status = "failed";
        snapshot.error = backend_error;
        cacheProbeSnapshot(snapshot);
        return snapshot;
    }

    try {
        const auto client = createInferenceClient(endpoints, config);
        snapshot.backend = client->backendName();

        InferenceEmbedRequest request;
        request.model = snapshot.model;
        request.inputs = {kEmbedProbeInput};

        const auto result = client->embed(request);
        if (!result.ok || result.embeddings.empty()) {
            snapshot.status = "failed";
            snapshot.error = result.error.empty() ? "embed request failed" : result.error;
            cacheProbeSnapshot(snapshot);
            return snapshot;
        }

        snapshot.dimension = static_cast<int>(result.embeddings.front().size());
        if (snapshot.dimension != expected_dim) {
            snapshot.status = "failed";
            snapshot.error = "dimension mismatch: got " + std::to_string(snapshot.dimension)
                + " expected " + std::to_string(expected_dim);
            cacheProbeSnapshot(snapshot);
            return snapshot;
        }

        snapshot.status = "ok";
        cacheProbeSnapshot(snapshot);
        return snapshot;
    } catch (const std::exception& ex) {
        snapshot.status = "failed";
        snapshot.error = ex.what();
        cacheProbeSnapshot(snapshot);
        return snapshot;
    }
}

EmbeddingProbeSnapshot getLastEmbeddingProbeSnapshot() {
    std::lock_guard<std::mutex> lock(g_probeMutex);
    if (g_probeCached) {
        return g_lastProbeSnapshot;
    }
    EmbeddingProbeSnapshot unknown;
    unknown.status = "unknown";
    return unknown;
}

nlohmann::json embeddingProbeJson(const EmbeddingProbeSnapshot& snapshot) {
    nlohmann::json body{
        {"status", snapshot.status},
        {"backend", snapshot.backend},
        {"embed_base_url", snapshot.embed_base_url},
        {"model", snapshot.model},
        {"dimension", snapshot.dimension},
    };
    if (!snapshot.error.empty()) {
        body["error"] = snapshot.error;
    }
    return body;
}

void logEmbeddingStartupProbe(const Config* config) {
    if (testSuiteDevTierEnabled()) {
        if (runtimeConfigDiagnosticsEnabled(config)) {
            std::cerr << "[Thoth] embed_probe=skipped (THOTH_TEST_SUITE_DEV TfIdf tier)\n";
        }
        (void)probeEmbeddingBackend(config);
        return;
    }

    const EmbeddingProbeSnapshot snapshot = probeEmbeddingBackend(config);

    if (snapshot.status == "ok") {
        std::cerr << "[Thoth] embed_probe=ok backend=" << snapshot.backend
                  << " embed_base=" << snapshot.embed_base_url
                  << " model=" << snapshot.model << " dimension=" << snapshot.dimension
                  << '\n';
        return;
    }

    std::cerr << "[Thoth] embed_probe=failed backend=" << snapshot.backend
              << " embed_base=" << snapshot.embed_base_url
              << " model=" << snapshot.model;
    if (!snapshot.error.empty()) {
        std::cerr << " error=" << snapshot.error;
    }
    std::cerr << '\n';
}

} // namespace Thoth
