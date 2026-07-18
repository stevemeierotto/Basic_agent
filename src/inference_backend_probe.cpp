/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — G1d-CO A0 inference backend probe (client-identity provenance)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/inference_backend_probe.h"
#include "../include/ollama_snapshot.h"

#include <stdexcept>

namespace Thoth {

InferenceBackendSnapshot snapshotFromInferenceClient(
    const InferenceClient& client,
    const InferenceEndpointConfig& endpoints,
    const InferenceHealthResult& health) {
    InferenceBackendSnapshot snap;
    snap.backend_name = client.backendName();
    snap.base_url = endpoints.base_url;
    snap.embed_base_url = endpoints.embed_base_url;
    snap.reachable = health.reachable;
    snap.error = health.error;

    nlohmann::json diagnostics = nlohmann::json::object();
    if (!health.available_models.empty()) {
        diagnostics["available_models"] = health.available_models;
    }

    // Backend-specific enrichment when available (Ollama tags snapshot).
    if (snap.backend_name == "ollama" && snap.reachable) {
        OllamaFetchOptions options;
        options.base_url = endpoints.base_url;
        if (auto ollama = fetchOllamaSnapshot(options)) {
            diagnostics["ollama_version"] = ollama->version;
            nlohmann::json models = nlohmann::json::array();
            for (const auto& [name, digest] : ollama->models) {
                models.push_back({{"name", name}, {"digest", digest}});
            }
            diagnostics["ollama_models"] = std::move(models);
        }
    }

    snap.diagnostics = std::move(diagnostics);
    return snap;
}

InferenceBackendSnapshot probeInferenceBackend(const Config* config) {
    InferenceBackendSnapshot snap;
    try {
        const InferenceEndpointConfig endpoints = config ? resolveInferenceEndpoints(*config)
                                                         : resolveInferenceEndpoints();
        auto client = createInferenceClient(endpoints, config);
        const InferenceHealthResult health = client->health();
        snap = snapshotFromInferenceClient(*client, endpoints, health);
    } catch (const std::exception& ex) {
        snap.reachable = false;
        snap.error = ex.what();
        try {
            const InferenceEndpointConfig endpoints = config ? resolveInferenceEndpoints(*config)
                                                             : resolveInferenceEndpoints();
            snap.base_url = endpoints.base_url;
            snap.embed_base_url = endpoints.embed_base_url;
            snap.backend_name = inferenceBackendName(resolveInferenceBackend());
        } catch (...) {
            snap.backend_name = "unknown";
        }
    }
    return snap;
}

bool isInferenceBackendReachable(const Config* config) {
    return probeInferenceBackend(config).reachable;
}

} // namespace Thoth
