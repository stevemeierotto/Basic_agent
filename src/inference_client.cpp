/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan H inference client factory
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/inference_client.h"
#include "../include/llama_server_client.h"
#include "../include/ollama_client.h"

#include <cstdlib>
#include <stdexcept>

namespace Thoth {

namespace {

std::string envOrEmpty(const char* key) {
    const char* value = std::getenv(key);
    return (value && *value) ? std::string(value) : std::string();
}

std::string normalizeBackendToken(std::string value) {
    for (char& c : value) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return value;
}

} // namespace

std::optional<InferenceBackend> tryResolveInferenceBackend(std::string& error_out) {
    error_out.clear();
    const std::string backend = normalizeBackendToken(envOrEmpty("THOTH_INFERENCE_BACKEND"));
    if (backend.empty() || backend == "ollama") {
        return InferenceBackend::Ollama;
    }
    if (backend == "llama_cpp") {
        return InferenceBackend::LlamaCpp;
    }
    error_out = "Invalid THOTH_INFERENCE_BACKEND='" + backend
                + "'. Expected 'ollama' or 'llama_cpp'.";
    return std::nullopt;
}

InferenceBackend resolveInferenceBackend() {
    std::string error;
    const auto backend = tryResolveInferenceBackend(error);
    if (!backend) {
        throw std::invalid_argument(error);
    }
    return *backend;
}

std::string inferenceBackendName(InferenceBackend backend) {
    switch (backend) {
        case InferenceBackend::Ollama:
            return "ollama";
        case InferenceBackend::LlamaCpp:
            return "llama_cpp";
    }
    return "unknown";
}

std::unique_ptr<InferenceClient> createInferenceClient(
    const InferenceEndpointConfig& endpoints,
    const Config* /*config*/) {
    std::string error;
    const auto backend = tryResolveInferenceBackend(error);
    if (!backend) {
        throw std::invalid_argument(error);
    }

    switch (*backend) {
        case InferenceBackend::Ollama:
            return std::make_unique<OllamaClient>(endpoints.base_url, endpoints.embed_base_url);
        case InferenceBackend::LlamaCpp:
            return std::make_unique<LlamaServerClient>(endpoints.base_url, endpoints.embed_base_url);
    }

    throw std::invalid_argument("Unsupported inference backend");
}

} // namespace Thoth
