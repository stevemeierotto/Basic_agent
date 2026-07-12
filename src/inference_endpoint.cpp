/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — configurable inference service base URLs (Plan A)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/inference_endpoint.h"
#include "../include/config.h"

#include <cstdlib>
#include <string>

namespace Thoth {

namespace {

constexpr const char* kDefaultBaseUrl = "http://127.0.0.1:11434";

std::string envOrEmpty(const char* key) {
    const char* value = std::getenv(key);
    return (value && *value) ? std::string(value) : std::string();
}

std::string normalizeOrigin(std::string url) {
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    return url;
}

std::string ensureScheme(std::string url) {
    if (url.empty()) {
        return url;
    }
    if (url.find("://") == std::string::npos) {
        return "http://" + url;
    }
    return url;
}

std::string normalizePath(std::string path) {
    if (path.empty()) {
        return "/";
    }
    if (path.front() != '/') {
        path.insert(path.begin(), '/');
    }
    return path;
}

std::string resolveBaseFromEnvAndConfig(const Config* config) {
    const std::string envInference = envOrEmpty("THOTH_INFERENCE_BASE_URL");
    if (!envInference.empty()) {
        return normalizeOrigin(ensureScheme(envInference));
    }

    const std::string ollamaHost = envOrEmpty("OLLAMA_HOST");
    if (!ollamaHost.empty()) {
        return normalizeOrigin(ensureScheme(ollamaHost));
    }

    if (config && !config->inference_base_url.empty()) {
        return normalizeOrigin(ensureScheme(config->inference_base_url));
    }

    return kDefaultBaseUrl;
}

std::string resolveEmbedFromEnvAndConfig(const std::string& llmBase, const Config* config) {
    const std::string envEmbed = envOrEmpty("THOTH_EMBED_BASE_URL");
    if (!envEmbed.empty()) {
        return normalizeOrigin(ensureScheme(envEmbed));
    }

    if (config && !config->embed_base_url.empty()) {
        return normalizeOrigin(ensureScheme(config->embed_base_url));
    }

    return llmBase;
}

InferenceEndpointConfig buildEndpoints(const Config* config) {
    InferenceEndpointConfig endpoints;
    endpoints.base_url = resolveBaseFromEnvAndConfig(config);
    endpoints.embed_base_url = resolveEmbedFromEnvAndConfig(endpoints.base_url, config);
    return endpoints;
}

} // namespace

InferenceEndpointConfig resolveInferenceEndpoints() {
    return buildEndpoints(nullptr);
}

InferenceEndpointConfig resolveInferenceEndpoints(const Config& config) {
    return buildEndpoints(&config);
}

std::string inferenceUrl(const std::string& base_url, const std::string& path) {
    return normalizeOrigin(base_url) + normalizePath(path);
}

} // namespace Thoth
