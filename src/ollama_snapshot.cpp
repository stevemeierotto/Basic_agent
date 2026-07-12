/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E1 Ollama snapshot helper (Checkpoint B)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/ollama_snapshot.h"
#include "../include/inference_endpoint.h"

#include "json.hpp"

#include <curl/curl.h>
#include <algorithm>

namespace Thoth {

namespace {

size_t writeCallback(char* contents, size_t size, size_t nmemb, void* userp) {
    const size_t total = size * nmemb;
    static_cast<std::string*>(userp)->append(contents, total);
    return total;
}

std::optional<std::string> httpGet(const OllamaFetchOptions& options, const std::string& path) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return std::nullopt;
    }

    const std::string url = inferenceUrl(options.base_url, path);
    std::string body;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, options.timeout_ms);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, options.timeout_ms);

    const CURLcode result = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    if (result != CURLE_OK || body.empty()) {
        return std::nullopt;
    }
    return body;
}

std::optional<OllamaSnapshot> parseOllamaResponses(const std::string& versionBody,
                                                   const std::string& tagsBody) {
    OllamaSnapshot snapshot;
    try {
        const auto versionJson = nlohmann::json::parse(versionBody);
        if (versionJson.contains("version") && versionJson["version"].is_string()) {
            snapshot.version = versionJson["version"].get<std::string>();
        }
    } catch (...) {
        return std::nullopt;
    }

    try {
        const auto tagsJson = nlohmann::json::parse(tagsBody);
        if (!tagsJson.contains("models") || !tagsJson["models"].is_array()) {
            return snapshot;
        }
        for (const auto& model : tagsJson["models"]) {
            if (!model.is_object()) {
                continue;
            }
            std::string name;
            std::string digest;
            if (model.contains("name") && model["name"].is_string()) {
                name = model["name"].get<std::string>();
            }
            if (model.contains("digest") && model["digest"].is_string()) {
                digest = model["digest"].get<std::string>();
            }
            if (!name.empty()) {
                snapshot.models.emplace_back(name, digest);
            }
        }
        std::sort(snapshot.models.begin(), snapshot.models.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    } catch (...) {
        return std::nullopt;
    }

    return snapshot;
}

OllamaFetchOptions effectiveFetchOptions(const OllamaFetchOptions& options) {
    OllamaFetchOptions effective = options;
    if (effective.base_url.empty()) {
        effective.base_url = resolveInferenceEndpoints().base_url;
    }
    return effective;
}

} // namespace

bool isOllamaReachable(const OllamaFetchOptions& options) {
    const OllamaFetchOptions effective = effectiveFetchOptions(options);
    return httpGet(effective, "/api/tags").has_value();
}

std::optional<OllamaSnapshot> fetchOllamaSnapshot(const OllamaFetchOptions& options) {
    const OllamaFetchOptions effective = effectiveFetchOptions(options);
    const auto versionBody = httpGet(effective, "/api/version");
    const auto tagsBody = httpGet(effective, "/api/tags");
    if (!versionBody.has_value() || !tagsBody.has_value()) {
        return std::nullopt;
    }
    return parseOllamaResponses(*versionBody, *tagsBody);
}

} // namespace Thoth
