/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — configurable inference service base URLs (Plan A)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include <string>

class Config;

namespace Thoth {

struct InferenceEndpointConfig {
    std::string base_url;
    std::string embed_base_url;
};

/** Resolve from environment variables and defaults only. */
InferenceEndpointConfig resolveInferenceEndpoints();

/** Resolve from environment, then optional config.json fields (lowest priority). */
InferenceEndpointConfig resolveInferenceEndpoints(const Config& config);

/** Join a normalized base origin with an API path (e.g. "/api/generate"). */
std::string inferenceUrl(const std::string& base_url, const std::string& path);

} // namespace Thoth
