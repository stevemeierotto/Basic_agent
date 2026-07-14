/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan H inference request/response types (provider-neutral)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "llm_interface.h"

#include <string>
#include <vector>

namespace Thoth {

struct InferenceGenerateRequest {
    std::string model;
    std::string prompt;
    double temperature = 0.7;
    double top_p = 1.0;
    int max_tokens = 2048;
};

struct InferenceGenerateResult {
    std::string text;
    LlmTokenUsage token_usage;
    std::string raw_json;
    bool ok = false;
    std::string error;
};

struct InferenceEmbedRequest {
    std::string model;
    std::vector<std::string> inputs;
};

struct InferenceEmbedResult {
    std::vector<std::vector<float>> embeddings;
    bool ok = false;
    std::string error;
};

struct InferenceHealthResult {
    bool reachable = false;
    std::vector<std::string> available_models;
    std::string error;
};

} // namespace Thoth
