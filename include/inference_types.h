/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan H inference request/response types (provider-neutral)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Thoth {

/** C6 / Plan H: token counts from an inference call. */
struct LlmTokenUsage {
    std::int64_t prompt_tokens = 0;
    std::int64_t completion_tokens = 0;
    std::int64_t total_tokens = 0;
};

struct InferenceGenerateRequest {
    std::string model;
    std::string prompt;
    double temperature = 0.7;
    double top_p = 1.0;
    int max_tokens = 2048;
    /** Plan M G3 — optional stop sequences; empty means omit from provider payload. */
    std::vector<std::string> stop_sequences;
};

struct InferenceGenerateResult {
    std::string text;
    LlmTokenUsage token_usage;
    std::string raw_json;
    bool ok = false;
    std::string error;
    /** Plan N N2 — provider finish reason when present (e.g. stop, length). */
    std::string finish_reason;
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
