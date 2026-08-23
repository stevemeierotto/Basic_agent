/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan H InferenceClient abstraction
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "inference_endpoint.h"
#include "inference_types.h"

#include <memory>
#include <optional>
#include <string>

class Config;

namespace Thoth {

enum class InferenceBackend {
    Ollama,
    LlamaCpp
};

class InferenceClient {
public:
    virtual ~InferenceClient() = default;

    virtual InferenceGenerateResult generate(const InferenceGenerateRequest& request) = 0;
    /** Phase A — default unsupported; llama_cpp overrides for /v1/chat/completions. */
    virtual InferenceGenerateResult generateChat(const InferenceChatRequest& request) {
        InferenceGenerateResult result;
        result.error = "Chat completions not supported for this inference backend";
        return result;
    }
    virtual InferenceEmbedResult embed(const InferenceEmbedRequest& request) = 0;
    virtual InferenceHealthResult health() = 0;
    virtual std::string backendName() const = 0;
};

/** Returns nullopt when THOTH_INFERENCE_BACKEND is invalid; error_out set. */
std::optional<InferenceBackend> tryResolveInferenceBackend(std::string& error_out);

InferenceBackend resolveInferenceBackend();

std::string inferenceBackendName(InferenceBackend backend);

std::unique_ptr<InferenceClient> createInferenceClient(
    const InferenceEndpointConfig& endpoints,
    const Config* config = nullptr);

} // namespace Thoth
