/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan H llama-server (llama.cpp) inference adapter
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "inference_client.h"

namespace Thoth {

class LlamaServerClient final : public InferenceClient {
public:
    LlamaServerClient(std::string base_url, std::string embed_base_url);

    InferenceGenerateResult generate(const InferenceGenerateRequest& request) override;
    InferenceGenerateResult generateChat(const InferenceChatRequest& request) override;
    InferenceEmbedResult embed(const InferenceEmbedRequest& request) override;
    InferenceHealthResult health() override;
    std::string backendName() const override { return "llama_cpp"; }

    static InferenceGenerateResult parseCompletionResponse(const std::string& raw_json);
    static InferenceEmbedResult parseEmbeddingsResponse(const std::string& raw_json);

    /** Plan M G3 — serialize generate payload (for tests + generate()). */
    static std::string serializeGeneratePayload(const InferenceGenerateRequest& request);
    /** Phase A — serialize chat payload (for tests + generateChat()). */
    static std::string serializeChatPayload(const InferenceChatRequest& request);
    static std::string serializeEmbedPayload(const InferenceEmbedRequest& request);

private:
    std::string base_url_;
    std::string embed_base_url_;
};

} // namespace Thoth
