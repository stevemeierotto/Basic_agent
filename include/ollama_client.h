/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan H Ollama inference adapter
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "inference_client.h"

namespace Thoth {

class OllamaClient final : public InferenceClient {
public:
    OllamaClient(std::string base_url, std::string embed_base_url);

    InferenceGenerateResult generate(const InferenceGenerateRequest& request) override;
    InferenceEmbedResult embed(const InferenceEmbedRequest& request) override;
    InferenceHealthResult health() override;
    std::string backendName() const override { return "ollama"; }

    static InferenceGenerateResult parseGenerateResponse(const std::string& raw_json);
    static InferenceEmbedResult parseEmbedResponse(const std::string& raw_json);
    static InferenceHealthResult parseTagsResponse(const std::string& raw_json);

private:
    std::string base_url_;
    std::string embed_base_url_;
};

} // namespace Thoth
