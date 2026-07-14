/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan H mock inference client for unit tests
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "inference_client.h"

#include <functional>

namespace Thoth {

class MockInferenceClient final : public InferenceClient {
public:
    std::function<InferenceGenerateResult(const InferenceGenerateRequest&)> on_generate;
    std::function<InferenceEmbedResult(const InferenceEmbedRequest&)> on_embed;
    std::function<InferenceHealthResult()> on_health;
    std::string backend_name = "mock";

    InferenceGenerateResult generate(const InferenceGenerateRequest& request) override;
    InferenceEmbedResult embed(const InferenceEmbedRequest& request) override;
    InferenceHealthResult health() override;
    std::string backendName() const override { return backend_name; }
};

} // namespace Thoth
