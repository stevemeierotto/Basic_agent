/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan H mock inference client
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/mock_inference_client.h"

namespace Thoth {

InferenceGenerateResult MockInferenceClient::generate(const InferenceGenerateRequest& request) {
    if (on_generate) {
        return on_generate(request);
    }
    InferenceGenerateResult result;
    result.ok = true;
    result.text = "mock-response";
    return result;
}

InferenceEmbedResult MockInferenceClient::embed(const InferenceEmbedRequest& request) {
    if (on_embed) {
        return on_embed(request);
    }
    InferenceEmbedResult result;
    result.ok = true;
    result.embeddings.assign(request.inputs.size(), std::vector<float>(768, 0.1f));
    return result;
}

InferenceHealthResult MockInferenceClient::health() {
    if (on_health) {
        return on_health();
    }
    InferenceHealthResult result;
    result.reachable = true;
    return result;
}

} // namespace Thoth
