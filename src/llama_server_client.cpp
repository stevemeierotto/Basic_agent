/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan H llama-server inference adapter
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/llama_server_client.h"
#include "../include/inference_endpoint.h"
#include "../include/inference_http.h"
#include "../include/llm_timeout_policy.h"

#include <../include/json.hpp>

#include <algorithm>
#include <cstdlib>

using json = nlohmann::json;

namespace Thoth {

namespace {

LlmTokenUsage parseOpenAiCompletionUsage(const json& j) {
    LlmTokenUsage usage;
    if (j.contains("usage") && j["usage"].is_object()) {
        const auto& u = j["usage"];
        if (u.contains("prompt_tokens") && u["prompt_tokens"].is_number_integer()) {
            usage.prompt_tokens = u["prompt_tokens"].get<std::int64_t>();
        }
        if (u.contains("completion_tokens") && u["completion_tokens"].is_number_integer()) {
            usage.completion_tokens = u["completion_tokens"].get<std::int64_t>();
        }
        if (u.contains("total_tokens") && u["total_tokens"].is_number_integer()) {
            usage.total_tokens = u["total_tokens"].get<std::int64_t>();
        } else {
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
        }
    }
    return usage;
}

} // namespace

LlamaServerClient::LlamaServerClient(std::string base_url, std::string embed_base_url)
    : base_url_(std::move(base_url)), embed_base_url_(std::move(embed_base_url)) {}

InferenceGenerateResult LlamaServerClient::parseCompletionResponse(const std::string& raw_json) {
    InferenceGenerateResult result;
    result.raw_json = raw_json;
    try {
        const auto j = json::parse(raw_json);
        if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
            const auto& choice = j["choices"][0];
            if (choice.contains("text") && choice["text"].is_string()) {
                result.text = choice["text"].get<std::string>();
            } else if (choice.contains("message") && choice["message"].is_object()
                       && choice["message"].contains("content")
                       && choice["message"]["content"].is_string()) {
                result.text = choice["message"]["content"].get<std::string>();
            }
            if (choice.contains("finish_reason") && choice["finish_reason"].is_string()) {
                result.finish_reason = choice["finish_reason"].get<std::string>();
            }
            result.token_usage = parseOpenAiCompletionUsage(j);
            // Plan N N2: empty text with a valid choices payload is provider-ok (soft-empty).
            result.ok = true;
            return result;
        }
        if (j.contains("error") && j["error"].is_object() && j["error"].contains("message")) {
            result.error = j["error"]["message"].get<std::string>();
            return result;
        }
    } catch (const std::exception& e) {
        result.error = std::string("JSON parse error: ") + e.what();
        return result;
    }
    result.error = "Malformed completion response from llama-server.";
    return result;
}

InferenceEmbedResult LlamaServerClient::parseEmbeddingsResponse(const std::string& raw_json) {
    InferenceEmbedResult result;
    try {
        const auto j = json::parse(raw_json);
        if (j.contains("data") && j["data"].is_array()) {
            std::vector<std::pair<int, std::vector<float>>> indexed;
            indexed.reserve(j["data"].size());
            for (const auto& item : j["data"]) {
                if (!item.contains("embedding") || !item["embedding"].is_array()) {
                    continue;
                }
                const int index = item.value("index", static_cast<int>(indexed.size()));
                indexed.emplace_back(index, item["embedding"].get<std::vector<float>>());
            }
            std::sort(indexed.begin(), indexed.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& [idx, vec] : indexed) {
                (void)idx;
                result.embeddings.push_back(vec);
            }
            result.ok = !result.embeddings.empty();
            if (!result.ok) {
                result.error = "Empty embeddings data";
            }
            return result;
        }
    } catch (const std::exception& e) {
        result.error = std::string("JSON parse error: ") + e.what();
        return result;
    }
    result.error = "Malformed embeddings response from llama-server.";
    return result;
}

InferenceGenerateResult LlamaServerClient::generate(const InferenceGenerateRequest& request) {
    InferenceGenerateResult result;
    if (request.model.empty()) {
        result.error = "Model name is required";
        return result;
    }

    const std::string url = inferenceUrl(base_url_, "/v1/completions");
    const auto http = inferenceHttpPost(url, serializeGeneratePayload(request), LlmTimeoutPolicy::timeoutSeconds());
    if (!http.ok) {
        result.error = http.error.empty() ? http.body : http.error;
        return result;
    }
    return parseCompletionResponse(http.body);
}

InferenceGenerateResult LlamaServerClient::generateChat(const InferenceChatRequest& request) {
    InferenceGenerateResult result;
    if (request.model.empty()) {
        result.error = "Model name is required";
        return result;
    }
    if (request.messages.empty()) {
        result.error = "At least one chat message is required";
        return result;
    }

    const std::string url = inferenceUrl(base_url_, "/v1/chat/completions");
    const auto http = inferenceHttpPost(url, serializeChatPayload(request), LlmTimeoutPolicy::timeoutSeconds());
    if (!http.ok) {
        result.error = http.error.empty() ? http.body : http.error;
        return result;
    }
    return parseCompletionResponse(http.body);
}

std::string LlamaServerClient::serializeGeneratePayload(const InferenceGenerateRequest& request) {
    json payload;
    payload["model"] = request.model;
    payload["prompt"] = request.prompt;
    payload["max_tokens"] = request.max_tokens;
    payload["temperature"] = request.temperature;
    payload["top_p"] = request.top_p;
    payload["stream"] = false;
    if (!request.stop_sequences.empty()) {
        payload["stop"] = request.stop_sequences;
    }
    return payload.dump();
}

std::string LlamaServerClient::serializeChatPayload(const InferenceChatRequest& request) {
    json payload;
    payload["model"] = request.model;
    json messages = json::array();
    for (const auto& message : request.messages) {
        messages.push_back({{"role", message.role}, {"content", message.content}});
    }
    payload["messages"] = std::move(messages);
    payload["max_tokens"] = request.max_tokens;
    payload["temperature"] = request.temperature;
    payload["top_p"] = request.top_p;
    payload["stream"] = false;
    if (!request.stop_sequences.empty()) {
        payload["stop"] = request.stop_sequences;
    }
    return payload.dump();
}

InferenceEmbedResult LlamaServerClient::embed(const InferenceEmbedRequest& request) {
    InferenceEmbedResult result;
    if (request.inputs.empty()) {
        result.error = "No embed inputs provided";
        return result;
    }
    if (request.model.empty()) {
        result.error = "Model name is required";
        return result;
    }

    json payload;
    payload["model"] = request.model;
    payload["encoding_format"] = "float";
    if (request.inputs.size() == 1) {
        payload["input"] = request.inputs.front();
    } else {
        payload["input"] = request.inputs;
    }

    const std::string url = inferenceUrl(embed_base_url_, "/v1/embeddings");
    const auto http = inferenceHttpPost(url, payload.dump(), 300);
    if (!http.ok) {
        result.error = http.error.empty() ? http.body : http.error;
        return result;
    }
    return parseEmbeddingsResponse(http.body);
}

InferenceHealthResult LlamaServerClient::health() {
    InferenceHealthResult result;
    const std::string url = inferenceUrl(base_url_, "/health");
    const auto http = inferenceHttpGet(url, 10);
    if (!http.ok) {
        result.error = http.error.empty() ? http.body : http.error;
        return result;
    }
    result.reachable = true;
    return result;
}

} // namespace Thoth
