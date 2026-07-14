/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan H Ollama inference adapter
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/ollama_client.h"
#include "../include/inference_endpoint.h"
#include "../include/inference_http.h"

#include <../include/json.hpp>

#include <cstdlib>

using json = nlohmann::json;

namespace Thoth {

namespace {

long llmTimeoutSeconds() {
    long timeout_seconds = 600;
    if (const char* env = std::getenv("THOTH_LLM_TIMEOUT_SECONDS")) {
        try {
            const long parsed = std::stol(env);
            if (parsed > 0) {
                timeout_seconds = parsed;
            }
        } catch (...) {
        }
    }
    return timeout_seconds;
}

LlmTokenUsage parseOllamaTokenUsage(const std::string& raw_json) {
    LlmTokenUsage usage;
    try {
        const auto j = json::parse(raw_json);
        if (j.contains("prompt_eval_count") && j["prompt_eval_count"].is_number_integer()) {
            usage.prompt_tokens = j["prompt_eval_count"].get<std::int64_t>();
        }
        if (j.contains("eval_count") && j["eval_count"].is_number_integer()) {
            usage.completion_tokens = j["eval_count"].get<std::int64_t>();
        }
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
    } catch (...) {
    }
    return usage;
}

} // namespace

OllamaClient::OllamaClient(std::string base_url, std::string embed_base_url)
    : base_url_(std::move(base_url)), embed_base_url_(std::move(embed_base_url)) {}

InferenceGenerateResult OllamaClient::parseGenerateResponse(const std::string& raw_json) {
    InferenceGenerateResult result;
    result.raw_json = raw_json;
    try {
        const auto j = json::parse(raw_json);
        if (j.contains("response") && j["response"].is_string()) {
            result.text = j["response"].get<std::string>();
            result.token_usage = parseOllamaTokenUsage(raw_json);
            result.ok = true;
            return result;
        }
        if (j.contains("message") && j["message"].is_object() && j["message"].contains("content")) {
            result.text = j["message"]["content"].get<std::string>();
            result.token_usage = parseOllamaTokenUsage(raw_json);
            result.ok = true;
            return result;
        }
        if (j.contains("error") && j["error"].is_string()) {
            result.error = j["error"].get<std::string>();
            return result;
        }
    } catch (const std::exception& e) {
        result.error = std::string("JSON parse error: ") + e.what();
        return result;
    }
    result.error = "Malformed response from Ollama.";
    return result;
}

InferenceEmbedResult OllamaClient::parseEmbedResponse(const std::string& raw_json) {
    InferenceEmbedResult result;
    try {
        const auto j = json::parse(raw_json);
        if (j.contains("embeddings") && j["embeddings"].is_array()) {
            for (const auto& emb : j["embeddings"]) {
                result.embeddings.push_back(emb.get<std::vector<float>>());
            }
            result.ok = !result.embeddings.empty();
            if (!result.ok) {
                result.error = "Empty embeddings array";
            }
            return result;
        }
    } catch (const std::exception& e) {
        result.error = std::string("JSON parse error: ") + e.what();
        return result;
    }
    result.error = "Malformed embed response from Ollama.";
    return result;
}

InferenceHealthResult OllamaClient::parseTagsResponse(const std::string& raw_json) {
    InferenceHealthResult result;
    result.reachable = true;
    try {
        const auto j = json::parse(raw_json);
        if (j.contains("models") && j["models"].is_array()) {
            for (const auto& model : j["models"]) {
                if (model.contains("name") && model["name"].is_string()) {
                    const std::string name = model["name"].get<std::string>();
                    if (!name.empty()) {
                        result.available_models.push_back(name);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        result.reachable = false;
        result.error = std::string("JSON parse error: ") + e.what();
    }
    return result;
}

InferenceGenerateResult OllamaClient::generate(const InferenceGenerateRequest& request) {
    InferenceGenerateResult result;
    if (request.model.empty()) {
        result.error = "Model name is required";
        return result;
    }

    json payload;
    payload["model"] = request.model;
    payload["prompt"] = request.prompt;
    payload["stream"] = false;
    payload["options"] = {
        {"temperature", request.temperature},
        {"top_p", request.top_p},
        {"num_predict", request.max_tokens},
    };

    const std::string url = inferenceUrl(base_url_, "/api/generate");
    const auto http = inferenceHttpPost(url, payload.dump(), llmTimeoutSeconds());
    if (!http.ok) {
        result.error = http.error.empty() ? http.body : http.error;
        return result;
    }
    return parseGenerateResponse(http.body);
}

InferenceEmbedResult OllamaClient::embed(const InferenceEmbedRequest& request) {
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
    if (request.inputs.size() == 1) {
        payload["input"] = request.inputs.front();
    } else {
        payload["input"] = request.inputs;
    }

    const std::string url = inferenceUrl(embed_base_url_, "/api/embed");
    const auto http = inferenceHttpPost(url, payload.dump(), 300);
    if (!http.ok) {
        result.error = http.error.empty() ? http.body : http.error;
        return result;
    }
    return parseEmbedResponse(http.body);
}

InferenceHealthResult OllamaClient::health() {
    InferenceHealthResult result;
    const std::string url = inferenceUrl(base_url_, "/api/tags");
    const auto http = inferenceHttpGet(url, 10);
    if (!http.ok) {
        result.error = http.error.empty() ? http.body : http.error;
        return result;
    }
    return parseTagsResponse(http.body);
}

} // namespace Thoth
