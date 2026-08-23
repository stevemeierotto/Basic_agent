#include "../include/llm_interface.h"
#include "../include/decision_trace.h"
#include "../include/inference_client.h"
#include "../include/inference_endpoint.h"
#include "../include/test_suite_dev.h"
#include "../include/robustness_mock_responses.h"
#include <../include/json.hpp>
#include <curl/curl.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <regex>
#include <sstream>
#include <cstdint>

using json = nlohmann::json;

namespace {

std::int64_t estimateTokensFromText(const std::string& text) {
    return static_cast<std::int64_t>((text.size() + 3) / 4);
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

} // namespace

void LLMInterface::resetSessionTokenUsage() {
    std::lock_guard<std::recursive_mutex> lock(llmMutex);
    last_call_usage_ = {};
    session_usage_ = {};
}

LlmTokenUsage LLMInterface::lastCallTokenUsage() const {
    std::lock_guard<std::recursive_mutex> lock(llmMutex);
    return last_call_usage_;
}

LlmTokenUsage LLMInterface::sessionTokenUsage() const {
    std::lock_guard<std::recursive_mutex> lock(llmMutex);
    return session_usage_;
}

void LLMInterface::recordTokenUsage(const LlmTokenUsage& usage) {
    std::lock_guard<std::recursive_mutex> lock(llmMutex);
    last_call_usage_ = usage;
    session_usage_.prompt_tokens += usage.prompt_tokens;
    session_usage_.completion_tokens += usage.completion_tokens;
    session_usage_.total_tokens += usage.total_tokens;
}

LlmTokenUsage LLMInterface::estimateTokenUsage(const std::string& prompt, const std::string& response) {
    LlmTokenUsage usage;
    usage.prompt_tokens = estimateTokensFromText(prompt);
    usage.completion_tokens = estimateTokensFromText(response);
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
    return usage;
}

LlmTokenUsage LLMInterface::parseOllamaTokenUsage(const std::string& rawJson) {
    LlmTokenUsage usage;
    try {
        auto j = json::parse(rawJson);
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

LlmTokenUsage LLMInterface::parseOpenAiTokenUsage(const std::string& rawJson) {
    LlmTokenUsage usage;
    try {
        auto j = json::parse(rawJson);
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
    } catch (...) {
    }
    return usage;
}

static std::string redactSensitiveText(const std::string& input) {
    std::string output = input;

    output = std::regex_replace(
        output,
        std::regex("Bearer\\s+[A-Za-z0-9._\\-]+", std::regex::icase),
        "Bearer [REDACTED]");

    output = std::regex_replace(
        output,
        std::regex("sk-[A-Za-z0-9]{8,}"),
        "sk-[REDACTED]");

    output = std::regex_replace(
        output,
        std::regex("(api[_-]?key\\s*[:=]\\s*)([^\\s\\\"']+)", std::regex::icase),
        "$1[REDACTED]");

    return output;
}

static std::string formatInferenceGenerateError(const Thoth::InferenceClient& client,
                                                const Config* config,
                                                const std::string& detail) {
    const auto endpoints = config ? Thoth::resolveInferenceEndpoints(*config)
                                  : Thoth::resolveInferenceEndpoints();
    const std::string redacted = redactSensitiveText(detail);
    const std::string& backend = client.backendName();

    const bool connection_like =
        detail.find("Couldn't connect") != std::string::npos
        || detail.find("connect to server") != std::string::npos
        || detail.find("Connection refused") != std::string::npos
        || detail.find("Failed to connect") != std::string::npos;

    if (backend == "llama_cpp") {
        if (connection_like) {
            return std::string("Assistant: [Error] llama_cpp: Unable to connect to llama-server at ")
                   + endpoints.base_url;
        }
        return std::string("Assistant: [Error] llama_cpp: ") + redacted;
    }

    if (backend == "ollama") {
        if (connection_like) {
            return std::string("Assistant: [Error] ollama: Unable to connect to inference server at ")
                   + endpoints.base_url;
        }
        return std::string("Assistant: [Error] ollama: ") + redacted;
    }

    if (connection_like) {
        return std::string(
                   "Assistant: [Error] Inference request failed: Unable to connect to inference server at ")
               + endpoints.base_url;
    }

    return std::string("Assistant: [Error] Inference request failed: ") + redacted;
}

void LLMInterface::ensureInferenceClient() {
    if (inference_client_) {
        return;
    }
    const auto endpoints = config ? Thoth::resolveInferenceEndpoints(*config)
                                  : Thoth::resolveInferenceEndpoints();
    inference_client_ = Thoth::createInferenceClient(endpoints, config);
}

LLMInterface::LLMInterface(LLMBackend b, Config* cfg)
    : backend(b),
      config(cfg),
      selectedModel("") {
    if (backend == LLMBackend::Ollama) {
        try {
            ensureInferenceClient();
        } catch (const std::exception& e) {
            DecisionTraceLogger traceLogger;
            DecisionTrace trace = traceLogger.startTrace("llm_init_error", 0);
            traceLogger.finishTrace(trace, false, std::string("Inference client init failed: ") + e.what());
            traceLogger.writeTrace(trace);
        }
    }
}

std::string LLMInterface::detectOllamaModel() {
    std::lock_guard<std::recursive_mutex> lock(llmMutex);
    try {
        ensureInferenceClient();
    } catch (...) {
        return "";
    }
    if (!inference_client_) {
        return "";
    }

    const auto health = inference_client_->health();
    if (!health.reachable || health.available_models.empty()) {
        return "";
    }
    return health.available_models.front();
}

std::string LLMInterface::resolveOllamaModel() {
    if (!selectedModel.empty()) return selectedModel;

    if (config && !config->llm_model.empty()) {
        selectedModel = config->llm_model;
        return selectedModel;
    }

    const char* envModel = std::getenv("OLLAMA_MODEL");
    if (envModel && *envModel) {
        selectedModel = envModel;
        return selectedModel;
    }

    selectedModel = detectOllamaModel();
    return selectedModel;
}

LLMInterface::~LLMInterface() = default;

void LLMInterface::setBackend(LLMBackend b) {
    std::lock_guard<std::recursive_mutex> lock(llmMutex);
    if (backend == b) return;
    backend = b;
    inference_client_.reset();

    if (backend == LLMBackend::Ollama) {
        try {
            ensureInferenceClient();
        } catch (const std::exception& e) {
            DecisionTraceLogger traceLogger;
            DecisionTrace trace = traceLogger.startTrace("llm_init_error", 0);
            traceLogger.finishTrace(trace, false, std::string("Inference client init failed: ") + e.what());
            traceLogger.writeTrace(trace);
        }
    }
}

std::string LLMInterface::query(const std::string& prompt) {
    return query(prompt, -1, {});
}

static bool envTruthy(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    const std::string flag(value);
    return flag == "1" || flag == "true" || flag == "TRUE" || flag == "yes";
}

std::string LLMInterface::query(const std::string& prompt, int num_predict_override) {
    return query(prompt, num_predict_override, {});
}

std::string LLMInterface::query(const std::string& prompt,
                                int num_predict_override,
                                const std::vector<std::string>& stop_sequences) {
    try {
        // Preserve legacy OpenAI string path (queryDetailed is llama/ollama structured only).
        if (backend == LLMBackend::OpenAI) {
            if (auto scripted = Thoth::RobustnessMockResponses::pop()) {
                const std::string response = *scripted;
                recordTokenUsage(estimateTokenUsage(prompt, response));
                return response;
            }
            if (envTruthy("THOTH_MOCK_LLM_UNAVAILABLE")) {
                const std::string response = "Assistant: [Error] LLM service unavailable (mock).";
                recordTokenUsage(estimateTokenUsage(prompt, response));
                return response;
            }
            if (Thoth::testSuiteDevTierEnabled()) {
                const std::string response = Thoth::mockTestSuiteLlmResponse(prompt);
                recordTokenUsage(estimateTokenUsage(prompt, response));
                return response;
            }
            return askOpenAI(prompt);
        }

        const Thoth::InferenceGenerateResult detailed =
            queryDetailed(prompt, num_predict_override, stop_sequences);
        if (!detailed.ok) {
            if (inference_client_) {
                return formatInferenceGenerateError(*inference_client_, config, detailed.error);
            }
            return std::string("Assistant: [Error] ") + detailed.error;
        }
        return detailed.text;
    } catch (const std::exception& e) {
        DecisionTraceLogger traceLogger;
        DecisionTrace trace = traceLogger.startTrace("llm_query_exception", prompt.size());
        traceLogger.finishTrace(trace, false, std::string("Query exception: ") + e.what());
        traceLogger.writeTrace(trace);
        return std::string("Assistant: [Error] LLM query failed: ") + e.what();
    } catch (...) {
        return "Assistant: [Error] LLM query failed with an unknown error.";
    }
}

Thoth::InferenceGenerateResult LLMInterface::queryDetailed(
    const std::string& prompt,
    int num_predict_override,
    const std::vector<std::string>& stop_sequences) {
    Thoth::InferenceGenerateResult result;
    try {
        if (auto fail = Thoth::RobustnessMockResponses::popFailure()) {
            result.ok = false;
            result.error = *fail;
            recordTokenUsage(estimateTokenUsage(prompt, ""));
            return result;
        }
        if (auto scripted = Thoth::RobustnessMockResponses::pop()) {
            result.ok = true;
            result.text = *scripted;
            recordTokenUsage(estimateTokenUsage(prompt, result.text));
            return result;
        }
        if (envTruthy("THOTH_MOCK_LLM_UNAVAILABLE")) {
            result.ok = false;
            result.error = "LLM service unavailable (mock).";
            recordTokenUsage(estimateTokenUsage(prompt, ""));
            return result;
        }
        if (Thoth::testSuiteDevTierEnabled()) {
            result.ok = true;
            result.text = Thoth::mockTestSuiteLlmResponse(prompt);
            recordTokenUsage(estimateTokenUsage(prompt, result.text));
            return result;
        }
        if (backend != LLMBackend::Ollama) {
            result.ok = false;
            result.error = "OpenAI backend not supported for queryDetailed";
            return result;
        }

        std::lock_guard<std::recursive_mutex> lock(llmMutex);
        try {
            ensureInferenceClient();
        } catch (const std::exception& e) {
            result.error = std::string("Inference client unavailable: ") + e.what();
            return result;
        }
        if (!inference_client_) {
            result.error = "Inference client not initialized.";
            return result;
        }

        double temperature = config ? config->temperature : 0.7;
        double topP = config ? config->top_p : 1.0;
        int maxTokens = config ? config->max_tokens : 2048;
        if (num_predict_override >= 0) {
            maxTokens = num_predict_override;
        }
        std::string model = resolveOllamaModel();
        if (model.empty()) {
            result.error =
                "No inference model configured. Set llm_model, OLLAMA_MODEL, or ensure the "
                "inference service is reachable.";
            return result;
        }

        Thoth::InferenceGenerateRequest request;
        request.model = model;
        request.prompt = prompt;
        request.temperature = temperature;
        request.top_p = topP;
        request.max_tokens = maxTokens;
        request.stop_sequences = stop_sequences;

        result = inference_client_->generate(request);
        if (!result.ok && result.error.find("not found") != std::string::npos) {
            const std::string detected = detectOllamaModel();
            if (!detected.empty() && detected != model) {
                selectedModel = detected;
                request.model = selectedModel;
                result = inference_client_->generate(request);
            }
        }

        if (!result.ok) {
            DecisionTraceLogger traceLogger;
            DecisionTrace trace = traceLogger.startTrace("inference_error", prompt.size());
            traceLogger.finishTrace(trace, false, std::string("Inference failure: ") + result.error);
            traceLogger.writeTrace(trace);
            return result;
        }

        LlmTokenUsage usage = result.token_usage;
        if (!result.raw_json.empty() && usage.total_tokens <= 0) {
            usage = parseOllamaTokenUsage(result.raw_json);
        }
        if (usage.total_tokens <= 0) {
            usage = parseOpenAiTokenUsage(result.raw_json);
        }
        if (usage.total_tokens <= 0) {
            usage = estimateTokenUsage(prompt, result.text);
        }
        result.token_usage = usage;
        recordTokenUsage(usage);
        return result;
    } catch (const std::exception& e) {
        result.ok = false;
        result.error = std::string("Inference request failed: ") + e.what();
        return result;
    } catch (...) {
        result.ok = false;
        result.error = "Inference request failed with an unknown error.";
        return result;
    }
}

namespace {

std::string flattenChatRequestForTelemetry(const Thoth::InferenceChatRequest& request) {
    std::ostringstream oss;
    for (const auto& message : request.messages) {
        oss << message.role << ": " << message.content << '\n';
    }
    return oss.str();
}

} // namespace

Thoth::InferenceGenerateResult LLMInterface::queryDetailedChat(
    const Thoth::InferenceChatRequest& chat_request,
    int num_predict_override,
    const std::vector<std::string>& stop_sequences) {
    Thoth::InferenceGenerateResult result;
    const std::string telemetryPrompt = flattenChatRequestForTelemetry(chat_request);
    try {
        if (auto fail = Thoth::RobustnessMockResponses::popFailure()) {
            result.ok = false;
            result.error = *fail;
            recordTokenUsage(estimateTokenUsage(telemetryPrompt, ""));
            return result;
        }
        if (auto scripted = Thoth::RobustnessMockResponses::pop()) {
            result.ok = true;
            result.text = *scripted;
            recordTokenUsage(estimateTokenUsage(telemetryPrompt, result.text));
            return result;
        }
        if (envTruthy("THOTH_MOCK_LLM_UNAVAILABLE")) {
            result.ok = false;
            result.error = "LLM service unavailable (mock).";
            recordTokenUsage(estimateTokenUsage(telemetryPrompt, ""));
            return result;
        }
        if (Thoth::testSuiteDevTierEnabled()) {
            result.ok = true;
            result.text = Thoth::mockTestSuiteLlmResponse(telemetryPrompt);
            recordTokenUsage(estimateTokenUsage(telemetryPrompt, result.text));
            return result;
        }
        if (backend != LLMBackend::Ollama) {
            result.ok = false;
            result.error = "OpenAI backend not supported for queryDetailedChat";
            return result;
        }

        std::lock_guard<std::recursive_mutex> lock(llmMutex);
        try {
            ensureInferenceClient();
        } catch (const std::exception& e) {
            result.error = std::string("Inference client unavailable: ") + e.what();
            return result;
        }
        if (!inference_client_) {
            result.error = "Inference client not initialized.";
            return result;
        }

        double temperature = config ? config->temperature : 0.7;
        double topP = config ? config->top_p : 1.0;
        int maxTokens = config ? config->max_tokens : 2048;
        if (num_predict_override >= 0) {
            maxTokens = num_predict_override;
        }
        std::string model = resolveOllamaModel();
        if (model.empty()) {
            result.error =
                "No inference model configured. Set llm_model, OLLAMA_MODEL, or ensure the "
                "inference service is reachable.";
            return result;
        }

        Thoth::InferenceChatRequest request = chat_request;
        request.model = model;
        request.temperature = temperature;
        request.top_p = topP;
        request.max_tokens = maxTokens;
        request.stop_sequences = stop_sequences;

        result = inference_client_->generateChat(request);
        if (!result.ok && result.error.find("not found") != std::string::npos) {
            const std::string detected = detectOllamaModel();
            if (!detected.empty() && detected != model) {
                selectedModel = detected;
                request.model = selectedModel;
                result = inference_client_->generateChat(request);
            }
        }

        if (!result.ok) {
            DecisionTraceLogger traceLogger;
            DecisionTrace trace = traceLogger.startTrace("inference_error", telemetryPrompt.size());
            traceLogger.finishTrace(trace, false, std::string("Inference failure: ") + result.error);
            traceLogger.writeTrace(trace);
            return result;
        }

        LlmTokenUsage usage = result.token_usage;
        if (!result.raw_json.empty() && usage.total_tokens <= 0) {
            usage = parseOllamaTokenUsage(result.raw_json);
        }
        if (usage.total_tokens <= 0) {
            usage = parseOpenAiTokenUsage(result.raw_json);
        }
        if (usage.total_tokens <= 0) {
            usage = estimateTokenUsage(telemetryPrompt, result.text);
        }
        result.token_usage = usage;
        recordTokenUsage(usage);
        return result;
    } catch (const std::exception& e) {
        result.ok = false;
        result.error = std::string("Inference request failed: ") + e.what();
        return result;
    } catch (...) {
        result.ok = false;
        result.error = "Inference request failed with an unknown error.";
        return result;
    }
}

std::string LLMInterface::formatProviderError(const std::string& detail) {
    try {
        ensureInferenceClient();
    } catch (...) {
    }
    if (inference_client_) {
        return formatInferenceGenerateError(*inference_client_, config, detail);
    }
    return std::string("Assistant: [Error] ") + redactSensitiveText(detail);
}

std::string LLMInterface::askOllama(const std::string& prompt) {
    return askOllama(prompt, -1, {});
}

std::string LLMInterface::askOllama(const std::string& prompt, int num_predict_override) {
    return askOllama(prompt, num_predict_override, {});
}

std::string LLMInterface::askOllama(const std::string& prompt,
                                    int num_predict_override,
                                    const std::vector<std::string>& stop_sequences) {
    return query(prompt, num_predict_override, stop_sequences);
}

std::string LLMInterface::askOpenAI(const std::string& prompt) {
    CURL* curl_local = curl_easy_init();
    std::string readBuffer;

    if (!curl_local) {
        return "Assistant: [Error] Failed to initialize CURL for OpenAI.";
    }

    struct curl_slist* headers_local = NULL;
    try {
        json data = {
            {"model", "gpt-3.5-turbo"}, 
            {"messages", {
                {{"role", "system"}, {"content", "You are a helpful C++ coding assistant."}},
                {{"role", "user"}, {"content", prompt}}
            }},
            {"temperature", 0.7}
        };

        std::string payload = data.dump();
        headers_local = curl_slist_append(headers_local, "Content-Type: application/json");

        const char* apiKey = std::getenv("OPENAI_API_KEY");
        if (!apiKey) {
            throw std::runtime_error("OPENAI_API_KEY not set in environment");
        }
        std::string authHeader = std::string("Authorization: Bearer ") + apiKey;
        headers_local = curl_slist_append(headers_local, authHeader.c_str());

        curl_easy_setopt(curl_local, CURLOPT_URL, "https://api.openai.com/v1/chat/completions");
        curl_easy_setopt(curl_local, CURLOPT_POST, 1L);
        curl_easy_setopt(curl_local, CURLOPT_HTTPHEADER, headers_local);
        curl_easy_setopt(curl_local, CURLOPT_POSTFIELDS, payload.c_str());
        curl_easy_setopt(curl_local, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl_local, CURLOPT_WRITEDATA, &readBuffer);

        CURLcode res = curl_easy_perform(curl_local);

        if (res != CURLE_OK) {
            throw std::runtime_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
        }

        json json_response = json::parse(readBuffer);
        if (!json_response.contains("choices") || json_response["choices"].empty())
            throw std::runtime_error("OpenAI API returned no choices");

        auto message = json_response["choices"][0]["message"];
        std::string content = message["content"].get<std::string>();

        LlmTokenUsage usage = parseOpenAiTokenUsage(readBuffer);
        if (usage.total_tokens <= 0) {
            usage = estimateTokenUsage(prompt, content);
        }
        recordTokenUsage(usage);
        
        curl_easy_cleanup(curl_local);
        curl_slist_free_all(headers_local);
        return content;

    } catch (const std::exception& e) {
        if (curl_local) curl_easy_cleanup(curl_local);
        if (headers_local) curl_slist_free_all(headers_local);
        
        DecisionTraceLogger traceLogger;
        DecisionTrace trace = traceLogger.startTrace("openai_error", prompt.size());
        traceLogger.finishTrace(trace, false, std::string("OpenAI failure: ") + e.what());
        traceLogger.writeTrace(trace);
        
        return std::string("Assistant: [Error] OpenAI request failed: ") + redactSensitiveText(e.what());
    }
}
