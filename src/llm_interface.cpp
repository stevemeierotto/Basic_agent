#include "../include/llm_interface.h"
#include "../include/decision_trace.h"
#include <../include/json.hpp>
#include <curl/curl.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <regex>

using json = nlohmann::json;

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


// Constructor
LLMInterface::LLMInterface(LLMBackend b, Config* cfg)
    : backend(b),
      config(cfg),
      curl(nullptr),
      headers(nullptr),
    selectedModel("")
{
    if (backend == LLMBackend::Ollama) {
        curl = curl_easy_init();
        if (!curl) {
            DecisionTraceLogger traceLogger;
            DecisionTrace trace = traceLogger.startTrace("llm_init_error", 0);
            traceLogger.finishTrace(trace, false, "Failed to initialize CURL handle for Ollama");
            traceLogger.writeTrace(trace);
            // We don't throw here to allow graceful degradation, but we log.
        } else {
            headers = curl_slist_append(nullptr, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        }
    }
}

std::string LLMInterface::detectOllamaModel() {
    std::lock_guard<std::recursive_mutex> lock(llmMutex);
    if (!curl) return "";

    try {
        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:11434/api/tags");
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(curl, CURLOPT_POST, 0L);

        std::string readBuffer;
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            return "";
        }

        auto j = json::parse(readBuffer);
        if (j.contains("models") && j["models"].is_array()) {
            for (const auto& model : j["models"]) {
                if (model.contains("name") && model["name"].is_string()) {
                    std::string name = model["name"].get<std::string>();
                    if (!name.empty()) return name;
                }
            }
        }
    } catch (...) {
        return "";
    }

    return "";
}

std::string LLMInterface::resolveOllamaModel() {
    if (!selectedModel.empty()) return selectedModel;

    // 1. Check config first
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

LLMInterface::~LLMInterface() {
    std::lock_guard<std::recursive_mutex> lock(llmMutex);
    if (headers) curl_slist_free_all(headers);
    if (curl) curl_easy_cleanup(curl);
    headers = nullptr;
    curl = nullptr;
}

void LLMInterface::setBackend(LLMBackend b) {
    std::lock_guard<std::recursive_mutex> lock(llmMutex);
    if (backend == b) return; // no-op if same backend
    backend = b;

    // Clean up old handles if switching to a different backend
    if (curl) curl_easy_cleanup(curl);
    if (headers) curl_slist_free_all(headers);

    curl = nullptr;
    headers = nullptr;

    if (backend == LLMBackend::Ollama) {
        curl = curl_easy_init();
        if (curl) {
            headers = curl_slist_append(nullptr, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        }
    }
}

std::string LLMInterface::query(const std::string& prompt) {
    try {
        if (backend == LLMBackend::Ollama) {
            return askOllama(prompt);
        } else {
            return askOpenAI(prompt);
        }
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


std::string LLMInterface::askOllama(const std::string& prompt) {
    std::lock_guard<std::recursive_mutex> lock(llmMutex);
    if (!curl) return "Assistant: [Error] Ollama CURL handle not initialized.";

    try {
        // Pull dynamic parameters from Config
        double temperature = config ? config->temperature : 0.7;
        double topP        = config ? config->top_p : 1.0;
        int maxTokens      = config ? config->max_tokens : 2048;
        std::string model = resolveOllamaModel();
        
        if (model.empty()) {
            return "Assistant: [Error] No Ollama model detected. Pull one first (e.g. 'ollama pull qwen2.5:3b') or set OLLAMA_MODEL.";
        }

        auto callGenerate = [&](const std::string& modelName, std::string& rawOut, std::string& errorOut) -> std::string {
            json payload;
            payload["model"] = modelName;
            payload["prompt"] = prompt;
            payload["stream"] = false;
            payload["temperature"] = temperature;
            payload["top_p"] = topP;
            payload["max_tokens"] = maxTokens;
            std::string jsonStr = payload.dump();

            curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:11434/api/generate");
            curl_easy_setopt(curl, CURLOPT_HTTPGET, 0L);
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, jsonStr.size());

            rawOut.clear();
            errorOut.clear();
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &rawOut);

            CURLcode res = curl_easy_perform(curl);
            if (res != CURLE_OK) {
                errorOut = std::string("CURL error: ") + curl_easy_strerror(res);
                return "";
            }

            try {
                auto j = json::parse(rawOut);
                if (j.contains("response") && j["response"].is_string()) {
                    return j["response"].get<std::string>();
                }
                if (j.contains("message") && j["message"].is_object() && j["message"].contains("content")) {
                    return j["message"]["content"].get<std::string>();
                }
                if (j.contains("error") && j["error"].is_string()) {
                    errorOut = j["error"].get<std::string>();
                    return "";
                }
            } catch (const std::exception& e) {
                errorOut = std::string("JSON parse error: ") + e.what();
                return "";
            }

            errorOut = "Malformed response from Ollama.";
            return "";
        };

        std::string raw;
        std::string err;
        std::string response = callGenerate(model, raw, err);
        
        if (response.empty() && err.find("not found") != std::string::npos) {
            const std::string detected = detectOllamaModel();
            if (!detected.empty() && detected != model) {
                selectedModel = detected;
                response = callGenerate(selectedModel, raw, err);
            }
        }

        if (response.empty()) {
            DecisionTraceLogger traceLogger;
            DecisionTrace trace = traceLogger.startTrace("ollama_error", prompt.size());
            traceLogger.finishTrace(trace, false, std::string("Ollama failure: ") + err);
            traceLogger.writeTrace(trace);
            return std::string("Assistant: [Error] Ollama failed: ") + redactSensitiveText(err);
        }

        return response;

    } catch (const std::exception& e) {
        return std::string("Assistant: [Error] Ollama interface exception: ") + e.what();
    }
}


// ---- OpenAI backend ----
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
