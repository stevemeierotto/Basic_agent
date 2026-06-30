/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * basic_agent - AI Agent with Memory and RAG Capabilities
 * uses either Ollama lacal models or OpenAI API
 *
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text
 */

#pragma once
#include <string>
#include <mutex>
#include <cstdint>
#include <curl/curl.h>
#include "config.h"

enum class LLMBackend {
    Ollama,
    OpenAI
};

/** C6: token counts from the most recent LLM call and cumulative session totals. */
struct LlmTokenUsage {
    std::int64_t prompt_tokens = 0;
    std::int64_t completion_tokens = 0;
    std::int64_t total_tokens = 0;
};

class LLMInterface {
public:
     LLMInterface(LLMBackend backend = LLMBackend::Ollama, Config* cfg = nullptr);
    ~LLMInterface();

    // Internal helpers
    std::string askOllama(const std::string& prompt);
    std::string askOllama(const std::string& prompt, int num_predict_override);
    std::string askOpenAI(const std::string& prompt);

    std::string query(const std::string& prompt);
    /** @param num_predict_override Ollama num_predict; -1 uses config->max_tokens. */
    std::string query(const std::string& prompt, int num_predict_override);
    LLMBackend getBackend() const { return backend; }
    
    // Allow switching dynamically
    void setBackend(LLMBackend backend);
    void setConfig(Config* cfg) { config = cfg; }

    void useModel(const std::string& model) { selectedModel = model; }
    const std::string& getSelectedModel() const { return selectedModel; }

    void resetSessionTokenUsage();
    LlmTokenUsage lastCallTokenUsage() const;
    LlmTokenUsage sessionTokenUsage() const;

private:
    void recordTokenUsage(const LlmTokenUsage& usage);
    static LlmTokenUsage estimateTokenUsage(const std::string& prompt, const std::string& response);
    static LlmTokenUsage parseOllamaTokenUsage(const std::string& rawJson);
    static LlmTokenUsage parseOpenAiTokenUsage(const std::string& rawJson);

    std::string detectOllamaModel();
    std::string resolveOllamaModel();

    LLMBackend backend;
    Config* config;
    CURL* curl = nullptr;
    struct curl_slist* headers = nullptr;
    std::string selectedModel; 
    mutable std::recursive_mutex llmMutex;
    LlmTokenUsage last_call_usage_;
    LlmTokenUsage session_usage_;

    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

};

