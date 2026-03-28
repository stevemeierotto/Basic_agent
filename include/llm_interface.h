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
#include <curl/curl.h>
#include "config.h"

enum class LLMBackend {
    Ollama,
    OpenAI
};

class LLMInterface {
public:
     LLMInterface(LLMBackend backend = LLMBackend::Ollama, Config* cfg = nullptr);
    ~LLMInterface();

    // Internal helpers
    std::string askOllama(const std::string& prompt);
    std::string askOpenAI(const std::string& prompt);

    std::string query(const std::string& prompt);
    LLMBackend getBackend() const { return backend; }
    
    // Allow switching dynamically
    void setBackend(LLMBackend backend);
    void setConfig(Config* cfg) { config = cfg; }

    void useModel(const std::string& model) { selectedModel = model; }
    const std::string& getSelectedModel() const { return selectedModel; }

private:
    std::string detectOllamaModel();
    std::string resolveOllamaModel();

    LLMBackend backend;
    Config* config;
    CURL* curl = nullptr;
    struct curl_slist* headers = nullptr;
    std::string selectedModel; 
    mutable std::recursive_mutex llmMutex;

    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

};

