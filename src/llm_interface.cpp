#include "../include/llm_interface.h"
#include <../include/json.hpp>
#include <curl/curl.h>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;


// Constructor
LLMInterface::LLMInterface(LLMBackend b, Config* cfg)
    : backend(b), curl(nullptr), headers(nullptr), config(cfg) // store Config pointer
{
    if (backend == LLMBackend::Ollama) {
        curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize CURL handle");
        }

        headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    }
}


LLMInterface::~LLMInterface() {
    if (headers) {
        curl_slist_free_all(headers);
        headers = nullptr;
    }
    if (curl) {
        curl_easy_cleanup(curl);
        curl = nullptr;
    }
}

void LLMInterface::setBackend(LLMBackend b) {
    backend = b;

    // Clean up old handles if switching backends
    if (curl) {
        curl_easy_cleanup(curl);
        curl = nullptr;
    }
    if (headers) {
        curl_slist_free_all(headers);
        headers = nullptr;
    }

    if (backend == LLMBackend::Ollama) {
        curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to re-initialize CURL handle");
        }
        headers = curl_slist_append(nullptr, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    }
}
// query method
std::string LLMInterface::query(const std::string& prompt) {
    // Read dynamic parameters from Config if available
    double temperature = config ? config->temperature : 0.7;
    double topP       = config ? config->top_p : 1.0;
    int maxTokens     = config ? config->max_tokens : 512;

    if (backend == LLMBackend::Ollama) {
        return askOllama(prompt);
    } else {
        return askOpenAI(prompt);
    }
}


// ---- Ollama backend ----

std::string LLMInterface::askOllama(const std::string& prompt) {
    std::string readBuffer;

    if (!curl) return "CURL not initialized.";

    // Pull dynamic parameters from Config if available
    double temperature = config ? config->temperature : 0.7;
    double topP       = config ? config->top_p : 1.0;
    int maxTokens     = config ? config->max_tokens : 512;

    json payload;
    payload["model"] = "qwen3:0.6b";
    payload["prompt"] = prompt;
    payload["stream"] = false;

    // Inject LLM control parameters
    payload["temperature"] = temperature;
    payload["top_p"] = topP;
    payload["max_tokens"] = maxTokens;

    std::string jsonStr = payload.dump();   // keep alive!

    // Reset headers each call
    struct curl_slist* localHeaders = nullptr;
    localHeaders = curl_slist_append(localHeaders, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, localHeaders);

    // ✅ Use the stable string
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, jsonStr.size());

    // ✅ Hook up callback
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(localHeaders);

    if (res != CURLE_OK) {
        return std::string("CURL error: ") + curl_easy_strerror(res);
    }

    try {
        auto j = json::parse(readBuffer);
        if (j.contains("response")) {
            return j["response"].get<std::string>();
        }
    } catch (const std::exception& e) {
        return std::string("JSON parse error: ") + e.what() + "\nRaw: " + readBuffer;
    }

    return "No response from Ollama.\nRaw: " + readBuffer;
}


// ---- OpenAI backend ----
std::string LLMInterface::askOpenAI(const std::string& prompt) {
    CURL* curl = curl_easy_init();
    std::string readBuffer;

    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }

    // Build request payload
    json data = {
        {"model", "gpt-3.5-turbo"},   // Change this to gpt-4, gpt-4o, etc.
        {"messages", {
            {{"role", "system"}, {"content", "You are a helpful C++ coding assistant."}},
            {{"role", "user"}, {"content", prompt}}
        }},
        {"temperature", 0.7}
    };

    std::string payload = data.dump();

    // Setup headers
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    // Pull API key from environment instead of hardcoding
    const char* apiKey = std::getenv("OPENAI_API_KEY");
    if (!apiKey) {
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
        throw std::runtime_error("OPENAI_API_KEY not set in environment");
    }
    std::string authHeader = std::string("Authorization: Bearer ") + apiKey;
    headers = curl_slist_append(headers, authHeader.c_str());

    // Configure CURL
    curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/chat/completions");
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);   // <-- you already have this
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

    CURLcode res = curl_easy_perform(curl);

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        throw std::runtime_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
    }

    // Parse JSON response
    try {
        json json_response = json::parse(readBuffer);
        if (!json_response.contains("choices") || json_response["choices"].empty())
            throw std::runtime_error("OpenAI API returned no choices");

        auto message = json_response["choices"][0]["message"];
        return message["content"].get<std::string>();
    } catch (const std::exception& e) {
        return std::string("Error parsing OpenAI response: ") + e.what() + "\nRaw: " + readBuffer;
    }
}

