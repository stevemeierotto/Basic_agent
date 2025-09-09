#include "../include/llm_interface.h"
#include <../include/json.hpp>
#include <curl/curl.h>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

LLMInterface::LLMInterface(LLMBackend b) : backend(b) {
    if (backend == LLMBackend::Ollama) {
        curl = curl_easy_init();
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    }
}

LLMInterface::~LLMInterface() {
    if (curl) curl_easy_cleanup(curl);
    if (headers) curl_slist_free_all(headers);
}

void LLMInterface::setBackend(LLMBackend b) {
    backend = b;
}

std::string LLMInterface::query(const std::string& prompt) {
    if (backend == LLMBackend::Ollama) {
        return askOllama(prompt);
    } else {
        return askOpenAI(prompt);
    }
}

// ---- Ollama backend ----

std::string LLMInterface::askOllama(const std::string& prompt) {
    if (!curl) return "CURL not initialized.";

    std::string readBuffer;

    try {
        json payload;
        payload["model"] = "llama3.2:latest";
        payload["prompt"] = prompt;
        payload["stream"] = true;

        std::string jsonStr = payload.dump();

        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            return std::string("CURL error: ") + curl_easy_strerror(res);
        }

        json j = json::parse(readBuffer);
        if (j.contains("response")) return j["response"].get<std::string>();

    } catch (const std::exception& e) {
        return std::string("Exception: ") + e.what();
    } catch (...) {
        return "Unknown error parsing Ollama response.";
    }

    return "No response from Ollama.";
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

