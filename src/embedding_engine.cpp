#include "../include/embedding_engine.h"
#include "../include/config.h"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <curl/curl.h>
#include <../include/json.hpp>

using json = nlohmann::json;

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

EmbeddingEngine::EmbeddingEngine(Method method, Config* config) 
    : method(method), config(config), curl_headers(nullptr) {
    if (method == Method::External) {
        curl_headers = curl_slist_append(nullptr, "Content-Type: application/json");
    }
}

EmbeddingEngine::~EmbeddingEngine() {
    std::lock_guard<std::mutex> lock(engineMutex);
    if (curl_headers) curl_slist_free_all(curl_headers);
    for (void* handle : curl_pool) {
        curl_easy_cleanup(static_cast<CURL*>(handle));
    }
}

void* EmbeddingEngine::acquireCurlHandle() {
    std::lock_guard<std::mutex> lock(engineMutex);
    if (!curl_pool.empty()) {
        void* handle = curl_pool.back();
        curl_pool.pop_back();
        return handle;
    }

    CURL* curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    }
    return curl;
}

void EmbeddingEngine::releaseCurlHandle(void* handle) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(engineMutex);
    curl_pool.push_back(handle);
}

bool EmbeddingEngine::initCurl() {
    return true;
}

void EmbeddingEngine::setMethod(Method m) {
    method = m;
}

std::vector<float> EmbeddingEngine::embed(const std::string& text) {
    if (text.empty()) return {};

    std::vector<float> vec;
    switch (method) {
        case Method::Simple:
            vec = embedSimple(text);
            break;
        case Method::TfIdf:
            vec = embedTfIdf(text);
            break;
        case Method::WordHash:
            vec = embedWordHash(text);
            break;
        case Method::External:
            vec = embedExternal(text);
            break;
        default:
            return {};
    }

    if (vec.empty()) return {};
    return normalizeVector(std::move(vec));
}

std::vector<std::vector<float>> EmbeddingEngine::embedBatch(const std::vector<std::string>& texts) {
    if (texts.empty()) return {};

    if (method != Method::External) {
        std::vector<std::vector<float>> results;
        results.reserve(texts.size());
        for (const auto& t : texts) {
            results.push_back(embed(t));
        }
        return results;
    }

    void* curl = acquireCurlHandle();
    if (!curl) {
        std::cerr << "[EmbeddingEngine] Ollama service unreachable. Falling back to local TfIdf.\n";
        // Fallback for this batch
        std::vector<std::vector<float>> finalResults;
        for (const auto& t : texts) finalResults.push_back(normalizeVector(embedTfIdf(t)));
        return finalResults;
    }

    const std::string model = getModelName();
    std::vector<std::vector<float>> finalResults;
    finalResults.reserve(texts.size());

    // Phase 13 Hardening: Use micro-batches for efficiency
    const size_t MAX_BATCH_SIZE = 10; 
    const size_t MAX_CHAR_LIMIT = 8000; // Standard truncation (~2000 tokens)
    
    for (size_t i = 0; i < texts.size(); i += MAX_BATCH_SIZE) {
        if (method != Method::External) break; // Safety if changed during loop

        size_t end = std::min(i + MAX_BATCH_SIZE, texts.size());
        std::vector<std::string> microBatch;
        for (size_t j = i; j < end; ++j) {
            if (texts[j].size() > MAX_CHAR_LIMIT) {
                microBatch.push_back(texts[j].substr(0, MAX_CHAR_LIMIT));
            } else {
                microBatch.push_back(texts[j]);
            }
        }

        json payload;
        payload["model"] = model;
        payload["input"] = microBatch;
        
        std::string jsonStr = payload.dump();
        std::string readBuffer;

        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:11434/api/embed");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        CURLcode res = curl_easy_perform(static_cast<CURL*>(curl));
        bool microBatchSuccess = false;

        if (res == CURLE_OK) {
            try {
                auto j = json::parse(readBuffer);
                if (j.contains("embeddings") && j["embeddings"].is_array()) {
                    for (const auto& emb : j["embeddings"]) {
                        finalResults.push_back(normalizeVector(emb.get<std::vector<float>>()));
                    }
                    microBatchSuccess = true;
                    // std::cout << "." << std::flush;
                }
            } catch (...) {
                std::cerr << "\n[EmbeddingEngine] Ollama parse failed for batch starting at " << i << "\n";
            }
        } else {
            std::cerr << "\n[EmbeddingEngine] Ollama connection failed: " << curl_easy_strerror(res) << "\n";
        }

        if (!microBatchSuccess) {
            std::cerr << "[EmbeddingEngine] Falling back micro-batch items to TfIdf.\n";
            for (const auto& text : microBatch) {
                finalResults.push_back(normalizeVector(embedTfIdf(text)));
            }
        }
    }

    releaseCurlHandle(curl);
    return finalResults;
}

std::vector<float> EmbeddingEngine::embedSimple(const std::string& text) {
    std::vector<float> vec;
    vec.reserve(text.size());
    for (unsigned char c : text) vec.push_back(static_cast<float>(c));
    return vec;
}

std::vector<float> EmbeddingEngine::embedTfIdf(const std::string& text) {
    std::vector<float> vec(VOCAB_SIZE, 0.0f);
    auto tokens = tokenize(text);
    if (tokens.empty()) return vec;

    std::unordered_map<std::string, float> tf;
    for (const auto& t : tokens) tf[t] += 1.0f;

    for (const auto& [term, count] : tf) {
        size_t idx = hashToIndex(term);
        float termFreq = count / static_cast<float>(tokens.size());
        vec[idx] = termFreq * calculateIdf(term);
    }
    return vec;
}

std::vector<float> EmbeddingEngine::embedWordHash(const std::string& text) {
    std::vector<float> vec(VOCAB_SIZE, 0.0f);
    auto tokens = tokenize(text);
    for (const auto& t : tokens) {
        vec[hashToIndex(t)] += 1.0f;
    }
    return vec;
}

std::vector<float> EmbeddingEngine::embedExternal(const std::string& text) {
    void* curl = acquireCurlHandle();
    if (!curl) return {};

    const size_t MAX_CHAR_LIMIT = 8000;
    std::string safeText = text;
    if (safeText.size() > MAX_CHAR_LIMIT) {
        std::cerr << "[WARN] Truncating large single-item chunk for embedding (" << safeText.size() << " chars)\n";
        safeText = safeText.substr(0, MAX_CHAR_LIMIT);
    }

    json payload;
    payload["model"] = getModelName();
    payload["input"] = safeText;
    
    std::string jsonStr = payload.dump();
    std::string readBuffer;

    curl_easy_setopt(static_cast<CURL*>(curl), CURLOPT_URL, "http://127.0.0.1:11434/api/embed");
    curl_easy_setopt(static_cast<CURL*>(curl), CURLOPT_POSTFIELDS, jsonStr.c_str());
    curl_easy_setopt(static_cast<CURL*>(curl), CURLOPT_WRITEDATA, &readBuffer);

    CURLcode res = curl_easy_perform(static_cast<CURL*>(curl));
    if (res != CURLE_OK) {
        releaseCurlHandle(curl);
        return embedTfIdf(text);
    }

    try {
        auto j = nlohmann::json::parse(readBuffer);
        if (j.contains("embeddings") && j["embeddings"].is_array() && !j["embeddings"].empty()) {
            auto result = j["embeddings"][0].get<std::vector<float>>();
            releaseCurlHandle(curl);
            return result;
        }
    } catch (...) {}

    releaseCurlHandle(curl);
    return embedTfIdf(text);
}

std::vector<std::string> EmbeddingEngine::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            token += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (!token.empty()) {
            tokens.push_back(token);
            token.clear();
        }
    }
    if (!token.empty()) tokens.push_back(token);
    return tokens;
}

size_t EmbeddingEngine::hashToIndex(const std::string& term) const {
    return std::hash<std::string>{}(term) % VOCAB_SIZE;
}

float EmbeddingEngine::calculateIdf(const std::string& term) const {
    auto it = documentFreq.find(term);
    if (it == documentFreq.end() || it->second == 0) return 0.0f;
    return std::log((1.0f + static_cast<float>(documents.size())) / (1.0f + static_cast<float>(it->second))) + 1.0f;
}

void EmbeddingEngine::updateVocabulary(const std::string& text) {
    std::lock_guard<std::mutex> lock(engineMutex);
    auto tokens = tokenize(text);
    for (const auto& t : tokens) {
        globalTermFreq[t] += 1.0f;
        documentFreq[t] += 1;
    }
    documents.push_back(text);
}

std::vector<float> EmbeddingEngine::normalizeVector(std::vector<float> vec) const {
    if (vec.empty()) return vec;
    float norm = 0.0f;
    for (float v : vec) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-9f) {
        for (float& v : vec) v /= norm;
    }
    return vec;
}

std::string EmbeddingEngine::getModelName() const {
    if (config && !config->embedding_model.empty()) return config->embedding_model;
    return "nomic-embed-text:v1.5";
}

int EmbeddingEngine::getDimension() const {
    if (method == Method::External) return 768; // nomic-embed-text
    return VOCAB_SIZE;
}

bool EmbeddingEngine::saveState(const std::string& path) const {
    return true;
}

bool EmbeddingEngine::loadState(const std::string& path) {
    return true;
}
