#include "../include/embedding_engine.h"
#include "../include/config.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <curl/curl.h>
#include <../include/json.hpp>

using json = nlohmann::json;

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

EmbeddingEngine::EmbeddingEngine(Method method, Config* config)
    : method(method), config(config), curl_handle(nullptr), curl_headers(nullptr) {
}

EmbeddingEngine::~EmbeddingEngine() {
    if (curl_headers) curl_slist_free_all(curl_headers);
    if (curl_handle) curl_easy_cleanup(static_cast<CURL*>(curl_handle));
}

std::string EmbeddingEngine::getModelName() const {
    if (method == Method::External) {
        const char* envModel = std::getenv("OLLAMA_EMBED_MODEL");
        return envModel ? envModel : "nomic-embed-text";
    }
    return "local-tfidf";
}

int EmbeddingEngine::getDimension() const {
    if (method == Method::External) {
        std::string model = getModelName();
        if (model.find("nomic") != std::string::npos) return 768;
        if (model.find("bge-small") != std::string::npos) return 384;
        return 768; 
    }
    return static_cast<int>(VOCAB_SIZE);
}

bool EmbeddingEngine::initCurl() {
    if (curl_handle) return true;
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    curl_headers = curl_slist_append(nullptr, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    
    // Increased timeouts for robustness
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    
    curl_handle = curl;
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

    if (!initCurl()) {
        std::cerr << "[EmbeddingEngine] Ollama service unreachable. Permanent fallback to local TfIdf.\n";
        method = Method::TfIdf;
        return embedBatch(texts);
    }

    std::string model = getModelName();

    json payload;
    payload["model"] = model;
    payload["input"] = texts;
    
    std::string jsonStr = payload.dump();
    std::string readBuffer;

    CURL* curl = static_cast<CURL*>(curl_handle);
    curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/embed");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "[EmbeddingEngine] Ollama batch failed: " << curl_easy_strerror(res) << ". Permanent fallback to TfIdf.\n";
        method = Method::TfIdf;
        return embedBatch(texts);
    }

    try {
        auto j = json::parse(readBuffer);
        if (j.contains("embeddings") && j["embeddings"].is_array()) {
            std::vector<std::vector<float>> results;
            for (const auto& emb : j["embeddings"]) {
                results.push_back(normalizeVector(emb.get<std::vector<float>>()));
            }
            return results;
        }
    } catch (...) {}

    std::cerr << "[EmbeddingEngine] Ollama parse failed. Permanent fallback to TfIdf.\n";
    method = Method::TfIdf;
    return embedBatch(texts);
}

std::vector<float> EmbeddingEngine::embedSimple(const std::string& text) {
    std::vector<float> vec;
    vec.reserve(text.size());
    for (unsigned char c : text) vec.push_back(static_cast<float>(c));
    return vec;
}

std::vector<float> EmbeddingEngine::embedTfIdf(const std::string& text) {
    updateVocabulary(text);
    std::vector<float> vec(VOCAB_SIZE, 0.0f);
    auto tokens = tokenize(text);
    if (tokens.empty()) return vec;

    for (const auto& t : tokens) {
        size_t idx = hashToIndex(t);
        float tf = std::count(tokens.begin(), tokens.end(), t) / static_cast<float>(tokens.size());
        vec[idx] = tf * calculateIdf(t);
    }
    return vec;
}

std::vector<float> EmbeddingEngine::embedWordHash(const std::string& text) {
    auto tokens = tokenize(text);
    std::vector<float> vec(VOCAB_SIZE, 0.0f);
    for (const auto& t : tokens) vec[hashToIndex(t)] += 1.0f;
    return vec;
}

std::vector<float> EmbeddingEngine::embedExternal(const std::string& text) {
    if (!initCurl()) {
        std::cerr << "[EmbeddingEngine] Ollama service unreachable. Permanent fallback to local TfIdf.\n";
        method = Method::TfIdf;
        return embedTfIdf(text);
    }

    std::string model = getModelName();

    json payload;
    payload["model"] = model;
    payload["input"] = text;
    
    std::string jsonStr = payload.dump();
    std::string readBuffer;

    CURL* curl = static_cast<CURL*>(curl_handle);
    curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/embed");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "[EmbeddingEngine] Ollama failed: " << curl_easy_strerror(res) << ". Permanent fallback to TfIdf.\n";
        method = Method::TfIdf;
        return embedTfIdf(text);
    }

    try {
        auto j = json::parse(readBuffer);
        if (j.contains("embeddings") && j["embeddings"].is_array() && !j["embeddings"].empty()) {
            return j["embeddings"][0].get<std::vector<float>>();
        }
    } catch (...) {}

    std::cerr << "[EmbeddingEngine] Ollama result empty. Permanent fallback to TfIdf.\n";
    method = Method::TfIdf;
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
    auto tokens = tokenize(text);
    for (const auto& t : tokens) {
        globalTermFreq[t] += 1.0f;
        documentFreq[t] += 1;
    }
    documents.push_back(text);
}

std::vector<float> EmbeddingEngine::normalizeVector(std::vector<float> vec) const {
    if (vec.empty()) return vec;
    double sumSq = 0.0;
    for (float v : vec) sumSq += static_cast<double>(v) * v;
    float norm = static_cast<float>(std::sqrt(sumSq));
    if (norm > 1e-9f) {
        for (auto& v : vec) v /= norm;
    }
    return vec;
}

bool EmbeddingEngine::saveState(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) return false;
    int methodInt = static_cast<int>(method);
    out.write(reinterpret_cast<const char*>(&methodInt), sizeof(methodInt));
    size_t numDocs = documents.size();
    out.write(reinterpret_cast<const char*>(&numDocs), sizeof(numDocs));
    for (const auto& doc : documents) {
        size_t len = doc.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(doc.data(), len);
    }
    size_t gtfSize = globalTermFreq.size();
    out.write(reinterpret_cast<const char*>(&gtfSize), sizeof(gtfSize));
    for (const auto& kv : globalTermFreq) {
        size_t len = kv.first.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(kv.first.data(), len);
        out.write(reinterpret_cast<const char*>(&kv.second), sizeof(kv.second));
    }
    size_t dfSize = documentFreq.size();
    out.write(reinterpret_cast<const char*>(&dfSize), sizeof(dfSize));
    for (const auto& kv : documentFreq) {
        size_t len = kv.first.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(kv.first.data(), len);
        out.write(reinterpret_cast<const char*>(&kv.second), sizeof(kv.second));
    }
    return true;
}

bool EmbeddingEngine::loadState(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) return false;
    int methodInt;
    in.read(reinterpret_cast<char*>(&methodInt), sizeof(methodInt));
    method = static_cast<Method>(methodInt);
    size_t numDocs;
    in.read(reinterpret_cast<char*>(&numDocs), sizeof(numDocs));
    documents.clear();
    for (size_t i = 0; i < numDocs; ++i) {
        size_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string doc(len, '\0');
        in.read(&doc[0], len);
        documents.push_back(doc);
    }
    size_t gtfSize;
    in.read(reinterpret_cast<char*>(&gtfSize), sizeof(gtfSize));
    globalTermFreq.clear();
    for (size_t i = 0; i < gtfSize; ++i) {
        size_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string term(len, '\0');
        in.read(&term[0], len);
        float freq;
        in.read(reinterpret_cast<char*>(&freq), sizeof(freq));
        globalTermFreq[term] = freq;
    }
    size_t dfSize;
    in.read(reinterpret_cast<char*>(&dfSize), sizeof(dfSize));
    documentFreq.clear();
    for (size_t i = 0; i < dfSize; ++i) {
        size_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string term(len, '\0');
        in.read(&term[0], len);
        size_t count;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        documentFreq[term] = count;
    }
    return true;
}
