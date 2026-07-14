#include "../include/embedding_engine.h"
#include "../include/config.h"
#include "../include/inference_client.h"
#include "../include/inference_endpoint.h"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

void EmbeddingEngine::ensureInferenceClient() {
    if (inference_client_) {
        return;
    }
    const auto endpoints = config ? Thoth::resolveInferenceEndpoints(*config)
                                  : Thoth::resolveInferenceEndpoints();
    inference_client_ = Thoth::createInferenceClient(endpoints, config);
}

EmbeddingEngine::EmbeddingEngine(Method method, Config* config) 
    : method(method), config(config) {}

EmbeddingEngine::~EmbeddingEngine() = default;

void EmbeddingEngine::setMethod(Method m) {
    method = m;
    if (m != Method::External) {
        inference_client_.reset();
    }
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

    try {
        ensureInferenceClient();
    } catch (const std::exception& e) {
        std::cerr << "[EmbeddingEngine] Inference client unavailable: " << e.what()
                  << ". Falling back to local TfIdf.\n";
        std::vector<std::vector<float>> finalResults;
        for (const auto& t : texts) {
            finalResults.push_back(normalizeVector(embedTfIdf(t)));
        }
        return finalResults;
    }

    if (!inference_client_) {
        std::cerr << "[EmbeddingEngine] Inference service unreachable. Falling back to local TfIdf.\n";
        std::vector<std::vector<float>> finalResults;
        for (const auto& t : texts) {
            finalResults.push_back(normalizeVector(embedTfIdf(t)));
        }
        return finalResults;
    }

    const std::string model = getModelName();
    std::vector<std::vector<float>> finalResults;
    finalResults.reserve(texts.size());

    const size_t MAX_BATCH_SIZE = 10;
    const size_t MAX_CHAR_LIMIT = 8000;

    for (size_t i = 0; i < texts.size(); i += MAX_BATCH_SIZE) {
        if (method != Method::External) break;

        size_t end = std::min(i + MAX_BATCH_SIZE, texts.size());
        std::vector<std::string> microBatch;
        for (size_t j = i; j < end; ++j) {
            if (texts[j].size() > MAX_CHAR_LIMIT) {
                microBatch.push_back(texts[j].substr(0, MAX_CHAR_LIMIT));
            } else {
                microBatch.push_back(texts[j]);
            }
        }

        Thoth::InferenceEmbedRequest request;
        request.model = model;
        request.inputs = microBatch;

        const auto embedded = inference_client_->embed(request);
        if (embedded.ok && embedded.embeddings.size() == microBatch.size()) {
            for (const auto& vec : embedded.embeddings) {
                finalResults.push_back(normalizeVector(vec));
            }
            continue;
        }

        std::cerr << "[EmbeddingEngine] " << inference_client_->backendName() << " embed failed";
        if (!embedded.error.empty()) {
            std::cerr << ": " << embedded.error;
        }
        std::cerr << ". Falling back micro-batch items to TfIdf.\n";
        for (const auto& text : microBatch) {
            finalResults.push_back(normalizeVector(embedTfIdf(text)));
        }
    }

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
    try {
        ensureInferenceClient();
    } catch (...) {
        return embedTfIdf(text);
    }
    if (!inference_client_) {
        return embedTfIdf(text);
    }

    const size_t MAX_CHAR_LIMIT = 8000;
    std::string safeText = text;
    if (safeText.size() > MAX_CHAR_LIMIT) {
        std::cerr << "[WARN] Truncating large single-item chunk for embedding (" << safeText.size() << " chars)\n";
        safeText = safeText.substr(0, MAX_CHAR_LIMIT);
    }

    Thoth::InferenceEmbedRequest request;
    request.model = getModelName();
    request.inputs = {safeText};

    const auto embedded = inference_client_->embed(request);
    if (embedded.ok && !embedded.embeddings.empty()) {
        return embedded.embeddings.front();
    }

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
    if (method == Method::External) return 768;
    return VOCAB_SIZE;
}

bool EmbeddingEngine::saveState(const std::string& path) const {
    return true;
}

bool EmbeddingEngine::loadState(const std::string& path) {
    return true;
}
