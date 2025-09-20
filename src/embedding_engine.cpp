#include "../include/embedding_engine.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <iostream>

EmbeddingEngine::EmbeddingEngine(Method method) : method(method) {}

void EmbeddingEngine::setMethod(Method m) {
    method = m;
}

// ------------------------------------------------------------------
// Public API: central entrypoint for all callers
// ------------------------------------------------------------------
std::vector<float> EmbeddingEngine::embed(const std::string& text) {
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
            std::cerr << "[EmbeddingEngine] Unknown method, returning empty vector\n";
            return {};
    }

    // Basic validation
    if (vec.empty()) {
        std::cerr << "[EmbeddingEngine] Warning: embedding returned empty vector (text length="
                  << text.size() << ")\n";
        return {};
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        if (!std::isfinite(vec[i])) {
            std::cerr << "[EmbeddingEngine] Warning: non-finite embedding value at index "
                      << i << " (text length=" << text.size() << ")\n";
            return {};
        }
    }

    // Normalize once and return
    return normalizeVector(std::move(vec));
}

// ------------------------------------------------------------------
// Embedding implementations (produce raw vectors only)
// ------------------------------------------------------------------
std::vector<float> EmbeddingEngine::embedSimple(const std::string& text) {
    // Simple per-character counts (raw)
    std::vector<float> vec;
    vec.reserve(text.size());
    for (unsigned char c : text) {
        vec.push_back(static_cast<float>(c));
    }
    return vec;
}

std::vector<float> EmbeddingEngine::embedTfIdf(const std::string& text) {
    // Update vocabulary/state for TF-IDF (keeps corpus stats)
    updateVocabulary(text);

    // Create TF-IDF-like vector (VOCAB_SIZE may be large)
    std::vector<float> vec(VOCAB_SIZE, 0.0f);
    auto tokens = tokenize(text);
    if (tokens.empty()) return vec;

    // Compute term frequencies in this document
    for (const auto& t : tokens) {
        size_t idx = hashToIndex(t);
        float tf = std::count(tokens.begin(), tokens.end(), t) / static_cast<float>(tokens.size());
        vec[idx] = tf * calculateIdf(t);
    }

    return vec; // raw
}

std::vector<float> EmbeddingEngine::embedWordHash(const std::string& text) {
    auto tokens = tokenize(text);
    std::vector<float> vec(VOCAB_SIZE, 0.0f);
    for (const auto& t : tokens) {
        vec[hashToIndex(t)] += 1.0f;
    }
    return vec; // raw
}

std::vector<float> EmbeddingEngine::embedExternal(const std::string& text) {
    // Placeholder for external provider call. Return a raw vector.
    // For now use a simple fallback so callers still get a non-empty vector.
    std::vector<float> vec;
    if (text.empty()) {
        vec.push_back(0.0f);
        return vec;
    }

    // Very small deterministic stub: fill with hashed values
    const size_t OUT_SZ = std::min<size_t>(VOCAB_SIZE, 512);
    vec.assign(OUT_SZ, 0.0f);
    size_t h = std::hash<std::string>{}(text);
    vec[h % OUT_SZ] = static_cast<float>((h & 0xffff) / 65535.0);
    return vec;
}

// ------------------------------------------------------------------
// Tokenization / helpers
// ------------------------------------------------------------------
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
    return std::log(static_cast<float>(documents.size()) / static_cast<float>(1 + it->second));
}

void EmbeddingEngine::updateVocabulary(const std::string& text) {
    auto tokens = tokenize(text);
    // Update corpus statistics
    for (const auto& t : tokens) {
        globalTermFreq[t] += 1.0f;
        documentFreq[t] += 1;
    }
    documents.push_back(text);  // add document to corpus
}

// ------------------------------------------------------------------
// Normalization helper (kept as member; accepts by-value or moved vector)
// ------------------------------------------------------------------
std::vector<float> EmbeddingEngine::normalizeVector(std::vector<float> vec) const {
    // Use inner_product to compute squared norm
    float norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));
    if (norm > 0.0f) {
        for (auto& v : vec) v /= norm;
    } else {
        // If the vector is effectively zero, leave as-is but warn
        std::cerr << "[EmbeddingEngine] Warning: zero-norm embedding encountered during normalization\n";
    }
    return vec;
}

// ------------------------------------------------------------------
// Persistence (unchanged, preserved)
 // ------------------------------------------------------------------
bool EmbeddingEngine::saveState(const std::string& filepath) const {
    try {
        std::ofstream out(filepath, std::ios::binary);
        if (!out) return false;

        // Save method
        int methodInt = static_cast<int>(method);
        out.write(reinterpret_cast<const char*>(&methodInt), sizeof(methodInt));

        // Save documents
        size_t numDocs = documents.size();
        out.write(reinterpret_cast<const char*>(&numDocs), sizeof(numDocs));
        for (const auto& doc : documents) {
            size_t len = doc.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(doc.data(), len);
        }

        // Save globalTermFreq
        size_t gtfSize = globalTermFreq.size();
        out.write(reinterpret_cast<const char*>(&gtfSize), sizeof(gtfSize));
        for (const auto& kv : globalTermFreq) {
            const std::string& term = kv.first;
            float freq = kv.second;
            size_t len = term.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(term.data(), len);
            out.write(reinterpret_cast<const char*>(&freq), sizeof(freq));
        }

        // Save documentFreq
        size_t dfSize = documentFreq.size();
        out.write(reinterpret_cast<const char*>(&dfSize), sizeof(dfSize));
        for (const auto& kv : documentFreq) {
            const std::string& term = kv.first;
            size_t count = kv.second;
            size_t len = term.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(term.data(), len);
            out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        }

        return true;
    } catch (...) {
        return false;
    }
}

bool EmbeddingEngine::loadState(const std::string& filepath) {
    try {
        std::ifstream in(filepath, std::ios::binary);
        if (!in) return false;

        documents.clear();
        globalTermFreq.clear();
        documentFreq.clear();

        // Load method
        int methodInt = 0;
        in.read(reinterpret_cast<char*>(&methodInt), sizeof(methodInt));
        method = static_cast<Method>(methodInt);

        // Load documents
        size_t numDocs = 0;
        in.read(reinterpret_cast<char*>(&numDocs), sizeof(numDocs));
        for (size_t i = 0; i < numDocs; ++i) {
            size_t len = 0;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string doc(len, '\0');
            in.read(&doc[0], len);
            documents.push_back(std::move(doc));
        }

        // Load globalTermFreq
        size_t gtfSize = 0;
        in.read(reinterpret_cast<char*>(&gtfSize), sizeof(gtfSize));
        for (size_t i = 0; i < gtfSize; ++i) {
            size_t len = 0;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string term(len, '\0');
            in.read(&term[0], len);
            float freq;
            in.read(reinterpret_cast<char*>(&freq), sizeof(freq));
            globalTermFreq[std::move(term)] = freq;
        }

        // Load documentFreq
        size_t dfSize = 0;
        in.read(reinterpret_cast<char*>(&dfSize), sizeof(dfSize));
        for (size_t i = 0; i < dfSize; ++i) {
            size_t len = 0;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string term(len, '\0');
            in.read(&term[0], len);
            size_t count;
            in.read(reinterpret_cast<char*>(&count), sizeof(count));
            documentFreq[std::move(term)] = count;
        }

        return true;
    } catch (...) {
        return false;
    }
}

