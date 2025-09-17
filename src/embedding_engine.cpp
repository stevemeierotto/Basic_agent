#include "../include/embedding_engine.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <stdexcept>

EmbeddingEngine::EmbeddingEngine(Method method) : method(method) {}

void EmbeddingEngine::setMethod(Method m) {
    method = m;
}

std::vector<float> EmbeddingEngine::embed(const std::string& text) {
    switch (method) {
        case Method::Simple:  return embedSimple(text);
        case Method::TfIdf:   return embedTfIdf(text);
        case Method::WordHash:return embedWordHash(text);
        case Method::External:return embedExternal(text);
    }
    return {}; // fallback
}

std::vector<float> EmbeddingEngine::embedSimple(const std::string& text) {
    std::vector<float> vec(text.begin(), text.end());
    return normalizeVector(vec);
}

std::vector<float> EmbeddingEngine::embedTfIdf(const std::string& text) {
    updateVocabulary(text);
    std::vector<float> vec(VOCAB_SIZE, 0.0f);
    auto tokens = tokenize(text);
    for (const auto& t : tokens) {
        size_t idx = hashToIndex(t);
        float tf = std::count(tokens.begin(), tokens.end(), t) / float(tokens.size());
        vec[idx] = tf * calculateIdf(t);
    }
    return normalizeVector(vec);
}

std::vector<float> EmbeddingEngine::embedWordHash(const std::string& text) {
    auto tokens = tokenize(text);
    std::vector<float> vec(VOCAB_SIZE, 0.0f);
    for (const auto& t : tokens) {
        vec[hashToIndex(t)] += 1.0f;
    }
    return normalizeVector(vec);
}

std::vector<float> EmbeddingEngine::embedExternal(const std::string& text) {
    // Placeholder: call external API or service
    return embedSimple(text); // fallback for now
}

std::vector<std::string> EmbeddingEngine::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : text) {
        if (isalnum(c)) token += tolower(c);
        else if (!token.empty()) { tokens.push_back(token); token.clear(); }
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
    return log(float(documents.size()) / float(1 + it->second));
}

void EmbeddingEngine::updateVocabulary(const std::string& text) {
    auto tokens = tokenize(text);
    for (const auto& t : tokens) {
        globalTermFreq[t] += 1.0f;
        documentFreq[t] += 1;
    }
    documents.push_back(text);  // add document to corpus
}

std::vector<float> EmbeddingEngine::normalizeVector(std::vector<float> vec) const {
    float norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));
    if (norm > 0.0f) {
        for (auto& v : vec) v /= norm;
    }
    return vec;
}

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
        for (const auto& [term, freq] : globalTermFreq) {
            size_t len = term.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(term.data(), len);
            out.write(reinterpret_cast<const char*>(&freq), sizeof(freq));
        }

        // Save documentFreq
        size_t dfSize = documentFreq.size();
        out.write(reinterpret_cast<const char*>(&dfSize), sizeof(dfSize));
        for (const auto& [term, count] : documentFreq) {
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

