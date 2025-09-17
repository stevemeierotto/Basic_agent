#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class EmbeddingEngine {
public:
    enum class Method {
        Simple,
        TfIdf,
        WordHash,
        External
    };

    EmbeddingEngine(Method method = Method::TfIdf);

    void setMethod(Method method);

    // Create embedding vector for text
    std::vector<float> embed(const std::string& text);

    // Save/load engine state (method + TF-IDF vocab/stats)
    bool saveState(const std::string& filepath) const;
    bool loadState(const std::string& filepath);

private:
    Method method;
    static constexpr size_t VOCAB_SIZE = 10000;
    std::vector<std::string> documents; // tracks all indexed texts
    // TF-IDF state
    std::unordered_map<std::string, float> globalTermFreq;
    std::unordered_map<std::string, size_t> documentFreq;

    // Embedding implementations
    std::vector<float> embedSimple(const std::string& text);
    std::vector<float> embedTfIdf(const std::string& text);
    std::vector<float> embedWordHash(const std::string& text);
    std::vector<float> embedExternal(const std::string& text);

    // Helpers
    std::vector<std::string> tokenize(const std::string& text) const;
    size_t hashToIndex(const std::string& term) const;
    float calculateIdf(const std::string& term) const;
    void updateVocabulary(const std::string& text);
    std::vector<float> normalizeVector(std::vector<float> vec) const;
};

