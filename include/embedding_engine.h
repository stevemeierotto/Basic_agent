#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class Config;

class EmbeddingEngine {
public:
    enum class Method {
        Simple,
        TfIdf,
        WordHash,
        External
    };

    explicit EmbeddingEngine(Method method = Method::TfIdf, Config* config = nullptr);
    ~EmbeddingEngine();

    void setMethod(Method method);
    Method getMethod() const { return method; }

    // Get current model name and dimension for metadata tracking
    std::string getModelName() const;
    int getDimension() const;
    int getInternalVersion() const { return 2; } // Increment when schema changes

    // Create embedding vector for text
    std::vector<float> embed(const std::string& text);

    // Create embedding vectors for multiple texts (Phase 2.3)
    std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts);

    // Save/load engine state (method + TF-IDF vocab/stats)
    bool saveState(const std::string& filepath) const;
    bool loadState(const std::string& filepath);

    // Normalize once and return
    std::vector<float> normalizeVector(std::vector<float> vec) const;

private:
    Method method;
    Config* config;
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

    // CURL resources for External method
    void* curl_handle;
    struct curl_slist* curl_headers;
    bool initCurl();
};
