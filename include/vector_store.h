#pragma once
#include "similarity.h"
#include "embedding_engine.h"

#include <string>
#include <vector>
#include <utility>
#include <memory>

class VectorStore {
public:
    // non-owning pointer: RAGPipeline owns the engine via unique_ptr
    explicit VectorStore(EmbeddingEngine* engine)
        : embeddingEngine(engine) {}

    void addDocument(const std::string& text);
    void addDocuments(const std::vector<std::string>& texts);

    std::vector<std::vector<float>> embeddings;
    bool loadEmbeddings(const std::string& path);
    bool saveEmbeddings(const std::string& filepath) const;
    void enforceMemoryLimit(size_t maxMemoryBytes);
    size_t getMemoryUsage() const;

    void clear();

    void setSimilarity(std::unique_ptr<ISimilarity> sim);

    std::vector<std::pair<std::string, float>> retrieve(const std::string& query, int topK = 3);

private:
    static constexpr float SIMILARITY_THRESHOLD = 0.01f;

    std::vector<std::string> documents;

    EmbeddingEngine* embeddingEngine;  // non-owning raw pointer
    std::unique_ptr<ISimilarity> similarity =
        std::make_unique<DotProductSimilarity>();
};

