#pragma once
#include "similarity.h"
#include "embedding_engine.h"

#include <string>
#include <vector>
#include <utility>
#include <memory>

class VectorStore {
public:
    // non-owning pointer: RAGPipeline owns the engine
    explicit VectorStore(EmbeddingEngine* eng) 
        : embeddingEngine(eng), similarity(std::make_unique<CosineSimilarity>()) {}

    void setSimilarity(std::unique_ptr<ISimilarity> similarity);

    // Add document text (will use current engine to embed)
    void addDocument(const std::string& text);
    
    // Explicitly add document + embedding (e.g. from file)
    void addDocumentWithEmbedding(const std::string& text, std::vector<float> embedding);

    // Batch add documents
    void addDocuments(const std::vector<std::string>& texts);

    // Clear everything
    void clear();

    // Query for topK matches
    std::vector<std::pair<std::string, float>> retrieve(const std::string& query, int topK);

    // Persistence
    bool loadEmbeddings(const std::string& filepath);
    bool saveEmbeddings(const std::string& filepath) const;

    size_t getMemoryUsage() const;
    void enforceMemoryLimit(size_t maxMemoryBytes);

    size_t chunk_count() const { return documents.size(); }

    // Phase 7.1 Accessors
    const std::vector<std::vector<float>>& getEmbeddings() const { return embeddings; }
    const std::vector<std::string>& getDocuments() const { return documents; }

private:
    std::vector<std::vector<float>> embeddings;
    
    // Threshold for retrieval relevance (Phase 13 fix: lower to allow all signal)
    static constexpr float SIMILARITY_THRESHOLD = -1.0f;

    std::vector<std::string> documents;

    EmbeddingEngine* embeddingEngine;  // non-owning raw pointer
    std::unique_ptr<ISimilarity> similarity;
};
