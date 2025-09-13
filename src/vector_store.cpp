#include "../include/vector_store.h"
#include <cmath>
#include <algorithm>

// Add document: store text and embedding
void VectorStore::addDocument(const std::string& text) {
    documents.push_back(text);
    embeddings.push_back(embed(text));
}

void VectorStore::clear() {
    documents.clear();
    embeddings.clear();
}

// Retrieve topK similar documents
std::vector<std::pair<std::string, float>> VectorStore::retrieve(const std::string& query, int topK) {
    std::vector<std::pair<std::string, float>> results;

    auto queryEmbedding = embed(query);

    for (size_t i = 0; i < documents.size(); ++i) {
        float score = cosineSimilarity(queryEmbedding, embeddings[i]);
        results.push_back({documents[i], score});
    }

    // Sort by similarity score (descending)
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    if (results.size() > static_cast<size_t>(topK)) {
        results.resize(topK);
    }

    return results;
}

// Placeholder embed function: turns text into simple vector
// For now: each char value normalized, just to test retrieval
std::vector<float> VectorStore::embed(const std::string& text) {
    std::vector<float> vec;
    for (char c : text) {
        vec.push_back(static_cast<float>(c) / 255.0f); // normalize ASCII values
    }
    return vec;
}

// Cosine similarity between two vectors
float VectorStore::cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty()) return 0.0f;

    size_t len = std::min(a.size(), b.size());

    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    if (normA == 0.0f || normB == 0.0f) return 0.0f;
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

