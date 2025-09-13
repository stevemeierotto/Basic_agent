#ifndef VECTOR_STORE_H
#define VECTOR_STORE_H

#include <string>
#include <vector>
#include <utility> // for std::pair

class VectorStore {
public:
    // Add a document and its vector embedding
    void addDocument(const std::string& text);

    // Clear in-memory store so it can be rebuilt from chunks
    void clear();

    // Retrieve topK most similar documents for a query
    std::vector<std::pair<std::string, float>> retrieve(const std::string& query, int topK = 3);

private:
    // Stored raw text + embeddings
    std::vector<std::string> documents;
    std::vector<std::vector<float>> embeddings;

    // Create embedding for a text (placeholder for now)
    std::vector<float> embed(const std::string& text);

    // Similarity function
    float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
};

#endif // VECTOR_STORE_H

