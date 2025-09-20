#include "vector_store.h"
#include <algorithm>
#include <fstream>
#include <queue>
#include <vector>
#include <string>
#include <functional>
#include <iostream>

void VectorStore::addDocument(const std::string& text) {
    documents.push_back(text);

    auto emb = embeddingEngine->embed(text);
    std::cerr << "[DEBUG] Embedding generated, size=" << emb.size() << "\n";

    if (emb.empty()) {
        std::cerr << "[ERROR] Empty embedding for document! Text=\"" 
                  << text.substr(0, 50) << (text.size() > 50 ? "..." : "") 
                  << "\"\n";
    }

    embeddings.push_back(std::move(emb));
    std::cerr << "[DEBUG] Added doc. Total docs=" << documents.size() 
              << ", total embeddings=" << embeddings.size() << "\n";
}


void VectorStore::addDocuments(const std::vector<std::string>& texts) {
    for (const auto& t : texts) addDocument(t);
}

void VectorStore::clear() {
    documents.clear();
    embeddings.clear();
}

void VectorStore::setSimilarity(std::unique_ptr<ISimilarity> sim) {
    similarity = std::move(sim);
}

std::vector<std::pair<std::string, float>> VectorStore::retrieve(const std::string& query, int topK) {
    if (documents.empty() || embeddings.empty()) {
        std::cerr << "[ERROR] retrieve() called but no documents/embeddings loaded.\n";
        return {};
    }

    auto queryVec = embeddingEngine->embed(query);
    if (queryVec.empty()) {
        std::cerr << "[ERROR] Query embedding failed! Query=\"" << query << "\"\n";
        return {};
    }
    std::cerr << "[DEBUG] Query embedding size=" << queryVec.size() 
              << ", docs=" << documents.size() << "\n";

    // Min-heap: smallest score at the top
    auto cmp = [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
        return a.second > b.second;
    };
    std::priority_queue<
        std::pair<std::string, float>,
        std::vector<std::pair<std::string, float>>,
        decltype(cmp)
    > minHeap(cmp);

    for (size_t i = 0; i < documents.size(); ++i) {
        float score = (*similarity)(queryVec, embeddings[i]);
        std::cerr << "[DEBUG] Doc " << i << " score=" << score << "\n";

        if (score < SIMILARITY_THRESHOLD) continue;

        if ((int)minHeap.size() < topK) {
            minHeap.emplace(documents[i], score);
        } else if (score > minHeap.top().second) {
            minHeap.pop();
            minHeap.emplace(documents[i], score);
        }
    }

    std::vector<std::pair<std::string, float>> results;
    while (!minHeap.empty()) {
        results.push_back(minHeap.top());
        minHeap.pop();
    }
    std::reverse(results.begin(), results.end());

    if (results.empty()) {
        std::cerr << "[WARN] No relevant results found for query=\"" << query << "\"\n";
    } else {
        std::cerr << "[DEBUG] Retrieved " << results.size() << " results.\n";
    }

    return results;
}


bool VectorStore::loadEmbeddings(const std::string& filepath) {
    try {
        std::ifstream in(filepath, std::ios::binary);
        if (!in) return false;

        // Clear existing data
        documents.clear();
        embeddings.clear();

        // Read metadata
        size_t numDocs = 0;
        in.read(reinterpret_cast<char*>(&numDocs), sizeof(numDocs));

        // Read embedding method
        int methodInt = 0;
        in.read(reinterpret_cast<char*>(&methodInt), sizeof(methodInt));
       // embeddingMethod = static_cast<EmbeddingMethod>(methodInt);

        // Read documents and embeddings
        for (size_t i = 0; i < numDocs; ++i) {
            // Read document text
            size_t textLen = 0;
            in.read(reinterpret_cast<char*>(&textLen), sizeof(textLen));
            std::string text(textLen, '\0');
            in.read(&text[0], textLen);
            documents.push_back(text);

            // Read embedding vector
            size_t embeddingSize = 0;
            in.read(reinterpret_cast<char*>(&embeddingSize), sizeof(embeddingSize));
            std::vector<float> emb(embeddingSize);
            in.read(reinterpret_cast<char*>(emb.data()), embeddingSize * sizeof(float));
            embeddings.push_back(std::move(emb));
        }

        return true;
    } catch (...) {
        return false;
    }
if (documents.size() != embeddings.size()) {
    std::cerr << "[ERROR] Mismatch: documents=" << documents.size() 
              << ", embeddings=" << embeddings.size() << "\n";
}

}


bool VectorStore::saveEmbeddings(const std::string& filepath) const {
        try {
            std::ofstream out(filepath, std::ios::binary);
            if (!out) return false;
            
            // Write metadata
            size_t numDocs = documents.size();
            out.write(reinterpret_cast<const char*>(&numDocs), sizeof(numDocs));
            
            // Write embedding method
            //out.write(reinterpret_cast<const char*>(&embeddingMethod), sizeof(embeddingMethod));
            
            // Write documents and embeddings
            for (size_t i = 0; i < numDocs; ++i) {
                size_t textLen = documents[i].length();
                out.write(reinterpret_cast<const char*>(&textLen), sizeof(textLen));
                out.write(documents[i].data(), textLen);
                
                size_t embeddingSize = embeddings[i].size();
                out.write(reinterpret_cast<const char*>(&embeddingSize), sizeof(embeddingSize));
                out.write(reinterpret_cast<const char*>(embeddings[i].data()), 
                         embeddingSize * sizeof(float));
            }
            if (documents.size() != embeddings.size()) {
    std::cerr << "[ERROR] Mismatch: documents=" << documents.size() 
              << ", embeddings=" << embeddings.size() << "\n";
}

            return true;
        } catch (...) {
            return false;
        }
    }
    
    // Memory usage estimation
    size_t VectorStore::getMemoryUsage() const {
        size_t total = 0;
        for (const auto& doc : documents) {
            total += doc.size();
        }
        for (const auto& emb : embeddings) {
            total += emb.size() * sizeof(float);
        }
        return total;
    }
    
    // Remove old documents when memory gets too large
    void VectorStore::enforceMemoryLimit(size_t maxMemoryBytes) {
        while (getMemoryUsage() > maxMemoryBytes && !documents.empty()) {
            documents.pop_back();
            embeddings.pop_back();
        }
    }
