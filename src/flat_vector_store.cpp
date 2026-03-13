#include "../include/flat_vector_store.h"
#include <algorithm>
#include <cmath>

namespace Thoth {

FlatVectorStore::FlatVectorStore(EmbeddingEngine* engine, const std::string& persist_path)
    : internal_store_(std::make_unique<VectorStore>(engine)), persist_path_(persist_path) {}

bool FlatVectorStore::insert(const std::string& id, const std::vector<float>& vec) {
    internal_store_->addDocumentWithEmbedding(id, vec);
    return true;
}

std::vector<VectorSearchResult> FlatVectorStore::search(const std::vector<float>& query_embedding, int top_k) const {
    if (internal_store_->getEmbeddings().empty()) return {};

    struct ScoredIndex {
        size_t index;
        float score;
    };

    std::vector<ScoredIndex> scored_indices;
    scored_indices.reserve(internal_store_->getEmbeddings().size());

    const auto& embeddings = internal_store_->getEmbeddings();
    for (size_t i = 0; i < embeddings.size(); ++i) {
        const auto& emb = embeddings[i];
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        for (size_t j = 0; j < std::min(query_embedding.size(), emb.size()); ++j) {
            dot += query_embedding[j] * emb[j];
            norm_a += query_embedding[j] * query_embedding[j];
            norm_b += emb[j] * emb[j];
        }
        float score = (norm_a > 0 && norm_b > 0) ? (dot / (std::sqrt(norm_a) * std::sqrt(norm_b))) : 0.0f;
        scored_indices.push_back({i, score});
    }

    std::sort(scored_indices.begin(), scored_indices.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });

    std::vector<VectorSearchResult> results;
    int count = std::min(top_k, (int)scored_indices.size());
    const auto& docs = internal_store_->getDocuments();
    for (int i = 0; i < count; ++i) {
        size_t idx = scored_indices[i].index;
        VectorSearchResult res;
        res.id = docs[idx];
        res.score = scored_indices[i].score;
        res.vector = embeddings[idx];
        results.push_back(res);
    }

    return results;
}

bool FlatVectorStore::delete_chunk(const std::string& /*id*/) {
    // Basic implementation doesn't support random delete yet
    return false;
}

bool FlatVectorStore::flush() {
    if (persist_path_.empty()) return false;
    return internal_store_->saveEmbeddings(persist_path_);
}

void FlatVectorStore::clear() {
    internal_store_->clear();
}

size_t FlatVectorStore::chunk_count() const {
    return internal_store_->getEmbeddings().size();
}

} // namespace Thoth
