/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — FlatVectorStore implementation wrapping existing VectorStore
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_FLAT_VECTOR_STORE_H
#define THOTH_FLAT_VECTOR_STORE_H

#include "i_vector_store.h"
#include "vector_store.h"
#include <memory>

namespace Thoth {

/**
 * @class FlatVectorStore
 * @brief Simple, in-memory vector store that persists to a binary file.
 * Wraps the existing VectorStore class to satisfy the IVectorStore interface.
 */
class FlatVectorStore : public IVectorStore {
public:
    explicit FlatVectorStore(EmbeddingEngine* engine, const std::string& persist_path = "");

    bool insert(const std::string& id, const std::vector<float>& vector) override;
    std::vector<VectorSearchResult> search(const std::vector<float>& query, int k) const override;
    bool delete_chunk(const std::string& id) override;
    bool flush() override;
    size_t chunk_count() const override;
    void clear() override;

private:
    std::unique_ptr<VectorStore> internal_store_;
    std::string persist_path_;
};

} // namespace Thoth

#endif // THOTH_FLAT_VECTOR_STORE_H
