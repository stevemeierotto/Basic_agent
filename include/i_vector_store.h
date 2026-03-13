/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — IVectorStore interface for swappable vector backends
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_I_VECTOR_STORE_H
#define THOTH_I_VECTOR_STORE_H

#include <vector>
#include <string>
#include <optional>

namespace Thoth {

/**
 * @struct VectorSearchResult
 * @brief Represents a single hit from a vector search.
 */
struct VectorSearchResult {
    std::string id;
    float score;
    std::vector<float> vector;
};

/**
 * @interface IVectorStore
 * @brief Abstract interface for vector storage and retrieval.
 */
class IVectorStore {
public:
    virtual ~IVectorStore() = default;

    /**
     * @brief Inserts or updates a vector in the store.
     */
    virtual bool insert(const std::string& id, const std::vector<float>& vector) = 0;

    /**
     * @brief Performs a similarity search for the top K nearest neighbors.
     */
    virtual std::vector<VectorSearchResult> search(const std::vector<float>& query, int k) const = 0;

    /**
     * @brief Deletes a vector from the store.
     */
    virtual bool delete_chunk(const std::string& id) = 0;

    /**
     * @brief Persists any in-memory state to disk.
     */
    virtual bool flush() = 0;

    /**
     * @brief Returns the total number of vectors in the store.
     */
    virtual size_t chunk_count() const = 0;

    /**
     * @brief Clears all vectors from the store.
     */
    virtual void clear() = 0;
};

} // namespace Thoth

#endif // THOTH_I_VECTOR_STORE_H
