/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP-D1 rag_index.bin path retag (standalone, no IndexManager)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ALP_INDEX_MUTATOR_H
#define THOTH_ALP_INDEX_MUTATOR_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace Thoth {
namespace AlpIndexMutator {

struct ChunkRecord {
    std::string fileName;
    std::string symbolName;
    int startLine = 0;
    int endLine = 0;
    std::string code;
    std::vector<float> embedding;
    std::int64_t last_modified = 0;
    std::string commit_hash;
    int embedding_version = 0;
    float keyword_score = 0.f;
};

struct IndexSnapshot {
    std::string header_json;
    std::vector<ChunkRecord> chunks;
    std::vector<char> tail_bytes;
};

/** Load index or return empty snapshot if missing/unreadable. */
IndexSnapshot loadIndex(const std::string& index_path);

/** Apply path rewrites (normalized absolute paths). Unmapped paths unchanged. */
void applyPathRetag(IndexSnapshot& snapshot,
                    const std::unordered_map<std::string, std::string>& path_map);

/** Atomic write-temp-rename. Returns false on failure; prior file unchanged. */
bool saveIndexAtomic(const std::string& index_path, const IndexSnapshot& snapshot);

} // namespace AlpIndexMutator
} // namespace Thoth

#endif // THOTH_ALP_INDEX_MUTATOR_H
