/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Read-only rag_index.bin chunk counts (ALP-D0)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ALP_INDEX_READER_H
#define THOTH_ALP_INDEX_READER_H

#include <string>
#include <unordered_map>

namespace Thoth {
namespace AlpIndexReader {

/** Map normalized absolute fileName → chunk count. Empty map if index missing/unreadable. */
std::unordered_map<std::string, int> loadChunkCountsByPath(const std::string& index_path);

} // namespace AlpIndexReader
} // namespace Thoth

#endif // THOTH_ALP_INDEX_READER_H
