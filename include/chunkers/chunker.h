// include/chunkers/chunker.h
#pragma once
#include "code_chunk.h"
#include <string>
#include <vector>

namespace Chunker {
    std::vector<CodeChunk> createSmartChunks(const std::string& filePath,
                                             const std::string& content);

    std::vector<CodeChunk> chunkByParagraphs(const std::string& filePath,
                                             const std::string& content);

    std::vector<CodeChunk> chunkByFunctions(const std::string& filePath,
                                            const std::string& content);

    std::vector<CodeChunk> chunkBySize(const std::string& filePath,
                                       const std::string& content);
}
