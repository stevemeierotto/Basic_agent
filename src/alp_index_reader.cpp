/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Read-only rag_index.bin chunk counts (ALP-D0)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "alp_index_reader.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace Thoth {
namespace AlpIndexReader {

std::unordered_map<std::string, int> loadChunkCountsByPath(const std::string& index_path) {
    std::unordered_map<std::string, int> counts;
    std::ifstream in(index_path, std::ios::binary);
    if (!in) {
        return counts;
    }

    size_t headerLen = 0;
    in.read(reinterpret_cast<char*>(&headerLen), sizeof(headerLen));
    if (headerLen > 0 && headerLen < 1024 * 1024) {
        in.seekg(static_cast<std::streamoff>(headerLen), std::ios::cur);
    }

    size_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in || n > 10'000'000) {
        return counts;
    }

    for (size_t i = 0; i < n; ++i) {
        size_t len = 0;
        std::string fileName;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!in || len > 1'000'000) {
            break;
        }
        fileName.resize(len);
        in.read(fileName.data(), static_cast<std::streamsize>(len));

        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!in || len > 1'000'000) {
            break;
        }
        in.seekg(static_cast<std::streamoff>(len), std::ios::cur);

        in.seekg(static_cast<std::streamoff>(sizeof(int) * 2), std::ios::cur);

        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!in || len > 100'000'000) {
            break;
        }
        in.seekg(static_cast<std::streamoff>(len), std::ios::cur);

        size_t embLen = 0;
        in.read(reinterpret_cast<char*>(&embLen), sizeof(embLen));
        if (!in) {
            break;
        }
        in.seekg(static_cast<std::streamoff>(embLen * sizeof(float)), std::ios::cur);

        in.seekg(static_cast<std::streamoff>(sizeof(int64_t)), std::ios::cur);

        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!in) {
            break;
        }
        in.seekg(static_cast<std::streamoff>(len), std::ios::cur);

        in.seekg(static_cast<std::streamoff>(sizeof(int) + sizeof(float)), std::ios::cur);

        try {
            fileName = fs::absolute(fileName).lexically_normal().string();
        } catch (...) {
        }
        counts[fileName]++;
    }

    return counts;
}

} // namespace AlpIndexReader
} // namespace Thoth
