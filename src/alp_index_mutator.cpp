/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP-D1 rag_index.bin path retag (standalone, no IndexManager)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "alp_index_mutator.h"

#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace fs = std::filesystem;

namespace Thoth {
namespace AlpIndexMutator {
namespace {

std::string normalizePath(const fs::path& p) {
    try {
        return fs::absolute(p).lexically_normal().string();
    } catch (...) {
        return p.string();
    }
}

bool readExact(std::istream& in, void* buf, std::size_t n) {
    in.read(static_cast<char*>(buf), static_cast<std::streamsize>(n));
    return static_cast<std::size_t>(in.gcount()) == n;
}

} // namespace

IndexSnapshot loadIndex(const std::string& index_path) {
    IndexSnapshot snap;
    std::ifstream in(index_path, std::ios::binary);
    if (!in) {
        return snap;
    }

    const auto tail_start = [&]() -> std::streampos {
        in.seekg(0, std::ios::end);
        return in.tellg();
    }();

    in.seekg(0, std::ios::beg);

    std::size_t headerLen = 0;
    if (!readExact(in, &headerLen, sizeof(headerLen))) {
        return IndexSnapshot{};
    }
    if (headerLen > 0 && headerLen < 1024 * 1024) {
        snap.header_json.resize(headerLen);
        if (!readExact(in, snap.header_json.data(), headerLen)) {
            return IndexSnapshot{};
        }
    }

    std::size_t n = 0;
    if (!readExact(in, &n, sizeof(n)) || n > 10'000'000) {
        return IndexSnapshot{};
    }

    snap.chunks.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        ChunkRecord c;
        std::size_t len = 0;

        if (!readExact(in, &len, sizeof(len)) || len > 1'000'000) {
            return IndexSnapshot{};
        }
        c.fileName.resize(len);
        if (!readExact(in, c.fileName.data(), len)) {
            return IndexSnapshot{};
        }

        if (!readExact(in, &len, sizeof(len)) || len > 1'000'000) {
            return IndexSnapshot{};
        }
        c.symbolName.resize(len);
        if (!readExact(in, c.symbolName.data(), len)) {
            return IndexSnapshot{};
        }

        if (!readExact(in, &c.startLine, sizeof(c.startLine))
            || !readExact(in, &c.endLine, sizeof(c.endLine))) {
            return IndexSnapshot{};
        }

        if (!readExact(in, &len, sizeof(len)) || len > 100'000'000) {
            return IndexSnapshot{};
        }
        c.code.resize(len);
        if (len > 0 && !readExact(in, c.code.data(), len)) {
            return IndexSnapshot{};
        }

        std::size_t embLen = 0;
        if (!readExact(in, &embLen, sizeof(embLen))) {
            return IndexSnapshot{};
        }
        c.embedding.resize(embLen);
        if (embLen > 0
            && !readExact(in, c.embedding.data(), embLen * sizeof(float))) {
            return IndexSnapshot{};
        }

        if (!readExact(in, &c.last_modified, sizeof(c.last_modified))) {
            return IndexSnapshot{};
        }

        std::size_t hashLen = 0;
        if (!readExact(in, &hashLen, sizeof(hashLen))) {
            return IndexSnapshot{};
        }
        if (hashLen > 1'000'000) {
            return IndexSnapshot{};
        }
        c.commit_hash.resize(hashLen);
        if (hashLen > 0 && !readExact(in, c.commit_hash.data(), hashLen)) {
            return IndexSnapshot{};
        }

        if (!readExact(in, &c.embedding_version, sizeof(c.embedding_version))
            || !readExact(in, &c.keyword_score, sizeof(c.keyword_score))) {
            return IndexSnapshot{};
        }

        try {
            c.fileName = normalizePath(fs::path(c.fileName));
        } catch (...) {
        }
        snap.chunks.push_back(std::move(c));
    }

    const auto pos = in.tellg();
    if (pos >= 0 && pos < tail_start) {
        in.seekg(pos, std::ios::beg);
        snap.tail_bytes.assign(std::istreambuf_iterator<char>(in),
                               std::istreambuf_iterator<char>());
    }

    return snap;
}

void applyPathRetag(IndexSnapshot& snapshot,
                    const std::unordered_map<std::string, std::string>& path_map) {
    for (auto& chunk : snapshot.chunks) {
        const auto it = path_map.find(chunk.fileName);
        if (it != path_map.end()) {
            chunk.fileName = it->second;
        }
    }
}

bool saveIndexAtomic(const std::string& index_path, const IndexSnapshot& snapshot) {
    if (snapshot.header_json.empty() && snapshot.chunks.empty()) {
        return true;
    }

    const fs::path parent = fs::path(index_path).parent_path();
    std::error_code ec;
    fs::create_directories(parent, ec);

    const std::string temp_path =
        index_path + ".tmp." + std::to_string(static_cast<unsigned long>(getpid()));

    try {
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            return false;
        }

        const std::size_t headerLen = snapshot.header_json.size();
        out.write(reinterpret_cast<const char*>(&headerLen), sizeof(headerLen));
        if (headerLen > 0) {
            out.write(snapshot.header_json.data(),
                      static_cast<std::streamsize>(headerLen));
        }

        const std::size_t n = snapshot.chunks.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& c : snapshot.chunks) {
            std::size_t len = c.fileName.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(c.fileName.data(), static_cast<std::streamsize>(len));

            len = c.symbolName.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(c.symbolName.data(), static_cast<std::streamsize>(len));

            out.write(reinterpret_cast<const char*>(&c.startLine), sizeof(c.startLine));
            out.write(reinterpret_cast<const char*>(&c.endLine), sizeof(c.endLine));

            len = c.code.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(c.code.data(), static_cast<std::streamsize>(len));

            len = c.embedding.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            if (len > 0) {
                out.write(reinterpret_cast<const char*>(c.embedding.data()),
                          static_cast<std::streamsize>(len * sizeof(float)));
            }

            out.write(reinterpret_cast<const char*>(&c.last_modified),
                      sizeof(c.last_modified));

            len = c.commit_hash.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            if (len > 0) {
                out.write(c.commit_hash.data(), static_cast<std::streamsize>(len));
            }

            out.write(reinterpret_cast<const char*>(&c.embedding_version),
                      sizeof(c.embedding_version));
            out.write(reinterpret_cast<const char*>(&c.keyword_score),
                      sizeof(c.keyword_score));
        }

        if (!snapshot.tail_bytes.empty()) {
            out.write(snapshot.tail_bytes.data(),
                      static_cast<std::streamsize>(snapshot.tail_bytes.size()));
        }

        if (!out) {
            fs::remove(temp_path, ec);
            return false;
        }
        out.close();

        fs::rename(temp_path, index_path, ec);
        if (ec) {
            fs::remove(temp_path, ec);
            return false;
        }
        return true;
    } catch (...) {
        fs::remove(temp_path, ec);
        return false;
    }
}

} // namespace AlpIndexMutator
} // namespace Thoth
