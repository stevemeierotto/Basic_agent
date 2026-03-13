#include "../include/index_manager.h"
#include "../include/file_handler.h"
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <shared_mutex>
#include <mutex>
#include <fstream>
#include <cstdint>
#include <../include/json.hpp>

using json = nlohmann::json;

namespace fs = std::filesystem;
std::string sanitize_utf8(const std::string& input) {
    std::string output;
    output.reserve(input.size());
    for (unsigned char c : input) {
        if (c < 0x80) {
            output.push_back(c);  // ASCII
        } else {
            output.push_back(' ');
        }
    }
    return output;
}

static std::uint64_t computeFileFingerprint(const std::string& filePath) {
    std::error_code ec;
    const auto size = fs::file_size(filePath, ec);
    if (ec) return 0;
    const auto writeTime = fs::last_write_time(filePath, ec);
    if (ec) return static_cast<std::uint64_t>(size);
    const auto ticks = static_cast<std::uint64_t>(writeTime.time_since_epoch().count());
    return (ticks * 1315423911ULL) ^ static_cast<std::uint64_t>(size);
}

size_t IndexManager::getCurrentMemoryUsage() const {
    size_t total = 0;
    for (const auto& c : chunks) {
        total += c.fileName.size() + c.symbolName.size() + c.code.size();
        total += sizeof(c.startLine) + sizeof(c.endLine);
        total += c.embedding.size() * sizeof(float);
    }
    return total;
}

const std::vector<CodeChunk>& IndexManager::getChunks() const {
    return chunks;
}

const CodeChunk* IndexManager::getChunkByCode(const std::string& codeText) const {
    std::shared_lock<std::shared_mutex> lock(chunksMutex);
    auto it = codeToChunkIndex.find(codeText);
    if (it != codeToChunkIndex.end() && it->second < chunks.size()) {
        return &chunks[it->second];
    }
    return nullptr;
}

void IndexManager::enforceMemoryLimits() {
    std::unique_lock lock(chunksMutex);
    if (chunks.size() > MAX_CHUNKS || getCurrentMemoryUsage() > MAX_TOTAL_SIZE) {
        size_t toRemove = chunks.size() / 5;
        chunks.erase(chunks.begin(), chunks.begin() + toRemove);
        rebuildInternalStructures();
    }
}

void IndexManager::removeChunksFromPath(const std::string& rootPath) {
    std::unique_lock lock(chunksMutex);
    std::vector<std::string> removedFiles;
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
        [&](const CodeChunk& c) { 
            const bool remove = c.fileName.rfind(rootPath, 0) == 0;
            if (remove) removedFiles.push_back(c.fileName);
            return remove; 
        }), chunks.end());
    for (const auto& file : removedFiles) indexedFileFingerprints.erase(file);
}

void IndexManager::clear() {
    std::unique_lock lock(chunksMutex);
    chunks.clear();
    codeToChunkIndex.clear();
    indexedFileFingerprints.clear();
    store.clear();
}

void IndexManager::addChunk(CodeChunk&& chunk) {
    if (chunk.code.empty()) {
        return;
    }
    
    // Phase 3.1 Hardening: Accept chunk if it has EITHER semantic embedding OR keyword signal
    bool hasSemantic = !chunk.embedding.empty() && 
                       !std::all_of(chunk.embedding.begin(), chunk.embedding.end(), [](float v){ return v == 0.0f; });
    bool hasKeyword = chunk.keyword_score > 0.001f;

    if (!hasSemantic && !hasKeyword) {
        return;
    }

    {
        std::unique_lock lock(chunksMutex);
        if (codeToChunkIndex.find(chunk.code) != codeToChunkIndex.end()) return;
        size_t index = chunks.size();
        chunks.push_back(std::move(chunk));
        store.addDocumentWithEmbedding(chunks.back().code, chunks.back().embedding);
        codeToChunkIndex[chunks.back().code] = index;
    }
}

std::string IndexManager::getCurrentCommitHash() const {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen("git rev-parse HEAD 2>/dev/null", "r");
    if (!pipe) return "unknown";
    if (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result = buffer;
        if (!result.empty() && result.back() == '\n') result.pop_back();
    }
    pclose(pipe);
    return result.empty() ? "unknown" : result;
}

bool IndexManager::shouldReindexFile(const std::string& filePath) {
    std::string normalized;
    try {
        normalized = fs::absolute(filePath).lexically_normal().string();
    } catch (...) { normalized = filePath; }

    const auto fingerprint = computeFileFingerprint(normalized);
    if (fingerprint == 0) return true;
    
    std::shared_lock lock(chunksMutex);
    const auto it = indexedFileFingerprints.find(normalized);
    if (it == indexedFileFingerprints.end()) return true;
    if (it->second != fingerprint) return true;
    
    std::string currentHash = getCurrentCommitHash();
    for (const auto& c : chunks) {
        if (c.fileName == normalized) {
            if (c.embedding_version < engine->getInternalVersion()) return true;
            if (c.commit_hash != currentHash) return true;
        }
    }
    return false;
}

void IndexManager::removeChunksForFile(const std::string& filePath) {
    std::string normalized;
    try {
        normalized = fs::absolute(filePath).lexically_normal().string();
    } catch (...) { normalized = filePath; }

    std::unique_lock lock(chunksMutex);
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
            [&](const CodeChunk& c) { return c.fileName == normalized; }),
        chunks.end());
    indexedFileFingerprints.erase(normalized);
}

void IndexManager::init(const std::string& indexPath) {
    FileHandler fh;
    if (!indexPath.empty()) {
        fs::path supplied(indexPath);
        if (fs::exists(supplied) && fs::is_directory(supplied)) {
            indexFilePath = (supplied / "rag_index.bin").string();
        } else {
            indexFilePath = supplied.string();
        }
    } else {
        indexFilePath = fh.getRagPath("rag_index.bin");
    }
    loadIndex(indexFilePath);
    rebuildInternalStructures();
}

void IndexManager::indexFile(const std::string& filePath) {
    std::string normalizedPath;
    try {
        normalizedPath = fs::absolute(filePath).lexically_normal().string();
    } catch (...) { normalizedPath = filePath; }

    // STRICT SANDBOX ENFORCEMENT
    if (normalizedPath.find("/home/steve/Thoth/agent_workspace/") == std::string::npos) {
        std::cerr << "[SECURITY] REJECTED path outside sandbox: " << normalizedPath << "\n";
        return;
    }

    if (!shouldReindexFile(normalizedPath)) {
        return;
    }
    removeChunksForFile(normalizedPath);
    
    std::ifstream in(normalizedPath, std::ios::binary);
    if (!in) return;
    std::error_code ec;
    const auto fileSize = fs::file_size(normalizedPath, ec);
    if (!ec && fileSize > MAX_FILE_SIZE) return;
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string content = ss.str();
    if (content.empty()) return;
    if (!engine) return;
    content = sanitize_utf8(content);
    
    auto chunksVec = Chunker::createSmartChunks(normalizedPath, content);
    if (chunksVec.empty()) chunksVec = Chunker::chunkBySize(normalizedPath, content);
    
    if (chunksVec.empty()) {
        CodeChunk fallbackChunk;
        fallbackChunk.fileName = normalizedPath;
        fallbackChunk.startLine = 1;
        fallbackChunk.endLine = 0;
        fallbackChunk.code = std::move(content);
        std::error_code ec_m;
        auto f_time = fs::last_write_time(normalizedPath, ec_m);
        fallbackChunk.last_modified = std::chrono::duration_cast<std::chrono::milliseconds>(f_time.time_since_epoch()).count();
        fallbackChunk.embedding_version = engine->getInternalVersion();
        fallbackChunk.commit_hash = getCurrentCommitHash();
        fallbackChunk.code.erase(std::remove(fallbackChunk.code.begin(), fallbackChunk.code.end(), '\0'), fallbackChunk.code.end());
        try { 
            fallbackChunk.embedding = engine->embed(fallbackChunk.code);
            if (localTfIdfEngine) {
                auto tfidf = localTfIdfEngine->embed(fallbackChunk.code);
                float sum = 0;
                for (float v : tfidf) sum += v * v;
                fallbackChunk.keyword_score = std::sqrt(sum);
            }
        } catch (...) {}
        addChunk(std::move(fallbackChunk));
        {
            std::unique_lock lock(chunksMutex);
            indexedFileFingerprints[normalizedPath] = computeFileFingerprint(normalizedPath);
        }
        return;
    }
    
    std::vector<std::string> codes;
    for (const auto& c : chunksVec) codes.push_back(c.code);
    auto embeddings = engine->embedBatch(codes);

    for (size_t i = 0; i < chunksVec.size(); ++i) {
        CodeChunk &chunkRef = chunksVec[i];
        if (chunkRef.code.empty() || std::all_of(chunkRef.code.begin(), chunkRef.code.end(), ::isspace)) continue;
        chunkRef.code.erase(std::remove(chunkRef.code.begin(), chunkRef.code.end(), '\0'), chunkRef.code.end());
        size_t nonspace_count = std::count_if(chunkRef.code.begin(), chunkRef.code.end(), [](char c){ return !std::isspace(c); });
        if (nonspace_count < 10) continue;
        try {
            if (i < embeddings.size()) chunkRef.embedding = embeddings[i];
            else chunkRef.embedding = engine->embed(chunkRef.code);

            std::error_code ec_m;
            auto f_time = fs::last_write_time(normalizedPath, ec_m);
            chunkRef.last_modified = std::chrono::duration_cast<std::chrono::milliseconds>(f_time.time_since_epoch()).count();
            chunkRef.embedding_version = engine->getInternalVersion();
            chunkRef.commit_hash = getCurrentCommitHash();

            if (localTfIdfEngine) {
                auto tfidf = localTfIdfEngine->embed(chunkRef.code);
                float sum = 0;
                for (float v : tfidf) sum += v * v;
                chunkRef.keyword_score = std::sqrt(sum);
            }

            bool isZero = std::all_of(chunkRef.embedding.begin(), chunkRef.embedding.end(), [](float v){ return v == 0.0f; });
            if (isZero && chunkRef.keyword_score < 0.001f) continue;
        } catch (...) { continue; }
        addChunk(std::move(chunkRef));
    }
    {
        std::unique_lock lock(chunksMutex);
        indexedFileFingerprints[normalizedPath] = computeFileFingerprint(normalizedPath);
    }
}

void IndexManager::indexProject(const std::string& rootPath) {
    if (!fs::exists(rootPath) || !fs::is_directory(rootPath)) return;
    std::string rootAbs = fs::absolute(rootPath).lexically_normal().string();

    // STRICT SANDBOX ENFORCEMENT
    if (rootAbs.find("/home/steve/Thoth/agent_workspace/") == std::string::npos) {
        std::cerr << "[SECURITY] REJECTED rootPath outside sandbox: " << rootAbs << "\n";
        return;
    }

    int successCount = 0, errorCount = 0, skippedCount = 0;
    constexpr size_t MAX_FILES_TO_INDEX = 2000;
    size_t filesProcessed = 0;
    
    std::cout << "[RAG] Starting recursive index of: " << rootAbs << "\n";

    try {
        for (const auto& entry : fs::recursive_directory_iterator(rootPath)) {
            if (filesProcessed >= MAX_FILES_TO_INDEX) break;
            if (!entry.is_regular_file()) continue;
            
            std::string fullPath = fs::absolute(entry.path()).lexically_normal().string();
            std::string filename = entry.path().filename().string();
            
            if (filename == "rag_index.bin") continue;
            if (fullPath.find("/build/") != std::string::npos) continue;
            if (fullPath.find("/.git/") != std::string::npos) continue;
            if (fullPath.find("/docs/") != std::string::npos) continue;

            auto ext = entry.path().extension().string();
            if (isSupportedExtension(ext)) {
                try {
                    if (!shouldReindexFile(fullPath)) { 
                        skippedCount++; 
                        continue; 
                    }
                    indexFile(fullPath);
                    successCount++;
                    filesProcessed++;
                    if (filesProcessed % 50 == 0) {
                        std::cout << "[RAG] ... processed " << filesProcessed << " files\n";
                    }
                } catch (...) { errorCount++; }
            }
        }
    } catch (const std::exception& e) { 
        std::cerr << "[RAG] Critical error during indexProject: " << e.what() << "\n";
    }
    
    std::cout << "[RAG] Indexed " << rootAbs << " - Success: " << successCount << ", Skipped: " << skippedCount << ", Errors: " << errorCount << "\n";
}

void IndexManager::saveIndex() const {
    FileHandler fh;
    saveIndex(fh.getRagPath("rag_index.bin"));
}

void IndexManager::saveIndex(const std::string& dbPath) const {
    std::filesystem::create_directories(std::filesystem::path(dbPath).parent_path());
    std::ofstream out(dbPath, std::ios::binary | std::ios::trunc);
    if (!out) return;

    json header;
    header["magic"] = 0x54484F54;
    header["model_name"] = engine->getModelName();
    header["embedding_dimension"] = engine->getDimension();
    header["embedding_version"] = engine->getInternalVersion();
    std::string headerStr = header.dump();
    size_t headerLen = headerStr.size();
    out.write(reinterpret_cast<const char*>(&headerLen), sizeof(headerLen));
    out.write(headerStr.data(), headerLen);

    size_t n = chunks.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto& c : chunks) {
        size_t len;
        len = c.fileName.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len)); out.write(c.fileName.data(), len);
        len = c.symbolName.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len)); out.write(c.symbolName.data(), len);
        out.write(reinterpret_cast<const char*>(&c.startLine), sizeof(c.startLine));
        out.write(reinterpret_cast<const char*>(&c.endLine), sizeof(c.endLine));
        len = c.code.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len)); out.write(c.code.data(), len);
        len = c.embedding.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        if (len > 0) out.write(reinterpret_cast<const char*>(c.embedding.data()), len * sizeof(float));
        out.write(reinterpret_cast<const char*>(&c.last_modified), sizeof(c.last_modified));
        len = c.commit_hash.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        if (len > 0) out.write(c.commit_hash.data(), len);
        out.write(reinterpret_cast<const char*>(&c.embedding_version), sizeof(c.embedding_version));
        out.write(reinterpret_cast<const char*>(&c.keyword_score), sizeof(c.keyword_score));
    }
    
    // Save Primary Engine State
    {
        std::string tmpFile = dbPath + ".engine_tmp";
        engine->saveState(tmpFile);
        std::ifstream engIn(tmpFile, std::ios::binary);
        std::string engData((std::istreambuf_iterator<char>(engIn)), std::istreambuf_iterator<char>());
        size_t engSize = engData.size();
        out.write(reinterpret_cast<const char*>(&engSize), sizeof(engSize));
        out.write(engData.data(), engSize);
        std::filesystem::remove(tmpFile);
    }

    // Phase 13 Fix: Save Local TF-IDF Engine State
    if (localTfIdfEngine) {
        std::string tmpFile = dbPath + ".tfidf_tmp";
        localTfIdfEngine->saveState(tmpFile);
        std::ifstream engIn(tmpFile, std::ios::binary);
        std::string engData((std::istreambuf_iterator<char>(engIn)), std::istreambuf_iterator<char>());
        size_t engSize = engData.size();
        out.write(reinterpret_cast<const char*>(&engSize), sizeof(engSize));
        out.write(engData.data(), engSize);
        std::filesystem::remove(tmpFile);
    } else {
        size_t zero = 0;
        out.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    }

    std::cout << "[basic_agent:RAG] Index saved to: " << dbPath << " (entries=" << n << ")\n";
} 

void IndexManager::loadIndex() {
    FileHandler fh;
    loadIndex(fh.getRagPath("rag_index.bin"));
}

void IndexManager::loadIndex(const std::string& dbPath) {
    std::ifstream in(dbPath, std::ios::binary);
    if (!in) return;

    size_t headerLen = 0;
    in.read(reinterpret_cast<char*>(&headerLen), sizeof(headerLen));
    if (headerLen > 0 && headerLen < 1024*1024) {
        std::string headerStr(headerLen, '\0');
        in.read(&headerStr[0], headerLen);
        try {
            json header = json::parse(headerStr);
            bool mismatch = false;
            if (header.value("model_name", "") != engine->getModelName()) mismatch = true;
            if (header.value("embedding_dimension", 0) != engine->getDimension()) mismatch = true;
            if (header.value("embedding_version", 0) != engine->getInternalVersion()) mismatch = true;

            if (mismatch) {
                std::cout << "[RAG] Index metadata mismatch detected.\n";
                return;
            }
        } catch (...) {}
    }

    size_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    {
        std::unique_lock lock(chunksMutex);
        chunks.clear(); 
        codeToChunkIndex.clear();
        chunks.reserve(n);
    }
    for (size_t i = 0; i < n; ++i) {
        CodeChunk c; size_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len)); c.fileName.resize(len); in.read(&c.fileName[0], len);
        in.read(reinterpret_cast<char*>(&len), sizeof(len)); c.symbolName.resize(len); in.read(&c.symbolName[0], len);
        in.read(reinterpret_cast<char*>(&c.startLine), sizeof(c.startLine));
        in.read(reinterpret_cast<char*>(&c.endLine), sizeof(c.endLine));
        in.read(reinterpret_cast<char*>(&len), sizeof(len)); c.code.resize(len); in.read(&c.code[0], len);
        size_t embLen; in.read(reinterpret_cast<char*>(&embLen), sizeof(embLen)); c.embedding.resize(embLen);
        if (embLen > 0) in.read(reinterpret_cast<char*>(c.embedding.data()), embLen * sizeof(float));
        in.read(reinterpret_cast<char*>(&c.last_modified), sizeof(c.last_modified));
        size_t hashLen; in.read(reinterpret_cast<char*>(&hashLen), sizeof(hashLen));
        if (hashLen > 0) { c.commit_hash.resize(hashLen); in.read(&c.commit_hash[0], hashLen); }
        in.read(reinterpret_cast<char*>(&c.embedding_version), sizeof(c.embedding_version));
        in.read(reinterpret_cast<char*>(&c.keyword_score), sizeof(c.keyword_score));
        try { c.fileName = fs::absolute(c.fileName).lexically_normal().string(); } catch (...) {}
        {
            std::unique_lock lock(chunksMutex);
            codeToChunkIndex[c.code] = chunks.size();
            chunks.push_back(std::move(c));
            indexedFileFingerprints[chunks.back().fileName] = computeFileFingerprint(chunks.back().fileName);
        }
    }
    
    // Load Primary Engine State
    size_t engSize;
    in.read(reinterpret_cast<char*>(&engSize), sizeof(engSize));
    if (engSize > 0) {
        std::string engData(engSize, '\0');
        in.read(&engData[0], engSize);
        std::string tmpFile = dbPath + ".engine_tmp";
        { std::ofstream tmpOut(tmpFile, std::ios::binary); tmpOut.write(engData.data(), engSize); }
        engine->loadState(tmpFile);
        std::filesystem::remove(tmpFile);
    }

    // Phase 13 Fix: Load Local TF-IDF Engine State
    size_t tfidfSize;
    in.read(reinterpret_cast<char*>(&tfidfSize), sizeof(tfidfSize));
    if (tfidfSize > 0 && localTfIdfEngine) {
        std::string engData(tfidfSize, '\0');
        in.read(&engData[0], tfidfSize);
        std::string tmpFile = dbPath + ".tfidf_tmp";
        { std::ofstream tmpOut(tmpFile, std::ios::binary); tmpOut.write(engData.data(), tfidfSize); }
        localTfIdfEngine->loadState(tmpFile);
        std::filesystem::remove(tmpFile);
    }

    {
        std::unique_lock lock(chunksMutex);
        store.clear();
        for (size_t i = 0; i < chunks.size(); ++i) {
            auto& c = chunks[i];
            if (!c.embedding.empty() && !c.code.empty()) {
                store.addDocumentWithEmbedding(c.code, c.embedding);
            }
        }
    }
    std::cout << "[basic_agent:RAG] Index loaded from: " << dbPath << " (entries=" << n << ")\n";
}

void IndexManager::rebuildInternalStructures() {
    std::unique_lock lock(chunksMutex);
    store.clear();
    codeToChunkIndex.clear();
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        if (chunk.code.empty() || chunk.embedding.empty()) continue;
        if (codeToChunkIndex.find(chunk.code) != codeToChunkIndex.end()) continue;
        store.addDocumentWithEmbedding(chunk.code, chunk.embedding);
        codeToChunkIndex[chunk.code] = i;
    }
}

void IndexManager::addChunkToIndex(CodeChunk&& chunk) {
    addChunk(std::move(chunk));
}
