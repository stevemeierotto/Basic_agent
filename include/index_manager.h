#pragma once
#include "vector_store.h"
#include "embedding_engine.h"
#include "chunkers/chunker.h"
#include <vector>
#include <string>
#include <memory>
#include <shared_mutex>
#include <set>
#include <unordered_map>


class IndexManager {
public:
        explicit IndexManager(EmbeddingEngine* eng)
        : store(eng) ,engine(eng), localTfIdfEngine(std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf)) {}

    void init(const std::string& indexPath);



    // Index a single file
    void indexFile(const std::string& filePath);

    // Index all files in a directory recursively
    void indexProject(const std::string& rootPath);

    // Access indexed chunks
    const std::vector<CodeChunk>& getChunks() const;
    
    // Get chunk by code text using hash map (O(1) lookup)
    // Returns nullptr if not found, otherwise returns pointer to chunk
    const CodeChunk* getChunkByCode(const std::string& codeText) const;

    // Save/load the index
    void saveIndex() const;
    void saveIndex(const std::string& dbPath) const;
    void loadIndex();
    void loadIndex(const std::string& dbPath);

    void clear();
    void addChunkToIndex(CodeChunk&& chunk);
    VectorStore store;
    std::string getCurrentCommitHash() const;
    bool shouldReindexFile(const std::string& filePath);
    std::vector<std::pair<std::string,float>> retrieveChunks(const std::string& query, int topK) {
        return store.retrieve(query, topK);
    }

    EmbeddingEngine* getTfIdfEngine() const { return localTfIdfEngine.get(); }

private:
        // Constants
    static constexpr size_t MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    static constexpr size_t MAX_CHUNK_SIZE = 4096; // 4KB chunks
    static constexpr size_t MAX_CHUNKS = 10000;
    static constexpr size_t MAX_TOTAL_SIZE = 100 * 1024 * 1024; // 100MB

    bool isSupportedExtension(const std::string& ext) {
        return SUPPORTED_EXTENSIONS.find(ext) != SUPPORTED_EXTENSIONS.end();
    }

    std::unordered_map<std::string, size_t> codeToChunkIndex;

    inline static const std::set<std::string> SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".epub", ".pdf", ".cpp", ".h", ".hpp", ".c"
    };

    std::vector<CodeChunk> chunks;
    EmbeddingEngine* engine;
    std::unique_ptr<EmbeddingEngine> localTfIdfEngine;
    mutable std::shared_mutex chunksMutex;

    void addChunk(CodeChunk&& chunk);
    void enforceMemoryLimits();
    std::string indexFilePath;

    // Helper functions
    std::string limitText(const std::string& text, size_t maxChars);
    void rebuildInternalStructures();
    void removeChunksFromPath(const std::string& rootPath);
    void removeChunksForFile(const std::string& filePath);
    size_t getCurrentMemoryUsage() const;

    std::unordered_map<std::string, std::uint64_t> indexedFileFingerprints;
};

