#pragma once
#include "vector_store.h"
#include "embedding_engine.h"
#include "chunkers/chunker.h"
#include "controller_event.h"
#include "json.hpp"
#include "agent_context_retrieval.h"
#include <vector>
#include <string>
#include <memory>
#include <shared_mutex>
#include <set>
#include <unordered_map>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <optional>


class IndexManager {
public:
    explicit IndexManager(EmbeddingEngine* eng);
    ~IndexManager();

    void init(const std::string& indexPath);

    // Index a single file
    void indexFile(const std::string& filePath);
    void indexFileAsync(const std::string& filePath);

    // Index all files in a directory recursively
    void indexProject(const std::string& rootPath);
    void indexProjectAsync(const std::string& rootPath);

    bool isIndexing() const { return m_isIndexing; }

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

    /** Limit retrieval to session RAG paths (files and directory roots). Empty = no filter. */
    void setActiveCorpusFiles(const std::vector<std::string>& filePaths);

    /** Phase 8 — Engine-owned corpus document list (no paths in JSON). */
    nlohmann::json listCorpusDocuments(const std::string& ragDirectory) const;

    /** Phase 9 — atomic create + async indexing (acceptance ≠ INDEXING_*). */
    struct CreateCorpusDocumentResult {
        bool ok = false;
        std::string error;
        std::string document_id;
        std::string document_name;
    };

    CreateCorpusDocumentResult createCorpusDocument(const std::string& ragDirectory,
                                                    const std::string& suggested_name,
                                                    const std::string& content,
                                                    const std::string& owner_context_id = "");

    /** R2 — record worker outcome for corpus `failed` (in-memory; Option A). */
    void recordIndexingOutcome(const std::string& normalizedPath,
                               bool success,
                               int chunk_count,
                               const std::string& reason);

    const std::string& getSessionId() const { return session_id; }

    VectorStore store;
    std::string getCurrentCommitHash() const;
    bool shouldReindexFile(const std::string& filePath);
    std::vector<std::pair<std::string,float>> retrieveChunks(const std::string& query, int topK,
                                                             const Thoth::RetrievalScope* scope = nullptr);

    /** TCB2 / TCB3 — bind ingested document to Agent Context (v1: owner = session_id). */
    void registerAttachmentOwner(const std::string& normalizedPath,
                                 const std::string& owner_context_id);

    void classifyAllChunksMetadata();

    EmbeddingEngine* getTfIdfEngine() const { return localTfIdfEngine.get(); }

    void setEventCallback(EventCallback cb) { eventCallback = cb; }
    void setSessionId(const std::string& id) { session_id = id; }

private:
        // Constants
    static constexpr size_t MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    static constexpr size_t MAX_CHUNK_SIZE = 4096; // 4KB chunks
    static constexpr size_t MAX_CHUNKS = 10000;
    static constexpr size_t MAX_TOTAL_SIZE = 100 * 1024 * 1024; // 100MB

    bool isSupportedExtension(const std::string& ext) const {
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
    EventCallback eventCallback;
    std::string session_id;

    void addChunk(CodeChunk&& chunk);
    void enforceMemoryLimits();
    std::string indexFilePath;

    // Async infrastructure
    std::atomic<bool> m_isIndexing{false};
    std::thread m_workerThread;
    std::queue<std::function<void()>> m_taskQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCv;
    std::atomic<bool> m_stopWorker{false};

    void workerLoop();
    void startWorker();

    // Helper functions
    std::string limitText(const std::string& text, size_t maxChars);
    void rebuildInternalStructures();
    void removeChunksFromPath(const std::string& rootPath);
    void removeChunksForFile(const std::string& filePath);
    size_t getCurrentMemoryUsage() const;
    int countStoredChunksForFile(const std::string& normalizedPath) const;

    std::unordered_map<std::string, std::uint64_t> indexedFileFingerprints;
    /** R2 — terminal indexing failures (normalized absolute path → reason). */
    mutable std::shared_mutex outcomesMutex_;
    std::unordered_map<std::string, std::string> indexingFailureReasons_;

    std::set<std::string> activeCorpusFiles_;
    std::vector<std::string> activeCorpusRoots_;
    std::unordered_map<std::string, std::string> attachmentOwners_;
    bool chunkInActiveCorpus(const CodeChunk& chunk) const;
    bool chunkPassesScopeFilter(const CodeChunk& chunk, const Thoth::RetrievalScope& scope) const;
    void classifyChunkInPlace(CodeChunk& chunk) const;
    void classifyAllChunksMetadataUnlocked();
    void loadAttachmentRegistry();
    void saveAttachmentRegistry() const;

    /** Session-scoped ingest: existing attachment with same filename for this owner. */
    std::optional<std::string> findSessionOwnedAttachmentPath(
        const std::string& owner_context_id,
        const std::string& base_name) const;
};

