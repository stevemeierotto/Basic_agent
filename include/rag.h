/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * basic_agent - AI Agent with Memory and RAG Capabilities
 * Supports local embeddings and retrieval for C++ code.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once
#include "vector_store.h"
#include "embedding_engine.h"
#include "file_handler.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <shared_mutex>

// Represents a chunk of code (function, class, or global block)
struct CodeChunk {
    std::string fileName;
    std::string symbolName; 
    int startLine;
    int endLine;
    std::string code;
    std::vector<float> embedding; // reserved for later
};

// Core RAG pipeline manager
class RAGPipeline {
public:
    // constructor that owns the engine
    explicit RAGPipeline(std::unique_ptr<EmbeddingEngine> engine);
    
    // Initialize RAG pipeline and load any saved index
    void init(const std::string& indexPath = "");
    
    std::string query(const std::string& query);
    
    // Index a single source file
    void indexFile(const std::string& filePath);
    
    // Index all source files in a project directory
    void indexProject(const std::string& rootPath);
    
    // Retrieve top-k relevant code chunks
    std::vector<CodeChunk> retrieveRelevant(
        const std::string& query,
        const std::vector<int>& errorLines,
        int topK = 3);
    
    // Save/load the index
    void saveIndex() const;
    void saveIndex(const std::string& dbPath) const;
    void loadIndex();
    void loadIndex(const std::string& dbPath);
    
    // Reset the in-memory store
    void clear();

private:
    mutable std::shared_mutex chunksMutex;  // protects chunks and codeToChunkIndex
    // Constants
    static constexpr size_t MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    static constexpr size_t MAX_CHUNK_SIZE = 4096; // 4KB chunks
    static constexpr size_t MAX_CHUNKS = 10000;
    static constexpr size_t MAX_TOTAL_SIZE = 100 * 1024 * 1024; // 100MB
    
    inline static const std::set<std::string> SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".epub", ".pdf", ".cpp", ".h", ".hpp", ".c"
    };
    
    // Member variables
    std::string indexFilePath;
    std::vector<CodeChunk> chunks; 
    std::unique_ptr<EmbeddingEngine> engine; // owns engine
    VectorStore store;                       // non-owning use of engine
    std::unordered_map<std::string, size_t> codeToChunkIndex;
    
    // Helper functions
    void addChunkToIndex(CodeChunk&& chunk);
    std::string limitText(const std::string& text, size_t maxChars);
    void rebuildInternalStructures();
    void removeChunksFromPath(const std::string& rootPath);
    
    bool isSupportedExtension(const std::string& ext) {
        return SUPPORTED_EXTENSIONS.find(ext) != SUPPORTED_EXTENSIONS.end();
    }
    
    // Memory management
    size_t getCurrentMemoryUsage() const;
    void enforceMemoryLimits();
    
    // Chunking strategies
    std::vector<CodeChunk> createSmartChunks(const std::string& filePath, const std::string& content);
    std::vector<CodeChunk> chunkByParagraphs(const std::string& filePath, const std::string& content);
    std::vector<CodeChunk> chunkByFunctions(const std::string& filePath, const std::string& content);
    std::vector<CodeChunk> chunkBySize(const std::string& filePath, const std::string& content);
};
