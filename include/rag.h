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
#include "config.h"
#include "file_handler.h"
#include "index_manager.h"
#include "chunkers/chunker.h"

//#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <shared_mutex>

// Core RAG pipeline manager
class RAGPipeline {
public:
    explicit RAGPipeline(std::unique_ptr<EmbeddingEngine> eng, IndexManager* idxMgr, Config* cfg);

    std::string query(const std::string& query);

    std::vector<CodeChunk> retrieveRelevant(
        const std::string& query,
        const std::vector<int>& errorLines,
        int topK = 3);

    void clear();

    std::unique_ptr<EmbeddingEngine> engine;
    IndexManager* indexManager;
    IndexManager* getIndexManager() const { return indexManager; }

private:
    Config* config;
    mutable std::shared_mutex chunksMutex;
    //VectorStore store; // non-owning
    std::string indexFilePath;
    std::string limitText(const std::string& text, size_t maxChars);
};

