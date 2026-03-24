/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * basic_agent - AI Agent with Memory and RAG Capabilities
 * Supports local embeddings and retrieval for C++ code.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <shared_mutex>
#include <json.hpp>

#include "embedding_engine.h"
#include "index_manager.h"
#include "config.h"
#include "grag_scorer.h"
#include "controller_event.h"

class Memory;

/**
 * @brief Coordinates retrieval logic across multiple indices.
 */
class RAGPipeline {
public:
    RAGPipeline(std::unique_ptr<EmbeddingEngine> engine, IndexManager* indexManager, Config* config = nullptr, Memory* memory = nullptr);

    // Main entry point for retrieval
    std::vector<CodeChunk> retrieveRelevant(const std::string& query, 
                                            const std::vector<int>& errorLines = {}, 
                                            int topK = 5,
                                            const std::string& requestId = "",
                                            const std::string& planId = "",
                                            const std::string& stepId = "",
                                            const std::vector<float>& g_emb = {},
                                            const std::vector<float>& c_emb = {},
                                            const std::vector<float>& t_emb = {});

    // Simple textual query
    std::string query(const std::string& queryStr);

    void clear();

    void setEventCallback(EventCallback cb) { eventCallback = cb; }

    static std::string limitText(const std::string& text, size_t maxChars);

    IndexManager* getIndexManager() { return indexManager; }
    RetrievalConfig& getRetrievalConfig() { return retrievalConfig; }

    void setGoalEmbedding(const std::vector<float>& g) { goalEmbedding = g; }
    void setCurrentEmbedding(const std::vector<float>& c) { currentEmbedding = c; }
    void setTrajectoryEmbedding(const std::vector<float>& t) { trajectoryEmbedding = t; }
    void setPlanContext(const std::string& pId, const std::string& sId) { planId = pId; stepId = sId; }

    std::unique_ptr<EmbeddingEngine> engine;
    IndexManager* indexManager;
    Config* config;
    Memory* memory;
    EventCallback eventCallback;

    RetrievalConfig retrievalConfig;
    std::vector<float> goalEmbedding;
    std::vector<float> currentEmbedding;
    std::vector<float> trajectoryEmbedding;
    std::string planId;
    std::string stepId;

    void logGragBenchmark(const std::string& requestId, 
                          const std::string& query,
                          const GragDiagnostics& diagnostics);
private:
    std::shared_mutex chunksMutex;
};
