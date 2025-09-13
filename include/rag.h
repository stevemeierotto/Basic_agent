/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * basic_agent - AI Agent with Memory and RAG Capabilities
 * Supports local embeddings and retrieval for C++ code.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once
#include <string>
#include <vector>
#include "../include/vector_store.h"

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
    RAGPipeline() = default;

    // Initialize RAG pipeline and load any saved index
    void init(const std::string& indexPath);
    void init();

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
    std::string indexFilePath;
    std::vector<CodeChunk> chunks; 
    VectorStore store;
};

