/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * basic_agent - AI Agent with Memory and RAG Capabilities
 * uses either Ollama lacal models or OpenAI API
 *
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text
 */

#pragma once
#include <string>
#include <vector>

// Represents a chunk of code (e.g., a function, class, or code block)
struct CodeChunk {
    std::string fileName;
    std::string symbolName; // function/class name
    int startLine;
    int endLine;
    std::string code;
    std::vector<float> embedding; // new field
};

// Core RAG pipeline manager
class RAGPipeline {
public:
    RAGPipeline() = default;

    void init(const std::string& path);
    // Add a file's code chunks into the RAG store
    void indexFile(const std::string& filePath); 

    // Search for relevant chunks given a query (like an error message)
    std::vector<CodeChunk> retrieveRelevant(const std::string& query,
                                        const std::vector<int>& errorLines,
                                        int topK = 3);

    // Save the RAG index to disk (so you don’t have to rebuild each time)
    void saveIndex();
    void saveIndex(const std::string& dbPath);

    // Load an existing RAG index from disk
    void loadIndex();
    void loadIndex(const std::string& dbPath); 

    // Clear everything from memory
    void clear();

    void indexProject(const std::string& rootPath);
private:
    std::string indexFilePath;
    std::vector<CodeChunk> chunks; // in-memory storage for now
    // later: embeddings, database handles, etc.
};

