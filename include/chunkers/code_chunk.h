#pragma once
#include <string>
#include <vector>


// Represents a chunk of code (function, class, or global block)
struct CodeChunk {
    std::string fileName;
    std::string symbolName; 
    int startLine;
    int endLine;
    std::string code;
    std::vector<float> embedding;

    // Phase 6 Integrity Metadata
    int64_t last_modified = 0;
    std::string commit_hash;
    int embedding_version = 1;

    // Phase 3.1: Hybrid Reranking
    float keyword_score = 0.0f; // TF-IDF keyword relevance

    // TCB2 — runtime classification (not persisted in rag_index.bin v1)
    std::string corpus_tier;
    std::string owner_context_id;
};
