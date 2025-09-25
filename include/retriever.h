#pragma once
#include "index_manager.h"
#include <vector>
#include <string>

class Retriever {
public:
    explicit Retriever(IndexManager& indexMgr);

    // Retrieve top-k relevant chunks
    std::vector<CodeChunk> retrieveRelevant(
        const std::string& query,
        const std::vector<int>& errorLines = {},
        int topK = 3);

private:
    IndexManager& indexManager;
};

