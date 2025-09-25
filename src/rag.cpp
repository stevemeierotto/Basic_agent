#include "../include/rag.h"
#include "../include/file_handler.h"
#include "../include/chunkers/chunker.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <filesystem>
#include <mutex>


namespace fs = std::filesystem;


// --- Case-insensitive search ---
static bool ci_find(const std::string &data, const std::string &toSearch) {
    auto it = std::search(
        data.begin(), data.end(),
        toSearch.begin(), toSearch.end(),
        [](char ch1, char ch2){ return std::tolower(ch1) == std::tolower(ch2); }
    );
    return it != data.end();
}

// --- RAGPipeline API ---
// Constructor
RAGPipeline::RAGPipeline(std::unique_ptr<EmbeddingEngine> eng, IndexManager* idxMgr, Config* cfg)
    : engine(std::move(eng)), store(engine.get()), indexManager(idxMgr), config(cfg) {}



// retrieveRelevant
std::vector<CodeChunk> RAGPipeline::retrieveRelevant(
    const std::string& query, 
    const std::vector<int>& errorLines, 
    int topK) 
{
    std::vector<CodeChunk> matches;

    // Read topK dynamically from Config if available
    int effectiveTopK = config ? config->max_results : topK;

    auto results = store.retrieve(query, effectiveTopK);

    std::shared_lock lock(chunksMutex);
    const auto& chunks = indexManager->getChunks();

    for (auto& [text, score] : results) {
        auto it = std::find_if(chunks.begin(), chunks.end(),
                               [&text](const CodeChunk& c){ return c.code == text; });
        if (it != chunks.end()) matches.push_back(*it);
    }
    return matches;
}

std::string RAGPipeline::query(const std::string& query) {
    auto results = store.retrieve(query, 5);
    if (results.empty()) return "[No relevant context found]";

    std::ostringstream oss;
    const auto& chunks = indexManager->getChunks();

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& [text, score] = results[i];
        auto it = std::find_if(chunks.begin(), chunks.end(),
                               [&](const CodeChunk& c) { return c.code == text; });
        if (it != chunks.end()) {
            const auto& chunk = *it;
            oss << "=== Chunk " << (i + 1) << " (score: " 
                << std::fixed << std::setprecision(3) << score << ") ===\n";
            oss << "File: " << fs::path(chunk.fileName).filename() << "\n";
            if (!chunk.symbolName.empty()) oss << "Symbol: " << chunk.symbolName << "\n";
            if (chunk.startLine > 0) oss << "Lines: " << chunk.startLine << "-" << chunk.endLine << "\n";
            oss << "Content:\n" << limitText(text, 400) << "\n\n";
        }
    }

    return oss.str();
}


void RAGPipeline::clear() {
    std::shared_lock lock(chunksMutex);
    store.clear();
    if (indexManager) indexManager->clear();  // implement clear in IndexManager
}



// ERROR 2: Fix class name typos (RagPipeline -> RAGPipeline)
std::string RAGPipeline::limitText(const std::string& text, size_t maxChars) {
    if (text.length() <= maxChars) return text;
    
    size_t cutoff = text.find_last_of(" \n\t", maxChars);
    if (cutoff == std::string::npos || cutoff < maxChars / 2) {
        cutoff = maxChars;
    }
    
    return text.substr(0, cutoff) + "...";
}
