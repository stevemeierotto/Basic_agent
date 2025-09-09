#include "rag.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// --- Helper: split file into chunks (function + optional line-based fallback) ---
static std::vector<CodeChunk> parseChunks(const std::string& filePath, const std::string& content) {
    std::vector<CodeChunk> chunks;
    std::istringstream stream(content);
    std::string line;
    CodeChunk current;
    int lineNum = 0;
    bool inFunc = false;
    int braceCount = 0;
    std::regex funcRegex(R"(\s*[\w:<>\*&]+\s+(\w+)\s*\(.*\)\s*\{)");

    current.startLine = 1;
    current.fileName = filePath;
    current.symbolName = "";
    current.code = "";

    while (std::getline(stream, line)) {
        lineNum++;

        if (!inFunc) {
            std::smatch match;
            if (std::regex_search(line, match, funcRegex)) {
                if (!current.code.empty()) {
                    current.endLine = lineNum - 1;
                    chunks.push_back(current);
                }
                current = {};
                current.fileName = filePath;
                current.startLine = lineNum;
                current.symbolName = match[1];
                current.code = line + "\n";
                inFunc = true;
                braceCount = std::count(line.begin(), line.end(), '{') -
                             std::count(line.begin(), line.end(), '}');
            } else {
                // fallback: single-line chunk for global code
                CodeChunk singleLineChunk;
                singleLineChunk.fileName = filePath;
                singleLineChunk.startLine = lineNum;
                singleLineChunk.endLine = lineNum;
                singleLineChunk.symbolName = "";
                singleLineChunk.code = line + "\n";
                chunks.push_back(singleLineChunk);
            }
        } else {
            current.code += line + "\n";
            braceCount += std::count(line.begin(), line.end(), '{');
            braceCount -= std::count(line.begin(), line.end(), '}');

            if (braceCount == 0) {
                current.endLine = lineNum;
                chunks.push_back(current);
                current = {};
                current.fileName = filePath;
                current.startLine = lineNum + 1;
                inFunc = false;
            }
        }
    }

    if (!current.code.empty()) {
        current.endLine = lineNum;
        chunks.push_back(current);
    }

    return chunks;
}

// --- Case-insensitive find ---
bool ci_find(const std::string &data, const std::string &toSearch) {
    auto it = std::search(
        data.begin(), data.end(),
        toSearch.begin(), toSearch.end(),
        [](char ch1, char ch2){ return std::tolower(ch1) == std::tolower(ch2); }
    );
    return it != data.end();
}

// --- API Implementations ---
void RAGPipeline::init(const std::string& path) {
    indexFilePath = path;
    loadIndex(indexFilePath);
}

void RAGPipeline::indexFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "[RAG] Failed to open file: " << filePath << std::endl;
        return;
    }

    if (filePath.find("/build/") != std::string::npos ||
        filePath.find("json.hpp") != std::string::npos) {
        return;
    }

    // Remove old chunks for this file
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
        [&](const CodeChunk& c) { return c.fileName == filePath; }),
        chunks.end());

    std::stringstream buffer;
    buffer << file.rdbuf();
    auto newChunks = parseChunks(filePath, buffer.str());
    chunks.insert(chunks.end(), newChunks.begin(), newChunks.end());
}

// --- Line-focused retrieval ---
std::vector<CodeChunk> RAGPipeline::retrieveRelevant(const std::string& query, 
                                                     const std::vector<int>& errorLines,
                                                     int topK) {
    // Tokenize query
    std::vector<std::string> qtokens;
    std::istringstream qstream(query);
    std::string tok;
    while (qstream >> tok) {
        if (tok.size() >= 2) qtokens.push_back(tok);
    }

    std::vector<CodeChunk> results;
    if (qtokens.empty() || chunks.empty() || topK == 0) return results;
    if (topK < 0) topK = static_cast<int>(chunks.size());

    struct ScoredIdx { size_t idx; int score; };
    std::vector<ScoredIdx> scored;
    scored.reserve(chunks.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& c = chunks[i];

        // skip chunks that don't overlap error lines
        bool overlap = false;
        for (int el : errorLines) {
            if (el >= c.startLine && el <= c.endLine) { overlap = true; break; }
        }
        if (!overlap) continue;

        int score = 0;
        for (const auto& tok : qtokens) {
            if (ci_find(c.code, tok)) score += 1;
            if (!c.symbolName.empty() && ci_find(c.symbolName, tok)) score += 5;
            if (!c.fileName.empty() && ci_find(c.fileName, tok)) score += 1;
        }

        if (score > 0) scored.push_back({i, score});
    }

    std::sort(scored.begin(), scored.end(),
              [](const ScoredIdx& a, const ScoredIdx& b) { return a.score > b.score; });

    results.reserve(std::min<int>(topK, static_cast<int>(scored.size())));
    for (int k = 0; k < topK && k < static_cast<int>(scored.size()); ++k) {
        results.push_back(chunks[scored[k].idx]);
    }

    std::cout << "[RAG] retrieveRelevant for query: \"" << query
              << "\" returned " << results.size() << " chunk(s)."
              << " (searched " << chunks.size() << " chunks)" << std::endl;

    return results;
}

// --- The rest of your RAG API (save/load, clear, indexProject) stays unchanged ---
void RAGPipeline::saveIndex() { saveIndex(indexFilePath); }

void RAGPipeline::saveIndex(const std::string& dbPath) {
    std::ofstream out(dbPath, std::ios::binary | std::ios::trunc);
    if (!out) { std::cerr << "[RAG] Failed to open " << dbPath << " for writing.\n"; return; }
    size_t n = chunks.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto& c : chunks) {
        size_t len;
        len = c.fileName.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(c.fileName.data(), len);
        len = c.symbolName.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(c.symbolName.data(), len);
        out.write(reinterpret_cast<const char*>(&c.startLine), sizeof(c.startLine));
        out.write(reinterpret_cast<const char*>(&c.endLine), sizeof(c.endLine));
        len = c.code.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(c.code.data(), len);
    }
}

void RAGPipeline::loadIndex() { loadIndex(indexFilePath); }

void RAGPipeline::loadIndex(const std::string& dbPath) {
    std::ifstream in(dbPath, std::ios::binary);
    if (!in) { std::cerr << "[RAG] No index found at " << dbPath << " (starting fresh).\n"; return; }
    size_t n; in.read(reinterpret_cast<char*>(&n), sizeof(n));
    chunks.clear(); chunks.reserve(n);
    for (size_t i = 0; i < n; i++) {
        CodeChunk c; size_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len)); c.fileName.resize(len);
        in.read(&c.fileName[0], len);
        in.read(reinterpret_cast<char*>(&len), sizeof(len)); c.symbolName.resize(len);
        in.read(&c.symbolName[0], len);
        in.read(reinterpret_cast<char*>(&c.startLine), sizeof(c.startLine));
        in.read(reinterpret_cast<char*>(&c.endLine), sizeof(c.endLine));
        in.read(reinterpret_cast<char*>(&len), sizeof(len)); c.code.resize(len);
        in.read(&c.code[0], len);
        chunks.push_back(std::move(c));
    }
}

void RAGPipeline::clear() { chunks.clear(); std::cout << "[RAG] Cleared in-memory index" << std::endl; }

void RAGPipeline::indexProject(const std::string& rootPath) {
    int count = 0;
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
        [&](const CodeChunk& c) { return c.fileName.rfind(rootPath, 0) == 0; }),
        chunks.end());

    for (auto& p : fs::recursive_directory_iterator(rootPath)) {
        if (p.is_regular_file()) {
            auto ext = p.path().extension().string();
            if (ext == ".cpp" || ext == ".h" || ext == ".hpp" || ext == ".cxx") {
                indexFile(p.path().string()); count++;
            }
        }
    }
    std::cout << "[RAG] Indexed project at " << rootPath << " (" << count << " files)" << std::endl;
}

