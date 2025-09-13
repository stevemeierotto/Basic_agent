#include "../include/rag.h"
#include "../include/file_handler.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;


// Helper: return true if 'path' is inside 'directory'.
// Both paths are converted to absolute + normalized before comparison.
static bool pathIsUnderDirectory(const std::string& pathStr, const std::string& dirStr) {
    try {
        fs::path p = fs::absolute(pathStr).lexically_normal();
        fs::path d = fs::absolute(dirStr).lexically_normal();

        std::string pS = p.string();
        std::string dS = d.string();

        // ensure trailing separator on directory string so prefix-check is correct
        if (!dS.empty() && dS.back() != fs::path::preferred_separator) {
            dS.push_back(fs::path::preferred_separator);
        }

        // also add separator to file path when comparing (safer)
        if (!pS.empty() && pS.back() != fs::path::preferred_separator) {
            // no-op, don't append separator to file paths
        }

        return pS.rfind(dS, 0) == 0; // starts with
    } catch (...) {
        return false;
    }
}

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

// Init pipeline (load index if exists)

void RAGPipeline::init() {
    FileHandler fh;
    // default index file inside agent_workspace/rag
    indexFilePath = fh.getRagPath("rag_index.bin");
    loadIndex(indexFilePath);

    // After loading, prune any chunks that are NOT under the rag directory
    std::string ragDir = fh.getRagDirectory();

    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
        [&](const CodeChunk& c){
            // keep only chunks within ragDir
            return !pathIsUnderDirectory(c.fileName, ragDir);
        }), chunks.end());

    // Rebuild vector store from the remaining chunks (clear then add)
    store.clear();
    for (const auto &c : chunks) {
        store.addDocument(c.code);
    }

    std::cout << "[RAG] init completed; indexFilePath=" << indexFilePath << ", entries=" << chunks.size() << "\n";
}

void RAGPipeline::init(const std::string& indexPath) {
    // Respect provided indexPath override but still ensure ragDir pruning
    FileHandler fh;
    indexFilePath = indexPath.empty() ? fh.getRagPath("rag_index.bin") : indexPath;
    loadIndex(indexFilePath);

    std::string ragDir = fh.getRagDirectory();
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
        [&](const CodeChunk& c){
            return !pathIsUnderDirectory(c.fileName, ragDir);
        }), chunks.end());

    store.clear();
    for (const auto &c : chunks) {
        store.addDocument(c.code);
    }

    std::cout << "[RAG] init(path) completed; indexFilePath=" << indexFilePath << ", entries=" << chunks.size() << "\n";
}


void RAGPipeline::indexProject(const std::string& rootPath) {
    int count = 0;

    // Remove old chunks from this path
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
        [&](const CodeChunk& c){ return c.fileName.rfind(rootPath, 0) == 0; }),
        chunks.end());

    for (auto& entry : fs::recursive_directory_iterator(rootPath)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            if (ext == ".txt" || ext == ".md" || ext == ".epub" || ext == ".pdf") {
                indexFile(entry.path().string());
                count++;
            }
        }
    }

    std::cout << "[RAG] Indexed folder: " << fs::absolute(rootPath) 
              << " (" << count << " files)\n";
}

// --- Query using VectorStore ---
std::string RAGPipeline::query(const std::string& query) {
    // Retrieve top 5 text chunks from vector store
    auto results = store.retrieve(query, 5);

    if (results.empty()) {
        return "[No relevant context found]";
    }

    // Format retrieved chunks
    std::ostringstream oss;
    oss << "Top relevant context chunks:\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& [text, score] = results[i];
        oss << "Chunk " << (i + 1) << " (score: " << score << "):\n";
        oss << text.substr(0, 500) << "...\n"; // show first 500 chars
        oss << "---\n";
    }
    return oss.str();
}

// --- Index a single file (store code chunks in VectorStore) ---
// ----------------- indexFile -----------------
void RAGPipeline::indexFile(const std::string& filePath) {
    // Read the file contents; store absolute path
    std::ifstream in(filePath, std::ios::binary);
    if (!in) {
        std::cerr << "[RAG] Failed to open file for indexing: " << filePath << "\n";
        return;
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    std::string content = ss.str();

    CodeChunk chunk;
    chunk.fileName = fs::absolute(filePath).lexically_normal().string();
    chunk.symbolName = "";
    chunk.startLine = 1;
    chunk.endLine = 0;
    chunk.code = content;
    // embedding left empty until you have a real embedder
    chunk.embedding.clear();

    // Add to in-memory list and to vector store immediately
    chunks.push_back(chunk);
    store.addDocument(chunk.code);
}
// --- Retrieve top-k relevant CodeChunks ---
std::vector<CodeChunk> RAGPipeline::retrieveRelevant(
    const std::string& query,
    const std::vector<int>& errorLines,
    int topK)
{
    std::vector<CodeChunk> matches;

    // Retrieve top-k texts from VectorStore
    auto results = store.retrieve(query, topK);

    // Map each retrieved text back to its CodeChunk
    for (auto& [text, score] : results) {
        auto it = std::find_if(chunks.begin(), chunks.end(),
            [&text](const CodeChunk& c) { return c.code == text; });
        if (it != chunks.end()) {
            matches.push_back(*it);
        }
    }

    return matches;
}

// --- Save / Load ---


void RAGPipeline::saveIndex() const {
    FileHandler fh;
    std::string path = fh.getRagPath("rag_index.bin");
    saveIndex(path);
}

void RAGPipeline::saveIndex(const std::string& dbPath) const {
    // Ensure parent folder exists
    std::filesystem::create_directories(std::filesystem::path(dbPath).parent_path());

    std::ofstream out(dbPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[basic_agent:RAG] Failed to open " << dbPath << " for writing.\n";
        return;
    }

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

        // persist embeddings
        size_t embSize = c.embedding.size();
        out.write(reinterpret_cast<const char*>(&embSize), sizeof(embSize));
        if (embSize > 0) {
            out.write(reinterpret_cast<const char*>(c.embedding.data()), embSize * sizeof(float));
        }
    }

    std::cout << "[basic_agent:RAG] Index saved to: " << dbPath << "\n";
}

void RAGPipeline::loadIndex() {
    FileHandler fh;
    std::string path = fh.getRagPath("rag_index.bin");
    loadIndex(path);
}


// ----------------- loadIndex (reworked) -----------------
void RAGPipeline::loadIndex(const std::string& dbPath) {
    std::ifstream in(dbPath, std::ios::binary);
    if (!in) {
        std::cerr << "[basic_agent:RAG] No index found at " << dbPath << " (starting fresh).\n";
        return;
    }

    size_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    chunks.clear(); chunks.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        CodeChunk c; size_t len;

        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        c.fileName.resize(len);
        in.read(&c.fileName[0], len);

        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        c.symbolName.resize(len);
        in.read(&c.symbolName[0], len);

        in.read(reinterpret_cast<char*>(&c.startLine), sizeof(c.startLine));
        in.read(reinterpret_cast<char*>(&c.endLine), sizeof(c.endLine));

        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        c.code.resize(len);
        in.read(&c.code[0], len);

        size_t embSize;
        in.read(reinterpret_cast<char*>(&embSize), sizeof(embSize));
        c.embedding.resize(embSize);
        if (embSize > 0) {
            in.read(reinterpret_cast<char*>(c.embedding.data()), embSize * sizeof(float));
        }

        // normalize stored file path to absolute to make comparisons reliable
        try {
            c.fileName = fs::absolute(c.fileName).lexically_normal().string();
        } catch (...) {
            // if something goes wrong, keep original string
        }

        chunks.push_back(std::move(c));
        // NOTE: we do NOT add to store here; we'll rebuild store after possible pruning
    }

    std::cout << "[basic_agent:RAG] Index loaded from: " << dbPath << " (raw entries=" << n << ")\n";
}
// --- Clear ---
void RAGPipeline::clear() { 
    chunks.clear(); 
    std::cout << "[basic_agent:RAG] Cleared in-memory index" << std::endl; 
}



