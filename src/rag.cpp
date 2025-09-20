#include "../include/rag.h"
#include "../include/file_handler.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <filesystem>
#include <mutex>


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
RAGPipeline::RAGPipeline(std::unique_ptr<EmbeddingEngine> eng)
    : engine(std::move(eng)), store(engine.get()) {
    // Optionally initialize other members here
}
// Init pipeline (load index if exists)


// Initialization with custom index path
void RAGPipeline::init(const std::string& indexPath) {
    FileHandler fh;

    if (!indexPath.empty()) {
        indexFilePath = indexPath;
    } else {
        
        indexFilePath = fh.getRagPath("rag_index.bin");
    }
    std::cout << "[RAG] Loading index from: " << indexFilePath << "\n";
    loadIndex(indexFilePath);

    // Prune chunks not under RAG directory
    std::string ragDir = fh.getRagDirectory();
    size_t originalSize = chunks.size();

    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
        [&](const CodeChunk& c){
            return !pathIsUnderDirectory(c.fileName, ragDir);
        }), chunks.end());

    if (originalSize != chunks.size()) {
        std::cout << "[RAG] Pruned " << (originalSize - chunks.size())
                  << " out-of-scope chunks\n";
    }

    // Rebuild vector store + mappings
    rebuildInternalStructures();

    std::cout << "[RAG] Initialization complete: " << chunks.size()
              << " chunks ready\n";
}

void RAGPipeline::rebuildInternalStructures() {
    std::unique_lock lock(chunksMutex);  // exclusive access to chunks and codeToChunkIndex

    store.clear();
    codeToChunkIndex.clear();

    std::cout << "[RAG] Rebuilding vector store..." << std::flush;

    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        store.addDocument(chunk.code);
        codeToChunkIndex[chunk.code] = i;
    }

    std::cout << " done (" << chunks.size() << " embeddings)\n";
}


void RAGPipeline::indexProject(const std::string& rootPath) {
    if (!fs::exists(rootPath)) {
        std::cerr << "[RAG] Path does not exist: " << rootPath << "\n";
        return;
    }
    
    if (!fs::is_directory(rootPath)) {
        std::cerr << "[RAG] Path is not a directory: " << rootPath << "\n";
        return;
    }
    
    int successCount = 0, errorCount = 0;
    
    // Remove old chunks from this path first
    size_t oldSize = chunks.size();
    removeChunksFromPath(rootPath);
    
    if (oldSize != chunks.size()) {
        std::cout << "[RAG] Removed " << (oldSize - chunks.size()) 
                  << " old chunks from: " << rootPath << "\n";
        rebuildInternalStructures();
    }
    
    try {
        for (const auto& entry : fs::recursive_directory_iterator(rootPath)) {
            if (!entry.is_regular_file()) continue;
            
            auto ext = entry.path().extension().string();
            if (isSupportedExtension(ext)) {
                try {
                    indexFile(entry.path().string());
                    successCount++;
                } catch (const std::exception& e) {
                    std::cerr << "[RAG] Error indexing " << entry.path() 
                              << ": " << e.what() << "\n";
                    errorCount++;
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[RAG] Filesystem error: " << e.what() << "\n";
        return;
    }
    
    std::cout << "[RAG] Indexed " << fs::absolute(rootPath) 
              << " - Success: " << successCount << ", Errors: " << errorCount << "\n";
}
// --- Query using VectorStore ---
std::string RAGPipeline::query(const std::string& query) {
    auto results = store.retrieve(query, 5);
    
    if (results.empty()) {
        return "[No relevant context found for: \"" + query + "\"]";
    }
    
    std::ostringstream oss;
    oss << "Found " << results.size() << " relevant chunks for: \"" 
        << query << "\"\n\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& [text, score] = results[i];
        
        // Find the corresponding chunk for metadata
        auto it = codeToChunkIndex.find(text);
        if (it != codeToChunkIndex.end()) {
            const auto& chunk = chunks[it->second];
            
            oss << "=== Chunk " << (i + 1) << " (score: " 
                << std::fixed << std::setprecision(3) << score << ") ===\n";
            oss << "File: " << fs::path(chunk.fileName).filename() << "\n";
            
            if (!chunk.symbolName.empty()) {
                oss << "Symbol: " << chunk.symbolName << "\n";
            }
            
            if (chunk.startLine > 0) {
                oss << "Lines: " << chunk.startLine << "-" << chunk.endLine << "\n";
            }
            
            oss << "Content:\n" << limitText(text, 400) << "\n\n";
        }
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

    // Skip empty files
    if (content.empty()) {
        std::cerr << "[RAG] File is empty, skipping: " << filePath << "\n";
        return;
    }

    // Chunk the file intelligently
    auto chunksVec = createSmartChunks(filePath, content);

    if (chunksVec.empty()) {
        // fallback: add the whole file as a single chunk
        CodeChunk fallbackChunk;
        fallbackChunk.fileName = fs::absolute(filePath).lexically_normal().string();
        fallbackChunk.symbolName = "";
        fallbackChunk.startLine = 1;
        fallbackChunk.endLine = 0;
        fallbackChunk.code = content;
        fallbackChunk.embedding = engine->embed(content);

        addChunkToIndex(std::move(fallbackChunk));
        store.addDocument(content);
        std::cerr << "[DEBUG] Added fallback chunk for file: " << filePath << "\n";
        return;
    }

    // Process each chunk
    for (auto& chunk : chunksVec) {
        // Create embedding
        chunk.embedding = engine->embed(chunk.code);

        // Add to RAG in-memory index
        addChunkToIndex(std::move(chunk));

        // Also add text to VectorStore
        store.addDocument(chunksVec.back().code);
    }

    std::cerr << "[DEBUG] Indexed file with " << chunksVec.size() 
              << " chunk(s): " << filePath << "\n";
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

    // Acquire shared lock for safe read access to chunks
    std::shared_lock lock(chunksMutex);

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

// ----------------- saveIndex (unified layout) -----------------
void RAGPipeline::saveIndex(const std::string& dbPath) const {
    std::filesystem::create_directories(std::filesystem::path(dbPath).parent_path());
    std::ofstream out(dbPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[basic_agent:RAG] Failed to open " << dbPath << " for writing.\n";
        return;
    }

    // Write number of chunks
    size_t n = chunks.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));

    // Write chunks
    for (const auto& c : chunks) {
        size_t len;

        len = c.fileName.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(c.fileName.data(), len);

        len = c.symbolName.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(c.symbolName.data(), len);

        out.write(reinterpret_cast<const char*>(&c.startLine), sizeof(c.startLine));
        out.write(reinterpret_cast<const char*>(&c.endLine), sizeof(c.endLine));

        len = c.code.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(c.code.data(), len);

        // Write embedding
        len = c.embedding.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        if (len > 0) {
            out.write(reinterpret_cast<const char*>(c.embedding.data()), len * sizeof(float));
        }
    }

    // Save engine state
    {
        std::string tmpFile = dbPath + ".engine_tmp";
        engine->saveState(tmpFile);
        std::ifstream engIn(tmpFile, std::ios::binary);
        std::string engData((std::istreambuf_iterator<char>(engIn)),
                             std::istreambuf_iterator<char>());
        size_t engSize = engData.size();
        out.write(reinterpret_cast<const char*>(&engSize), sizeof(engSize));
        out.write(engData.data(), engSize);
        std::filesystem::remove(tmpFile);
    }

    std::cout << "[basic_agent:RAG] Index saved to: " << dbPath
              << " (entries=" << n << ")\n";
} 

void RAGPipeline::loadIndex() {
    FileHandler fh;
    std::string path = fh.getRagPath("rag_index.bin");
    loadIndex(path);
    
}



// ----------------- loadIndex (unified layout) -----------------
void RAGPipeline::loadIndex(const std::string& dbPath) {
    std::ifstream in(dbPath, std::ios::binary);
    if (!in) {
        std::cerr << "[basic_agent:RAG] No index found at " << dbPath << " (starting fresh).\n";
        return;
    }

    size_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));

    {
        std::unique_lock lock(chunksMutex);  // lock for writes
        chunks.clear(); 
        chunks.reserve(n);
    }

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

        // Read embedding
        size_t embLen;
        in.read(reinterpret_cast<char*>(&embLen), sizeof(embLen));
        c.embedding.resize(embLen);
        if (embLen > 0) {
            in.read(reinterpret_cast<char*>(c.embedding.data()), embLen * sizeof(float));
        }

        try { c.fileName = fs::absolute(c.fileName).lexically_normal().string(); } catch (...) {}

        {
            std::unique_lock lock(chunksMutex);
            chunks.push_back(std::move(c));
        }
    }

    // Restore engine state
    size_t engSize;
    in.read(reinterpret_cast<char*>(&engSize), sizeof(engSize));
    if (engSize > 0) {
        std::string engData(engSize, '\0');
        in.read(&engData[0], engSize);
        std::string tmpFile = dbPath + ".engine_tmp";
        {
            std::ofstream tmpOut(tmpFile, std::ios::binary);
            tmpOut.write(engData.data(), engSize);
        }
        engine->loadState(tmpFile);
        std::filesystem::remove(tmpFile);
    }

    // Rebuild store from loaded chunks
    {
        std::unique_lock lock(chunksMutex);
        store.clear();
        codeToChunkIndex.clear();
        for (size_t i = 0; i < chunks.size(); ++i) {
            auto& c = chunks[i];
            if (!c.embedding.empty()) {
                store.addDocument(c.code);
                store.embeddings.back() = c.embedding;
                codeToChunkIndex[c.code] = i;
            }
        }
    }

    std::cout << "[basic_agent:RAG] Index loaded from: " << dbPath
              << " (entries=" << n << ")\n";
}

// --- Clear ---
void RAGPipeline::clear() { 
    std::unique_lock lock(chunksMutex);
    chunks.clear();
    codeToChunkIndex.clear();
    store.clear();
    std::cout << "[basic_agent:RAG] Cleared in-memory index" << std::endl; 
}

// --- Chunk creation is read-only, no lock needed ---
std::vector<CodeChunk> RAGPipeline::createSmartChunks(const std::string& filePath, 
                                                      const std::string& content) {
    auto ext = fs::path(filePath).extension().string();
    
    if (ext == ".md" || ext == ".txt") {
        return chunkByParagraphs(filePath, content);
    } else if (ext == ".cpp" || ext == ".h") {
        return chunkByFunctions(filePath, content); 
    } else {
        return chunkBySize(filePath, content);
    }
}

// --- Add single chunk safely ---
void RAGPipeline::addChunkToIndex(CodeChunk&& chunk) {
        std::cerr << "[DEBUG] Adding chunk: file=" << chunk.fileName
              << ", symbol=" << chunk.symbolName
              << ", start=" << chunk.startLine
              << ", end=" << chunk.endLine
              << ", code size=" << chunk.code.size()
              << ", embedding size=" << chunk.embedding.size() << "\n";
    std::unique_lock lock(chunksMutex);
    size_t index = chunks.size();
    chunks.push_back(std::move(chunk));
    store.addDocument(chunks.back().code);
    codeToChunkIndex[chunks.back().code] = index;
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

size_t RAGPipeline::getCurrentMemoryUsage() const {
    size_t total = 0;
    for (const auto& c : chunks) {
        total += c.fileName.size() + c.symbolName.size() + c.code.size();
        total += sizeof(c.startLine) + sizeof(c.endLine);
        total += c.embedding.size() * sizeof(float);
    }
    return total;
}


void RAGPipeline::enforceMemoryLimits() {
    std::unique_lock lock(chunksMutex);  // exclusive lock for modification

    if (chunks.size() > MAX_CHUNKS || getCurrentMemoryUsage() > MAX_TOTAL_SIZE) {
        std::cout << "[RAG] Memory limits exceeded, removing oldest chunks\n";
        
        // Simple LRU: remove first 20% of chunks
        size_t toRemove = chunks.size() / 5;
        chunks.erase(chunks.begin(), chunks.begin() + toRemove);

        rebuildInternalStructures();  // also protected internally if needed
        std::cout << "[RAG] Removed " << toRemove << " chunks\n";
    }
}



void RAGPipeline::removeChunksFromPath(const std::string& rootPath) {
    std::unique_lock lock(chunksMutex); // exclusive lock for writes
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
        [&](const CodeChunk& c) { 
            return c.fileName.rfind(rootPath, 0) == 0; 
        }), chunks.end());
}

std::vector<CodeChunk> RAGPipeline::chunkByParagraphs(
    const std::string& filePath, const std::string& content)
{
    std::vector<CodeChunk> result;
    std::istringstream stream(content);
    std::string paragraph;
    std::string currentChunk;
    int lineNum = 1;
    int chunkStart = 1;

    while (std::getline(stream, paragraph)) {
        if (paragraph.empty()) {
            if (!currentChunk.empty()) {
                CodeChunk chunk;
                chunk.fileName = fs::absolute(filePath).string();
                chunk.symbolName = "";
                chunk.startLine = chunkStart;
                chunk.endLine = lineNum - 1;
                chunk.code = currentChunk;
                result.push_back(chunk);

                currentChunk.clear();
                chunkStart = lineNum + 1;
            }
        } else {
            currentChunk += paragraph + "\n";
        }
        lineNum++;
    }

    // Add final chunk if exists
    if (!currentChunk.empty()) {
        CodeChunk chunk;
        chunk.fileName = fs::absolute(filePath).string();
        chunk.symbolName = "";
        chunk.startLine = chunkStart;
        chunk.endLine = lineNum;
        chunk.code = currentChunk;
        result.push_back(chunk);
    }

    return result;
}

std::vector<CodeChunk> RAGPipeline::chunkByFunctions(
    const std::string& filePath, const std::string& content)
{
    std::vector<CodeChunk> result;
    std::istringstream stream(content);
    std::string line;
    std::string currentChunk;
    int lineNum = 1;
    int chunkStart = 1;

    std::regex functionRegex(R"(\s*([\w:~]+\s+)*[\w:~]+\s*\([^)]*\)\s*\{?)");

    while (std::getline(stream, line)) {
        currentChunk += line + "\n";

        if (std::regex_match(line, functionRegex) || (lineNum - chunkStart) > 50) {
            if (!currentChunk.empty()) {
                CodeChunk chunk;
                chunk.fileName = fs::absolute(filePath).string();
                chunk.symbolName = "";
                chunk.startLine = chunkStart;
                chunk.endLine = lineNum;
                chunk.code = currentChunk;
                result.push_back(chunk);

                currentChunk.clear();
                chunkStart = lineNum + 1;
            }
        }
        lineNum++;
    }

    if (!currentChunk.empty()) {
        CodeChunk chunk;
        chunk.fileName = fs::absolute(filePath).string();
        chunk.symbolName = "";
        chunk.startLine = chunkStart;
        chunk.endLine = lineNum;
        chunk.code = currentChunk;
        result.push_back(chunk);
    }

    return result;
}

std::vector<CodeChunk> RAGPipeline::chunkBySize(
    const std::string& filePath, const std::string& content)
{
    std::vector<CodeChunk> result;
    constexpr size_t CHUNK_SIZE = 2048; // 2KB chunks

    for (size_t pos = 0; pos < content.size(); pos += CHUNK_SIZE) {
        size_t chunkEnd = std::min(pos + CHUNK_SIZE, content.size());

        CodeChunk chunk;
        chunk.fileName = fs::absolute(filePath).string();
        chunk.symbolName = "";
        chunk.startLine = 1; // Approximate
        chunk.endLine = 1;
        chunk.code = content.substr(pos, chunkEnd - pos);

        result.push_back(chunk);
    }

    return result;
}









