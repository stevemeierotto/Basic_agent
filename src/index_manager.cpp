#include "../include/index_manager.h"
#include "../include/file_handler.h"
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <mutex>
#include <fstream>



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

std::string sanitize_utf8(const std::string& input) {
    std::string output;
    output.reserve(input.size());
    for (unsigned char c : input) {
        if (c < 0x80) {
            output.push_back(c);  // ASCII
        } else {
            // Replace any non-ASCII byte with a space or '?'
            output.push_back(' ');
        }
    }
    return output;
}


size_t IndexManager::getCurrentMemoryUsage() const {
    size_t total = 0;
    for (const auto& c : chunks) {
        total += c.fileName.size() + c.symbolName.size() + c.code.size();
        total += sizeof(c.startLine) + sizeof(c.endLine);
        total += c.embedding.size() * sizeof(float);
    }
    return total;
}

const std::vector<CodeChunk>& IndexManager::getChunks() const {
    return chunks; // return the internal vector of chunks
}
void IndexManager::enforceMemoryLimits() {
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



void IndexManager::removeChunksFromPath(const std::string& rootPath) {
    std::unique_lock lock(chunksMutex); // exclusive lock for writes
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
        [&](const CodeChunk& c) { 
            return c.fileName.rfind(rootPath, 0) == 0; 
        }), chunks.end());
}

// --- Clear all chunks, store, and mappings ---
void IndexManager::clear() {
    std::unique_lock lock(chunksMutex);
    chunks.clear();
    codeToChunkIndex.clear();
    store.clear();
    std::cout << "[IndexManager] Cleared all in-memory chunks and store.\n";
}

// --- Add single chunk safely (renamed from addChunkToIndex) ---
void IndexManager::addChunk(CodeChunk&& chunk) {
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


void IndexManager::init(const std::string& indexPath) {
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


void IndexManager::indexFile(const std::string& filePath) {
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

    if (!engine) {
        std::cerr << "[ERROR] Embedding engine is null; cannot index file: " << filePath << "\n";
        return;
    }

    content = sanitize_utf8(content);
    //Chunker chunker;
    // Create smart chunks (may return empty)
    auto chunksVec = Chunker::createSmartChunks(filePath, content);

    // If the smart chunker returned empty, fallback to size-based chunking.
    if (chunksVec.empty()) {
        std::cerr << "[WARN] createSmartChunks returned 0 chunks for: "
                  << filePath << ". Falling back to chunkBySize().\n";
        chunksVec = Chunker::chunkBySize(filePath, content);
    }

    // If still empty, add the whole file as a single fallback chunk (last resort).
    if (chunksVec.empty()) {
        std::cerr << "[ERROR] chunkBySize also returned 0 chunks for: "
                  << filePath << ". Adding whole file as a single chunk.\n";

        CodeChunk fallbackChunk;
        fallbackChunk.fileName = fs::absolute(filePath).lexically_normal().string();
        fallbackChunk.symbolName = "";
        fallbackChunk.startLine = 1;
        fallbackChunk.endLine = 0;
        fallbackChunk.code = std::move(content); // move the big string

        // Remove null bytes
        fallbackChunk.code.erase(std::remove(fallbackChunk.code.begin(), fallbackChunk.code.end(), '\0'),
                                 fallbackChunk.code.end());

        try {
            fallbackChunk.embedding = engine->embed(fallbackChunk.code);
        } catch (const std::exception& ex) {
            std::cerr << "[ERROR] Embedding failed for fallback chunk (" << filePath
                      << "): " << ex.what() << "\n";
        }

        addChunkToIndex(std::move(fallbackChunk));
        try {
            store.addDocument(chunksVec.empty() ? fallbackChunk.code : std::string()); // best-effort
        } catch (const std::exception& ex) {
            std::cerr << "[WARN] store.addDocument failed for fallback chunk: " << ex.what() << "\n";
        }
        std::cerr << "[DEBUG] Indexed file with 1 fallback chunk: " << filePath << "\n";
        return;
    }

    // Process each chunk: generate embedding, add to index and to vector store
    size_t added = 0;
    for (size_t i = 0; i < chunksVec.size(); ++i) {
        CodeChunk &chunkRef = chunksVec[i];

        // Skip empty or whitespace-only chunks
        if (chunkRef.code.empty() ||
            std::all_of(chunkRef.code.begin(), chunkRef.code.end(), ::isspace)) 
        {
            std::cerr << "[WARN] Skipping blank or whitespace-only chunk at index " << i
                      << " for file: " << filePath << "\n";
            continue;
        }

        // Remove null bytes to avoid JSON/UTF-8 crashes
        chunkRef.code.erase(std::remove(chunkRef.code.begin(), chunkRef.code.end(), '\0'),
                            chunkRef.code.end());

        // Skip very low-content chunks (few non-space characters)
        size_t nonspace_count = std::count_if(
            chunkRef.code.begin(),
            chunkRef.code.end(),
            [](char c){ return !std::isspace(c); }
        );
        if (nonspace_count < 10) { // threshold can be adjusted
            std::cerr << "[WARN] Skipping low-content chunk at index " << i
                      << " for file: " << filePath << "\n";
            continue;
        }

        try {
            chunkRef.embedding = engine->embed(chunkRef.code);

            // Skip zero-norm embeddings
            bool isZero = std::all_of(
                chunkRef.embedding.begin(),
                chunkRef.embedding.end(),
                [](float v){ return v == 0.0f; }
            );
            if (isZero) {
                std::cerr << "[WARN] Skipping zero-norm embedding for chunk " << i
                          << " in file: " << filePath << "\n";
                continue;
            }

        } catch (const std::exception& ex) {
            std::cerr << "[ERROR] Embedding failed for chunk " << i << " ("
                      << filePath << "): " << ex.what() << " — skipping chunk.\n";
            continue;
        }

        // Move the chunk out of the vector into a local variable before adding
        CodeChunk chunk = std::move(chunkRef);

        // Add to RAG in-memory index
        addChunkToIndex(std::move(chunk));

        // Add the chunk text to the vector store. Use chunk.code (still valid).
        try {
            store.addDocument(chunk.code);
        } catch (const std::exception& ex) {
            std::cerr << "[WARN] store.addDocument failed for chunk " << i
                      << " (" << filePath << "): " << ex.what() << "\n";
        }

        ++added;
    }

    std::cerr << "[DEBUG] Indexed file with " << added
              << " chunk(s) (requested: " << chunksVec.size()
              << "): " << filePath << "\n";
}

void IndexManager::indexProject(const std::string& rootPath) {
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


// --- Save / Load ---


void IndexManager::saveIndex() const {
    FileHandler fh;
    std::string path = fh.getRagPath("rag_index.bin");
    saveIndex(path);

}

// ----------------- saveIndex (unified layout) -----------------
void IndexManager::saveIndex(const std::string& dbPath) const {
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

void IndexManager::loadIndex() {
    FileHandler fh;
    std::string path = fh.getRagPath("rag_index.bin");
    loadIndex(path);
    
}



// ----------------- loadIndex (unified layout) -----------------
void IndexManager::loadIndex(const std::string& dbPath) {
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

void IndexManager::rebuildInternalStructures() {
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

// --- Add single chunk safely ---
void IndexManager::addChunkToIndex(CodeChunk&& chunk) {
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
