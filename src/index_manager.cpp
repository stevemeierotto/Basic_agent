#include "../include/index_manager.h"
#include "../include/corpus_create.h"
#include "../include/corpus_documents.h"
#include "../include/file_handler.h"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <shared_mutex>
#include <mutex>
#include <fstream>
#include <cstdint>
#include <sstream>
#include <unistd.h>
#include <../include/json.hpp>

using json = nlohmann::json;

namespace fs = std::filesystem;

namespace {

std::string agentWorkspaceRoot() {
    FileHandler fh;
    try {
        return fs::absolute(fh.getAgentWorkspacePath()).lexically_normal().string();
    } catch (...) {
        return fh.getAgentWorkspacePath();
    }
}

bool isUnderAgentWorkspace(const std::string& normalizedPath) {
    const std::string root = agentWorkspaceRoot();
    if (root.empty() || normalizedPath.size() < root.size()) {
        return false;
    }
    if (normalizedPath.compare(0, root.size(), root) != 0) {
        return false;
    }
    if (normalizedPath.size() == root.size()) {
        return true;
    }
    const char next = normalizedPath[root.size()];
    return next == '/';
}

/** Phase 5 + R2: pair INDEXING_STARTED with COMPLETED; COMPLETE carries outcome metadata. */
struct IndexingCompletionGuard {
    IndexManager* manager = nullptr;
    EventCallback& callback;
    std::string session_id;
    std::string file_path;
    bool armed = false;
    bool resolved = false;
    bool success = false;
    int chunk_count = 0;
    std::string reason;

    IndexingCompletionGuard(IndexManager* mgr,
                            EventCallback& cb,
                            std::string sid,
                            std::string path)
        : manager(mgr), callback(cb), session_id(std::move(sid)), file_path(std::move(path)) {}

    void arm() { armed = true; }

    void setSuccess(int chunks) {
        success = true;
        chunk_count = chunks;
        reason.clear();
        resolved = true;
    }

    void setFailure(const std::string& reason_code) {
        success = false;
        chunk_count = 0;
        reason = reason_code;
        resolved = true;
    }

    ~IndexingCompletionGuard() {
        if (!armed || !callback) {
            return;
        }
        if (!resolved) {
            success = false;
            chunk_count = 0;
            reason = "read_failed";
        }
        if (manager) {
            manager->recordIndexingOutcome(file_path, success, chunk_count, reason);
        }
        ControllerEvent ev;
        ev.type = EventType::INDEXING_COMPLETED;
        ev.session_id = session_id;
        nlohmann::json meta{
            {"file_path", file_path},
            {"success", success},
            {"chunk_count", chunk_count},
        };
        if (!success && !reason.empty()) {
            meta["reason"] = reason;
        }
        ev.metadata = std::move(meta);
        callback(ev);
        std::cout << "[RAG] indexing finished path=" << file_path
                  << " success=" << (success ? "true" : "false")
                  << " chunks=" << chunk_count;
        if (!reason.empty()) {
            std::cout << " reason=" << reason;
        }
        std::cout << "\n";
    }
};

bool isWhitespaceOnlyContent(const std::string& content) {
    return content.empty()
        || std::all_of(content.begin(), content.end(), [](unsigned char c) {
               return std::isspace(c) != 0;
           });
}

constexpr size_t kSmallDocumentSingleChunkMaxBytes = 4096;

struct IndexingPathDiagnostics {
    size_t input_bytes = 0;
    size_t paragraph_blocks = 0;
    size_t chunks_generated = 0;
    size_t chunks_skipped_empty = 0;
    size_t chunks_skipped_short = 0;
    size_t chunks_skipped_embed = 0;
    size_t chunks_stored_after_primary = 0;
    std::string fallback = "none";
    int final_stored = 0;
};

size_t countMarkdownParagraphBlocks(const std::string& content) {
    size_t blocks = 0;
    bool in_para = false;
    std::istringstream stream(content);
    std::string line;
    while (std::getline(stream, line)) {
        const bool blank =
            line.empty()
            || std::all_of(line.begin(), line.end(), [](unsigned char c) {
                   return std::isspace(c) != 0;
               });
        if (blank) {
            if (in_para) {
                ++blocks;
            }
            in_para = false;
        } else {
            in_para = true;
        }
    }
    if (in_para) {
        ++blocks;
    }
    return blocks;
}

void logIndexingDiagnostics(const std::string& normalizedPath,
                            const IndexingPathDiagnostics& diag) {
    std::cerr << "[RAG] indexing diagnostics path=" << normalizedPath
              << " input_bytes=" << diag.input_bytes
              << " paragraph_blocks=" << diag.paragraph_blocks
              << " chunks_generated=" << diag.chunks_generated
              << " skipped_empty=" << diag.chunks_skipped_empty
              << " skipped_short=" << diag.chunks_skipped_short
              << " skipped_embed=" << diag.chunks_skipped_embed
              << " stored_after_primary=" << diag.chunks_stored_after_primary
              << " fallback=" << diag.fallback
              << " final_stored=" << diag.final_stored << "\n";
}

} // namespace

IndexManager::IndexManager(EmbeddingEngine* eng)
    : store(eng), engine(eng), localTfIdfEngine(std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf)) 
{
    startWorker();
}

IndexManager::~IndexManager() {
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_stopWorker = true;
    }
    m_queueCv.notify_all();
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
}

void IndexManager::startWorker() {
    m_workerThread = std::thread(&IndexManager::workerLoop, this);
}

void IndexManager::workerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCv.wait(lock, [this]() { return m_stopWorker || !m_taskQueue.empty(); });
            
            if (m_stopWorker && m_taskQueue.empty()) break;
            
            task = std::move(m_taskQueue.front());
            m_taskQueue.pop();
            m_isIndexing = !m_taskQueue.empty();
        }
        
        if (task) {
            task();
        }

        // Check again after task execution
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_isIndexing = !m_taskQueue.empty();
        }
    }
}

void IndexManager::indexFileAsync(const std::string& filePath) {
    std::lock_guard<std::mutex> lock(m_queueMutex);
    m_taskQueue.push([this, filePath]() {
        indexFile(filePath);
    });
    m_isIndexing = true;
    m_queueCv.notify_one();
}

void IndexManager::indexProjectAsync(const std::string& rootPath) {
    std::lock_guard<std::mutex> lock(m_queueMutex);
    m_taskQueue.push([this, rootPath]() {
        indexProject(rootPath);
    });
    m_isIndexing = true;
    m_queueCv.notify_one();
}

std::string sanitize_utf8(const std::string& input) {
    std::string output;
    output.reserve(input.size());
    for (unsigned char c : input) {
        if (c < 0x80) {
            output.push_back(c);  // ASCII
        } else {
            output.push_back(' ');
        }
    }
    return output;
}

static std::uint64_t computeFileFingerprint(const std::string& filePath) {
    std::error_code ec;
    const auto size = fs::file_size(filePath, ec);
    if (ec) return 0;
    const auto writeTime = fs::last_write_time(filePath, ec);
    if (ec) return static_cast<std::uint64_t>(size);
    const auto ticks = static_cast<std::uint64_t>(writeTime.time_since_epoch().count());
    return (ticks * 1315423911ULL) ^ static_cast<std::uint64_t>(size);
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
    return chunks;
}

const CodeChunk* IndexManager::getChunkByCode(const std::string& codeText) const {
    std::shared_lock<std::shared_mutex> lock(chunksMutex);
    auto it = codeToChunkIndex.find(codeText);
    if (it != codeToChunkIndex.end() && it->second < chunks.size()) {
        return &chunks[it->second];
    }
    return nullptr;
}

void IndexManager::enforceMemoryLimits() {
    std::unique_lock lock(chunksMutex);
    if (chunks.size() > MAX_CHUNKS || getCurrentMemoryUsage() > MAX_TOTAL_SIZE) {
        size_t toRemove = chunks.size() / 5;
        chunks.erase(chunks.begin(), chunks.begin() + toRemove);
        rebuildInternalStructures();
    }
}

void IndexManager::removeChunksFromPath(const std::string& rootPath) {
    std::unique_lock lock(chunksMutex);
    std::vector<std::string> removedFiles;
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
        [&](const CodeChunk& c) { 
            const bool remove = c.fileName.rfind(rootPath, 0) == 0;
            if (remove) removedFiles.push_back(c.fileName);
            return remove; 
        }), chunks.end());
    for (const auto& file : removedFiles) indexedFileFingerprints.erase(file);
}

void IndexManager::clear() {
    std::unique_lock lock(chunksMutex);
    chunks.clear();
    codeToChunkIndex.clear();
    indexedFileFingerprints.clear();
    store.clear();
    {
        std::unique_lock olock(outcomesMutex_);
        indexingFailureReasons_.clear();
    }
}

void IndexManager::setActiveCorpusFiles(const std::vector<std::string>& filePaths) {
    std::unique_lock lock(chunksMutex);
    activeCorpusFiles_.clear();
    activeCorpusRoots_.clear();
    for (const auto& path : filePaths) {
        try {
            const fs::path normalized = fs::absolute(path).lexically_normal();
            if (fs::is_directory(normalized)) {
                activeCorpusRoots_.push_back(normalized.string());
            } else {
                activeCorpusFiles_.insert(normalized.string());
            }
        } catch (...) {
            activeCorpusFiles_.insert(path);
        }
    }
}

void IndexManager::registerAttachmentOwner(const std::string& normalizedPath,
                                           const std::string& owner_context_id) {
    if (normalizedPath.empty() || owner_context_id.empty()) {
        return;
    }
    {
        std::unique_lock lock(chunksMutex);
        try {
            attachmentOwners_[fs::absolute(normalizedPath).lexically_normal().string()] =
                owner_context_id;
        } catch (...) {
            attachmentOwners_[normalizedPath] = owner_context_id;
        }
    }
    classifyAllChunksMetadata();
    saveAttachmentRegistry();
}

std::optional<std::string> IndexManager::findSessionOwnedAttachmentPath(
    const std::string& owner_context_id,
    const std::string& base_name) const {
    if (owner_context_id.empty() || base_name.empty()) {
        return std::nullopt;
    }
    std::shared_lock lock(chunksMutex);
    for (const auto& entry : attachmentOwners_) {
        if (entry.second != owner_context_id) {
            continue;
        }
        if (fs::path(entry.first).filename().string() == base_name) {
            return entry.first;
        }
    }
    return std::nullopt;
}

namespace {

constexpr int kAttachmentRegistrySchemaVersion = 1;

std::string attachmentRegistryFilePath() {
    FileHandler fh;
    return fh.getAgentWorkspacePath("rag_attachment_registry.json");
}

} // namespace

void IndexManager::loadAttachmentRegistry() {
    const std::string path = attachmentRegistryFilePath();
    std::ifstream in(path);
    if (!in.is_open()) {
        return;
    }
    nlohmann::json body;
    try {
        in >> body;
    } catch (...) {
        return;
    }
    if (!body.is_object() || body.value("schema_version", 0) < kAttachmentRegistrySchemaVersion) {
        return;
    }
    if (!body.contains("owners") || !body["owners"].is_object()) {
        return;
    }
    std::unique_lock lock(chunksMutex);
    for (const auto& item : body["owners"].items()) {
        if (!item.value().is_string()) {
            continue;
        }
        const std::string owner = item.value().get<std::string>();
        if (owner.empty()) {
            continue;
        }
        attachmentOwners_[item.key()] = owner;
    }
}

void IndexManager::saveAttachmentRegistry() const {
    nlohmann::json body;
    body["schema_version"] = kAttachmentRegistrySchemaVersion;
    nlohmann::json owners = nlohmann::json::object();
    {
        std::shared_lock lock(chunksMutex);
        for (const auto& entry : attachmentOwners_) {
            owners[entry.first] = entry.second;
        }
    }
    body["owners"] = std::move(owners);
    const std::string path = attachmentRegistryFilePath();
    const std::string temp_path = path + ".tmp." + std::to_string(getpid());
    try {
        {
            std::ofstream out(temp_path, std::ios::trunc);
            if (!out) {
                return;
            }
            out << body.dump(2);
            if (!out) {
                std::error_code ec;
                fs::remove(temp_path, ec);
                return;
            }
        }
        std::error_code ec;
        fs::rename(temp_path, path, ec);
        if (ec) {
            fs::remove(temp_path, ec);
        }
    } catch (...) {
        std::error_code ec;
        fs::remove(temp_path, ec);
    }
}

void IndexManager::classifyChunkInPlace(CodeChunk& chunk) const {
    std::string ownerLookup;
    {
        std::shared_lock lock(chunksMutex);
        auto it = attachmentOwners_.find(chunk.fileName);
        if (it != attachmentOwners_.end()) {
            ownerLookup = it->second;
        }
    }
    Thoth::classifyChunkMetadata(chunk, ownerLookup);
}

void IndexManager::classifyAllChunksMetadataUnlocked() {
    for (auto& chunk : chunks) {
        std::string ownerLookup;
        auto it = attachmentOwners_.find(chunk.fileName);
        if (it != attachmentOwners_.end()) {
            ownerLookup = it->second;
        }
        Thoth::classifyChunkMetadata(chunk, ownerLookup);
    }
}

void IndexManager::classifyAllChunksMetadata() {
    std::unique_lock lock(chunksMutex);
    classifyAllChunksMetadataUnlocked();
}

bool IndexManager::chunkPassesScopeFilter(const CodeChunk& chunk,
                                          const Thoth::RetrievalScope& scope) const {
    return Thoth::chunkPassesRetrievalScope(chunk, scope);
}

bool IndexManager::chunkInActiveCorpus(const CodeChunk& chunk) const {
    if (activeCorpusFiles_.empty() && activeCorpusRoots_.empty()) {
        return true;
    }
    if (activeCorpusFiles_.count(chunk.fileName) > 0) {
        return true;
    }
    for (const auto& root : activeCorpusRoots_) {
        if (chunk.fileName.rfind(root, 0) == 0) {
            return true;
        }
    }
    return false;
}

std::vector<std::pair<std::string, float>> IndexManager::retrieveChunks(
    const std::string& query, int topK, const Thoth::RetrievalScope* scope) {
    const bool scopeFilter = scope != nullptr;
    const bool filterCorpus =
        scopeFilter || !activeCorpusFiles_.empty() || !activeCorpusRoots_.empty();
    const int recallK = filterCorpus ? std::max(topK * 8, 80) : std::max(topK * 2, topK);
    auto rawResults = store.retrieve(query, recallK);

    if (!filterCorpus) {
        if (static_cast<int>(rawResults.size()) > topK) {
            rawResults.resize(static_cast<std::size_t>(topK));
        }
        return rawResults;
    }

    std::shared_lock<std::shared_mutex> lock(chunksMutex);
    std::vector<std::pair<std::string, float>> filtered;
    filtered.reserve(rawResults.size());
    for (const auto& entry : rawResults) {
        auto it = codeToChunkIndex.find(entry.first);
        if (it == codeToChunkIndex.end() || it->second >= chunks.size()) {
            continue;
        }
        const CodeChunk& chunk = chunks[it->second];
        if (scopeFilter) {
            if (!chunkPassesScopeFilter(chunk, *scope)) {
                continue;
            }
        } else if (!chunkInActiveCorpus(chunk)) {
            continue;
        }
        filtered.push_back(entry);
        if (static_cast<int>(filtered.size()) >= topK) {
            break;
        }
    }
    return filtered;
}

void IndexManager::addChunk(CodeChunk&& chunk) {
    if (chunk.code.empty()) {
        return;
    }
    
    // Phase 3.1 Hardening: Accept chunk if it has EITHER semantic embedding OR keyword signal
    bool hasSemantic = !chunk.embedding.empty() && 
                       !std::all_of(chunk.embedding.begin(), chunk.embedding.end(), [](float v){ return v == 0.0f; });
    bool hasKeyword = chunk.keyword_score > 0.001f;

    if (!hasSemantic && !hasKeyword) {
        return;
    }

    {
        std::unique_lock lock(chunksMutex);
        if (codeToChunkIndex.find(chunk.code) != codeToChunkIndex.end()) return;
        size_t index = chunks.size();
        chunks.push_back(std::move(chunk));
        store.addDocumentWithEmbedding(chunks.back().code, chunks.back().embedding);
        codeToChunkIndex[chunks.back().code] = index;
        std::string ownerLookup;
        auto ownerIt = attachmentOwners_.find(chunks.back().fileName);
        if (ownerIt != attachmentOwners_.end()) {
            ownerLookup = ownerIt->second;
        }
        Thoth::classifyChunkMetadata(chunks.back(), ownerLookup);
    }
}

std::string IndexManager::getCurrentCommitHash() const {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen("git rev-parse HEAD 2>/dev/null", "r");
    if (!pipe) return "unknown";
    if (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result = buffer;
        if (!result.empty() && result.back() == '\n') result.pop_back();
    }
    pclose(pipe);
    return result.empty() ? "unknown" : result;
}

bool IndexManager::shouldReindexFile(const std::string& filePath) {
    std::string normalized;
    try {
        normalized = fs::absolute(filePath).lexically_normal().string();
    } catch (...) { normalized = filePath; }

    const auto fingerprint = computeFileFingerprint(normalized);
    if (fingerprint == 0) return true;
    
    std::shared_lock lock(chunksMutex);
    const auto it = indexedFileFingerprints.find(normalized);
    if (it == indexedFileFingerprints.end()) return true;
    if (it->second != fingerprint) return true;
    
    std::string currentHash = getCurrentCommitHash();
    for (const auto& c : chunks) {
        if (c.fileName == normalized) {
            if (c.embedding_version < engine->getInternalVersion()) return true;
            if (c.commit_hash != currentHash) return true;
        }
    }
    return false;
}

void IndexManager::removeChunksForFile(const std::string& filePath) {
    std::string normalized;
    try {
        normalized = fs::absolute(filePath).lexically_normal().string();
    } catch (...) { normalized = filePath; }

    std::unique_lock lock(chunksMutex);
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
            [&](const CodeChunk& c) { return c.fileName == normalized; }),
        chunks.end());
    indexedFileFingerprints.erase(normalized);
}

void IndexManager::init(const std::string& indexPath) {
    FileHandler fh;
    if (!indexPath.empty()) {
        fs::path supplied(indexPath);
        if (fs::exists(supplied) && fs::is_directory(supplied)) {
            indexFilePath = (supplied / "rag_index.bin").string();
        } else {
            indexFilePath = supplied.string();
        }
    } else {
        indexFilePath = fh.getRagPath("rag_index.bin");
    }
    loadAttachmentRegistry();
    loadIndex(indexFilePath);
    rebuildInternalStructures();
}

void IndexManager::indexFile(const std::string& filePath) {
    std::string normalizedPath;
    try {
        normalizedPath = fs::absolute(filePath).lexically_normal().string();
    } catch (...) { normalizedPath = filePath; }

    // STRICT SANDBOX ENFORCEMENT
    if (!isUnderAgentWorkspace(normalizedPath)) {
        std::cerr << "[SECURITY] REJECTED path outside sandbox: " << normalizedPath << "\n";
        return;
    }

    if (!shouldReindexFile(normalizedPath)) {
        return;
    }

    IndexingCompletionGuard completion(this, eventCallback, session_id, normalizedPath);
    if (eventCallback) {
        ControllerEvent ev;
        ev.type = EventType::INDEXING_STARTED;
        ev.session_id = session_id;
        ev.metadata = {{"file_path", normalizedPath}};
        eventCallback(ev);
        completion.arm();
    }

    removeChunksForFile(normalizedPath);
    
    std::ifstream in(normalizedPath, std::ios::binary);
    if (!in) {
        completion.setFailure("read_failed");
        return;
    }
    std::error_code ec;
    const auto fileSize = fs::file_size(normalizedPath, ec);
    if (!ec && fileSize > MAX_FILE_SIZE) {
        completion.setFailure("read_failed");
        return;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string content = ss.str();
    if (isWhitespaceOnlyContent(content)) {
        completion.setFailure("empty_document");
        return;
    }
    if (!engine) {
        completion.setFailure("engine_unavailable");
        return;
    }
    content = sanitize_utf8(content);
    
    if (localTfIdfEngine) {
        localTfIdfEngine->updateVocabulary(content);
    }

    IndexingPathDiagnostics diag;
    diag.input_bytes = content.size();
    diag.paragraph_blocks = countMarkdownParagraphBlocks(content);

    auto storeSmallWholeFileChunk = [&]() -> int {
        CodeChunk fallbackChunk;
        fallbackChunk.fileName = normalizedPath;
        fallbackChunk.startLine = 1;
        fallbackChunk.endLine = 0;
        fallbackChunk.code = content;
        std::error_code ec_m;
        auto f_time = fs::last_write_time(normalizedPath, ec_m);
        fallbackChunk.last_modified =
            std::chrono::duration_cast<std::chrono::milliseconds>(f_time.time_since_epoch())
                .count();
        fallbackChunk.embedding_version = engine->getInternalVersion();
        fallbackChunk.commit_hash = getCurrentCommitHash();
        fallbackChunk.code.erase(std::remove(fallbackChunk.code.begin(), fallbackChunk.code.end(), '\0'),
                                fallbackChunk.code.end());
        try {
            fallbackChunk.embedding = engine->embed(fallbackChunk.code);
            if (localTfIdfEngine) {
                auto tfidf = localTfIdfEngine->embed(fallbackChunk.code);
                float sum = 0;
                for (float v : tfidf) {
                    sum += v * v;
                }
                fallbackChunk.keyword_score = std::sqrt(sum);
            }
        } catch (...) {
        }
        addChunk(std::move(fallbackChunk));
        return countStoredChunksForFile(normalizedPath);
    };

    auto ingestChunkVector = [&](std::vector<CodeChunk>& vec) -> int {
        const size_t BATCH_SIZE = 10;
        for (size_t i = 0; i < vec.size(); i += BATCH_SIZE) {
            const size_t batchEnd = std::min(i + BATCH_SIZE, vec.size());
            std::vector<std::string> batchCodes;
            for (size_t j = i; j < batchEnd; ++j) {
                batchCodes.push_back(vec[j].code);
            }

            auto embeddings = engine->embedBatch(batchCodes);

            for (size_t j = 0; j < batchCodes.size(); ++j) {
                const size_t chunkIdx = i + j;
                CodeChunk& chunkRef = vec[chunkIdx];
                chunkRef.fileName = normalizedPath;

                if (chunkRef.code.empty()
                    || std::all_of(chunkRef.code.begin(), chunkRef.code.end(), ::isspace)) {
                    ++diag.chunks_skipped_empty;
                    continue;
                }
                chunkRef.code.erase(std::remove(chunkRef.code.begin(), chunkRef.code.end(), '\0'),
                                    chunkRef.code.end());

                const size_t nonspace_count = std::count_if(
                    chunkRef.code.begin(), chunkRef.code.end(),
                    [](char c) { return !std::isspace(c); });
                if (nonspace_count < 10) {
                    ++diag.chunks_skipped_short;
                    continue;
                }

                try {
                    if (j < embeddings.size()) {
                        chunkRef.embedding = embeddings[j];
                    } else {
                        chunkRef.embedding = engine->embed(chunkRef.code);
                    }

                    std::error_code ec_m;
                    auto f_time = fs::last_write_time(normalizedPath, ec_m);
                    chunkRef.last_modified =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            f_time.time_since_epoch())
                            .count();
                    chunkRef.embedding_version = engine->getInternalVersion();
                    chunkRef.commit_hash = getCurrentCommitHash();

                    if (localTfIdfEngine) {
                        auto tfidf = localTfIdfEngine->embed(chunkRef.code);
                        float sum = 0;
                        for (float v : tfidf) {
                            sum += v * v;
                        }
                        chunkRef.keyword_score = std::sqrt(sum);
                    }
                } catch (...) {
                    ++diag.chunks_skipped_embed;
                    continue;
                }
                addChunk(std::move(chunkRef));
            }
            if (i % 20 == 0) {
                std::cout << "." << std::flush;
            }
        }
        return countStoredChunksForFile(normalizedPath);
    };

    auto finalizeIndexing = [&](int stored) {
        diag.final_stored = stored;
        logIndexingDiagnostics(normalizedPath, diag);
        {
            std::unique_lock lock(chunksMutex);
            indexedFileFingerprints[normalizedPath] = computeFileFingerprint(normalizedPath);
        }
        if (stored > 0) {
            completion.setSuccess(stored);
        } else {
            completion.setFailure("no_chunks");
        }
    };

    auto chunksVec = Chunker::createSmartChunks(normalizedPath, content);
    if (chunksVec.empty()) {
        chunksVec = Chunker::chunkBySize(normalizedPath, content);
    }
    diag.chunks_generated = chunksVec.size();

    if (chunksVec.empty()) {
        int stored = 0;
        if (content.size() <= kSmallDocumentSingleChunkMaxBytes) {
            diag.fallback = "whole_small";
            stored = storeSmallWholeFileChunk();
        }
        finalizeIndexing(stored);
        return;
    }

    int stored = ingestChunkVector(chunksVec);
    diag.chunks_stored_after_primary = static_cast<size_t>(stored);

    enforceMemoryLimits();
    stored = countStoredChunksForFile(normalizedPath);

    if (stored == 0) {
        if (content.size() <= kSmallDocumentSingleChunkMaxBytes) {
            diag.fallback = "whole_small";
            stored = storeSmallWholeFileChunk();
        } else {
            diag.fallback = "chunk_by_size";
            auto sizeChunks = Chunker::chunkBySize(normalizedPath, content);
            diag.chunks_generated = sizeChunks.size();
            stored = ingestChunkVector(sizeChunks);
        }
    }

    finalizeIndexing(stored);
}

void IndexManager::indexProject(const std::string& rootPath) {
    if (!fs::exists(rootPath) || !fs::is_directory(rootPath)) return;
    std::string rootAbs = fs::absolute(rootPath).lexically_normal().string();

    // STRICT SANDBOX ENFORCEMENT
    if (!isUnderAgentWorkspace(rootAbs)) {
        std::cerr << "[SECURITY] REJECTED rootPath outside sandbox: " << rootAbs << "\n";
        return;
    }

    int successCount = 0, errorCount = 0, skippedCount = 0;
    constexpr size_t MAX_FILES_TO_INDEX = 2000;
    size_t filesProcessed = 0;
    
    std::cout << "[RAG] Starting recursive index of: " << rootAbs << "\n";

    try {
        for (const auto& entry : fs::recursive_directory_iterator(rootPath)) {
            if (filesProcessed >= MAX_FILES_TO_INDEX) break;
            if (!entry.is_regular_file()) continue;
            
            std::string fullPath = fs::absolute(entry.path()).lexically_normal().string();
            std::string filename = entry.path().filename().string();
            
            if (filename == "rag_index.bin") continue;
            if (fullPath.find("/build/") != std::string::npos) continue;
            if (fullPath.find("/.git/") != std::string::npos) continue;
            if (fullPath.find("/docs/") != std::string::npos) continue;

            auto ext = entry.path().extension().string();
            if (isSupportedExtension(ext)) {
                try {
                    if (!shouldReindexFile(fullPath)) { 
                        skippedCount++; 
                        continue; 
                    }
                    indexFile(fullPath);
                    successCount++;
                    filesProcessed++;
                    if (filesProcessed % 50 == 0) {
                        std::cout << "[RAG] ... processed " << filesProcessed << " files\n";
                    }
                } catch (...) { errorCount++; }
            }
        }
    } catch (const std::exception& e) { 
        std::cerr << "[RAG] Critical error during indexProject: " << e.what() << "\n";
    }
    
    std::cout << "[RAG] Indexed " << rootAbs << " - Success: " << successCount << ", Skipped: " << skippedCount << ", Errors: " << errorCount << "\n";
}

void IndexManager::saveIndex() const {
    FileHandler fh;
    saveIndex(fh.getRagPath("rag_index.bin"));
}

void IndexManager::saveIndex(const std::string& dbPath) const {
    std::filesystem::create_directories(std::filesystem::path(dbPath).parent_path());
    std::ofstream out(dbPath, std::ios::binary | std::ios::trunc);
    if (!out) return;

    json header;
    header["magic"] = 0x54484F54;
    header["model_name"] = engine->getModelName();
    header["embedding_dimension"] = engine->getDimension();
    header["embedding_version"] = engine->getInternalVersion();
    std::string headerStr = header.dump();
    size_t headerLen = headerStr.size();
    out.write(reinterpret_cast<const char*>(&headerLen), sizeof(headerLen));
    out.write(headerStr.data(), headerLen);

    size_t n = chunks.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto& c : chunks) {
        size_t len;
        len = c.fileName.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len)); out.write(c.fileName.data(), len);
        len = c.symbolName.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len)); out.write(c.symbolName.data(), len);
        out.write(reinterpret_cast<const char*>(&c.startLine), sizeof(c.startLine));
        out.write(reinterpret_cast<const char*>(&c.endLine), sizeof(c.endLine));
        len = c.code.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len)); out.write(c.code.data(), len);
        len = c.embedding.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        if (len > 0) out.write(reinterpret_cast<const char*>(c.embedding.data()), len * sizeof(float));
        out.write(reinterpret_cast<const char*>(&c.last_modified), sizeof(c.last_modified));
        len = c.commit_hash.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        if (len > 0) out.write(c.commit_hash.data(), len);
        out.write(reinterpret_cast<const char*>(&c.embedding_version), sizeof(c.embedding_version));
        out.write(reinterpret_cast<const char*>(&c.keyword_score), sizeof(c.keyword_score));
    }
    
    // Save Primary Engine State
    {
        std::string tmpFile = dbPath + ".engine_tmp";
        engine->saveState(tmpFile);
        std::ifstream engIn(tmpFile, std::ios::binary);
        std::string engData((std::istreambuf_iterator<char>(engIn)), std::istreambuf_iterator<char>());
        size_t engSize = engData.size();
        out.write(reinterpret_cast<const char*>(&engSize), sizeof(engSize));
        out.write(engData.data(), engSize);
        std::filesystem::remove(tmpFile);
    }

    // Phase 13 Fix: Save Local TF-IDF Engine State
    if (localTfIdfEngine) {
        std::string tmpFile = dbPath + ".tfidf_tmp";
        localTfIdfEngine->saveState(tmpFile);
        std::ifstream engIn(tmpFile, std::ios::binary);
        std::string engData((std::istreambuf_iterator<char>(engIn)), std::istreambuf_iterator<char>());
        size_t engSize = engData.size();
        out.write(reinterpret_cast<const char*>(&engSize), sizeof(engSize));
        out.write(engData.data(), engSize);
        std::filesystem::remove(tmpFile);
    } else {
        size_t zero = 0;
        out.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    }

    std::cout << "[basic_agent:RAG] Index saved to: " << dbPath << " (entries=" << n << ")\n";
} 

void IndexManager::loadIndex() {
    FileHandler fh;
    loadIndex(fh.getRagPath("rag_index.bin"));
}

void IndexManager::loadIndex(const std::string& dbPath) {
    std::ifstream in(dbPath, std::ios::binary);
    if (!in) return;

    size_t headerLen = 0;
    in.read(reinterpret_cast<char*>(&headerLen), sizeof(headerLen));
    if (headerLen > 0 && headerLen < 1024*1024) {
        std::string headerStr(headerLen, '\0');
        in.read(&headerStr[0], headerLen);
        try {
            json header = json::parse(headerStr);
            bool mismatch = false;
            if (header.value("model_name", "") != engine->getModelName()) mismatch = true;
            if (header.value("embedding_dimension", 0) != engine->getDimension()) mismatch = true;
            if (header.value("embedding_version", 0) != engine->getInternalVersion()) mismatch = true;

            if (mismatch) {
                std::cout << "[RAG] Index metadata mismatch detected.\n";
                return;
            }
        } catch (...) {}
    }

    size_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    {
        std::unique_lock lock(chunksMutex);
        chunks.clear(); 
        codeToChunkIndex.clear();
        chunks.reserve(n);
    }
    for (size_t i = 0; i < n; ++i) {
        CodeChunk c; size_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len)); c.fileName.resize(len); in.read(&c.fileName[0], len);
        in.read(reinterpret_cast<char*>(&len), sizeof(len)); c.symbolName.resize(len); in.read(&c.symbolName[0], len);
        in.read(reinterpret_cast<char*>(&c.startLine), sizeof(c.startLine));
        in.read(reinterpret_cast<char*>(&c.endLine), sizeof(c.endLine));
        in.read(reinterpret_cast<char*>(&len), sizeof(len)); c.code.resize(len); in.read(&c.code[0], len);
        size_t embLen; in.read(reinterpret_cast<char*>(&embLen), sizeof(embLen)); c.embedding.resize(embLen);
        if (embLen > 0) in.read(reinterpret_cast<char*>(c.embedding.data()), embLen * sizeof(float));
        in.read(reinterpret_cast<char*>(&c.last_modified), sizeof(c.last_modified));
        size_t hashLen; in.read(reinterpret_cast<char*>(&hashLen), sizeof(hashLen));
        if (hashLen > 0) { c.commit_hash.resize(hashLen); in.read(&c.commit_hash[0], hashLen); }
        in.read(reinterpret_cast<char*>(&c.embedding_version), sizeof(c.embedding_version));
        in.read(reinterpret_cast<char*>(&c.keyword_score), sizeof(c.keyword_score));
        try { c.fileName = fs::absolute(c.fileName).lexically_normal().string(); } catch (...) {}
        {
            std::unique_lock lock(chunksMutex);
            codeToChunkIndex[c.code] = chunks.size();
            chunks.push_back(std::move(c));
            indexedFileFingerprints[chunks.back().fileName] = computeFileFingerprint(chunks.back().fileName);
        }
    }
    
    // Load Primary Engine State
    size_t engSize;
    in.read(reinterpret_cast<char*>(&engSize), sizeof(engSize));
    if (engSize > 0) {
        std::string engData(engSize, '\0');
        in.read(&engData[0], engSize);
        std::string tmpFile = dbPath + ".engine_tmp";
        { std::ofstream tmpOut(tmpFile, std::ios::binary); tmpOut.write(engData.data(), engSize); }
        engine->loadState(tmpFile);
        std::filesystem::remove(tmpFile);
    }

    // Phase 13 Fix: Load Local TF-IDF Engine State
    size_t tfidfSize;
    in.read(reinterpret_cast<char*>(&tfidfSize), sizeof(tfidfSize));
    if (tfidfSize > 0 && localTfIdfEngine) {
        std::string engData(tfidfSize, '\0');
        in.read(&engData[0], tfidfSize);
        std::string tmpFile = dbPath + ".tfidf_tmp";
        { std::ofstream tmpOut(tmpFile, std::ios::binary); tmpOut.write(engData.data(), tfidfSize); }
        localTfIdfEngine->loadState(tmpFile);
        std::filesystem::remove(tmpFile);
    }

    {
        std::unique_lock lock(chunksMutex);
        store.clear();
        for (size_t i = 0; i < chunks.size(); ++i) {
            auto& c = chunks[i];
            if (!c.embedding.empty() && !c.code.empty()) {
                store.addDocumentWithEmbedding(c.code, c.embedding);
            }
        }
        classifyAllChunksMetadataUnlocked();
    }
    std::cout << "[basic_agent:RAG] Index loaded from: " << dbPath << " (entries=" << n << ")\n";
}

void IndexManager::rebuildInternalStructures() {
    std::unique_lock lock(chunksMutex);
    store.clear();
    codeToChunkIndex.clear();
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        if (chunk.code.empty() || chunk.embedding.empty()) continue;
        if (codeToChunkIndex.find(chunk.code) != codeToChunkIndex.end()) continue;
        store.addDocumentWithEmbedding(chunk.code, chunk.embedding);
        codeToChunkIndex[chunk.code] = i;
    }
}

void IndexManager::addChunkToIndex(CodeChunk&& chunk) {
    addChunk(std::move(chunk));
}

int IndexManager::countStoredChunksForFile(const std::string& normalizedPath) const {
    int count = 0;
    std::shared_lock lock(chunksMutex);
    for (const auto& chunk : chunks) {
        if (chunk.fileName == normalizedPath) {
            ++count;
        }
    }
    return count;
}

void IndexManager::recordIndexingOutcome(const std::string& normalizedPath,
                                         bool success,
                                         int /*chunk_count*/,
                                         const std::string& reason) {
    std::unique_lock lock(outcomesMutex_);
    if (success) {
        indexingFailureReasons_.erase(normalizedPath);
        return;
    }
    if (!reason.empty()) {
        indexingFailureReasons_[normalizedPath] = reason;
    } else {
        indexingFailureReasons_[normalizedPath] = "no_chunks";
    }
}

nlohmann::json IndexManager::listCorpusDocuments(const std::string& ragDirectory) const {
    using namespace Thoth::CorpusDocuments;

    struct DocAgg {
        int chunk_count = 0;
        std::int64_t last_modified = 0;
        bool has_chunks = false;
    };

    std::map<std::string, DocAgg> documents;

    {
        std::shared_lock lock(chunksMutex);
        for (const auto& chunk : chunks) {
            if (chunk.fileName.empty()) {
                continue;
            }
            auto& agg = documents[chunk.fileName];
            agg.chunk_count++;
            agg.has_chunks = true;
            if (chunk.last_modified > agg.last_modified) {
                agg.last_modified = chunk.last_modified;
            }
        }
        for (const auto& entry : indexedFileFingerprints) {
            (void)documents[entry.first];
        }
    }

    std::string rag_root;
    try {
        if (!ragDirectory.empty() && fs::exists(ragDirectory)) {
            rag_root = fs::absolute(ragDirectory).lexically_normal().string();
        }
    } catch (...) {
        rag_root.clear();
    }

    if (!rag_root.empty()) {
        try {
            for (const auto& entry : fs::recursive_directory_iterator(rag_root)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                const std::string ext = entry.path().extension().string();
                if (!isSupportedExtension(ext)) {
                    continue;
                }
                const std::string abs_path =
                    entry.path().lexically_normal().string();
                (void)documents[abs_path];
            }
        } catch (const std::exception& ex) {
            std::cerr << "[RAG] listCorpusDocuments scan failed: " << ex.what() << "\n";
        }
    }

    nlohmann::json out = emptyV1List();
    for (const auto& [storage_path, agg] : documents) {
        std::string storage_key = storage_path;
        if (!rag_root.empty()) {
            try {
                storage_key = fs::relative(storage_path, rag_root).lexically_normal().string();
            } catch (...) {
                storage_key = fs::path(storage_path).filename().string();
            }
        } else {
            storage_key = fs::path(storage_path).filename().string();
        }

        const std::string display_name = fs::path(storage_path).filename().string();
        std::string status;
        std::optional<std::string> failure_reason;
        if (agg.has_chunks) {
            status = "indexed";
        } else {
            std::shared_lock lock(outcomesMutex_);
            const auto fail_it = indexingFailureReasons_.find(storage_path);
            if (fail_it != indexingFailureReasons_.end()) {
                status = "failed";
                failure_reason = fail_it->second;
            } else {
                status = "pending";
            }
        }
        std::optional<int> chunk_count;
        if (agg.has_chunks && agg.chunk_count > 0) {
            chunk_count = agg.chunk_count;
        }
        out["documents"].push_back(makeDocument(
            stableDocumentId(storage_key),
            display_name,
            status,
            formatIndexedAtIso(agg.last_modified),
            chunk_count,
            failure_reason));
    }

    return out;
}

IndexManager::CreateCorpusDocumentResult IndexManager::createCorpusDocument(
    const std::string& ragDirectory,
    const std::string& suggested_name,
    const std::string& content,
    const std::string& owner_context_id) {
    using namespace Thoth::CorpusCreate;
    using namespace Thoth::CorpusDocuments;

    CreateCorpusDocumentResult result;
    if (content.empty()) {
        result.error = "content must not be empty";
        return result;
    }
    if (content.size() > MAX_FILE_SIZE) {
        result.error = "content exceeds maximum size";
        return result;
    }

    std::string rag_root;
    try {
        if (ragDirectory.empty()) {
            result.error = "rag directory unavailable";
            return result;
        }
        fs::create_directories(ragDirectory);
        rag_root = fs::absolute(ragDirectory).lexically_normal().string();
        if (!isUnderAgentWorkspace(rag_root)) {
            result.error = "rag directory outside sandbox";
            return result;
        }
    } catch (const std::exception& ex) {
        result.error = std::string("rag directory error: ") + ex.what();
        return result;
    }

    const std::string base_name = sanitizeSuggestedFilename(suggested_name);
    const fs::path ext_path(base_name);
    const std::string ext = ext_path.extension().string();
    if (!ext.empty() && !isSupportedExtension(ext)) {
        result.error = "unsupported file extension";
        return result;
    }

    fs::path final_path = fs::path(rag_root) / base_name;
    bool replacing_session_attachment = false;

    const auto adopt_session_owned_path =
        [&](const std::string& owner_id) -> bool {
            if (owner_id.empty()) {
                return false;
            }
            const auto existing = findSessionOwnedAttachmentPath(owner_id, base_name);
            if (!existing) {
                return false;
            }
            final_path = fs::path(*existing);
            replacing_session_attachment = true;
            try {
                const std::string normalized =
                    fs::absolute(final_path).lexically_normal().string();
                std::unique_lock lock(outcomesMutex_);
                indexingFailureReasons_.erase(normalized);
            } catch (...) {
            }
            return true;
        };

    if (!owner_context_id.empty()) {
        if (!adopt_session_owned_path(owner_context_id)
            && owner_context_id != "default") {
            (void)adopt_session_owned_path("default");
        }
    }

    if (!replacing_session_attachment && fs::exists(final_path)) {
        const std::string stem = ext_path.stem().string();
        const std::string suffix_ext = ext.empty() ? std::string(".txt") : ext;
        int attempt = 1;
        do {
            final_path = fs::path(rag_root)
                / (stem + "_" + std::to_string(attempt) + suffix_ext);
            ++attempt;
        } while (fs::exists(final_path) && attempt < 10000);
        if (fs::exists(final_path)) {
            result.error = "could not allocate unique document name";
            return result;
        }
    }

    const std::string temp_path =
        final_path.string() + ".tmp." + std::to_string(getpid());
    try {
        {
            std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
            if (!out) {
                result.error = "failed to write document";
                return result;
            }
            out.write(content.data(), static_cast<std::streamsize>(content.size()));
            if (!out) {
                std::error_code ec;
                fs::remove(temp_path, ec);
                result.error = "failed to write document";
                return result;
            }
        }
        std::error_code ec;
        fs::rename(temp_path, final_path, ec);
        if (ec) {
            fs::remove(temp_path, ec);
            result.error = "failed to finalize document";
            return result;
        }
    } catch (const std::exception& ex) {
        std::error_code ec;
        fs::remove(temp_path, ec);
        result.error = std::string("failed to store document: ") + ex.what();
        return result;
    }

    std::string storage_key;
    try {
        storage_key = fs::relative(final_path, rag_root).lexically_normal().string();
    } catch (...) {
        storage_key = final_path.filename().string();
    }

    result.document_name = final_path.filename().string();
    result.document_id = stableDocumentId(storage_key);
    result.ok = true;

    const std::string stored_path = final_path.string();
    if (!owner_context_id.empty()) {
        registerAttachmentOwner(stored_path, owner_context_id);
    }
    indexFileAsync(stored_path);
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_taskQueue.push([this]() { saveIndex(); });
    }
    m_queueCv.notify_one();

    return result;
}
