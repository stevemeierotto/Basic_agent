/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C2 Phase 1: golden chat-RAG retrieval benchmark
 *
 * Retrieval-only evaluation (same path as processQuery). Requires Ollama for embeddings.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/benchmark_context.h"
#include "../include/chat_rag_golden_cases.h"
#include "../include/config.h"
#include "../include/embedding_engine.h"
#include "../include/index_manager.h"
#include "../include/memory.h"
#include "../include/ollama_snapshot.h"
#include "../include/rag.h"
#include "file_handler.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../include/json.hpp"

namespace fs = std::filesystem;

namespace {

constexpr int kTopK = 5;

std::string fileBasename(const std::string& path) {
    try {
        return fs::path(path).filename().string();
    } catch (...) {
        return path;
    }
}

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string benchmarkLogPath() {
    FileHandler fh;
    return fh.getLogsPath("chat_rag_benchmark.jsonl");
}

Thoth::BenchmarkEnvironmentInputs makeChatRagBenchmarkInputs(
    EmbeddingEngine* engine,
    IndexManager* idx,
    const std::vector<std::string>& corpusPaths,
    const Config& config) {
    Thoth::BenchmarkEnvironmentInputs inputs;
    inputs.harness = "chat_rag_benchmark";
    inputs.tier = Thoth::BenchmarkTier::OLLAMA;
    inputs.model.llm_model = config.llm_model;
    inputs.model.embedding_model = config.embedding_model;
    if (engine) {
        inputs.model.embedding_method = "External";
        inputs.model.embedding_dimension = engine->getDimension();
        inputs.model.embedding_internal_version = engine->getInternalVersion();
    }
    inputs.corpus_paths = corpusPaths;
    inputs.corpus_mode = Thoth::CorpusFingerprintMode::FAST;
    inputs.corpus_chunk_count = idx ? static_cast<int>(idx->getChunks().size()) : 0;
    inputs.thoth_env_flags = Thoth::collectThothEnvFlags();
    inputs.ollama_reachable = Thoth::isOllamaReachable();
    if (inputs.ollama_reachable) {
        if (auto snap = Thoth::fetchOllamaSnapshot()) {
            inputs.ollama = *snap;
        }
    }
    return inputs;
}

Thoth::IndexEnvironment indexEnvironmentFrom(EmbeddingEngine* engine, IndexManager* idx) {
    Thoth::IndexEnvironment index;
    if (!engine || !idx) {
        return index;
    }
    index.rag_index_header = {
        {"model_name", engine->getModelName()},
        {"embedding_dimension", engine->getDimension()},
        {"embedding_version", engine->getInternalVersion()},
        {"chunk_count", static_cast<int>(idx->getChunks().size())},
    };
    return index;
}

/** RAII: emit CHAT_RAG_BENCHMARK_COMPLETE on normal exit, CHAT_RAG_BENCHMARK_ABORTED if scope exits early. */
class ChatRagRunRecorder {
public:
    explicit ChatRagRunRecorder(Thoth::BenchmarkRun& run) : run_(run) {}

    ~ChatRagRunRecorder() {
        if (!finished_) {
            run_.emit("CHAT_RAG_BENCHMARK_ABORTED", payload());
        }
    }

    void complete(int hitsAt1, std::size_t caseCount, float meanNdcg1, float meanMrr) {
        hits_at_1_ = hitsAt1;
        case_count_ = caseCount;
        mean_ndcg_at_1_ = meanNdcg1;
        mean_mrr_ = meanMrr;
        run_.emit("CHAT_RAG_BENCHMARK_COMPLETE", payload());
        finished_ = true;
    }

private:
    nlohmann::json payload() const {
        return {{"hits_at_1", hits_at_1_},
                {"case_count", case_count_},
                {"mean_ndcg_at_1", mean_ndcg_at_1_},
                {"mean_mrr", mean_mrr_}};
    }

    Thoth::BenchmarkRun& run_;
    int hits_at_1_ = 0;
    std::size_t case_count_ = 0;
    float mean_ndcg_at_1_ = 0.0f;
    float mean_mrr_ = 0.0f;
    bool finished_ = false;
};

struct CaseResult {
    Thoth::ChatRagGoldenCase spec;
    std::vector<std::pair<std::string, float>> rankedDocs;
    std::string topFile;
    float topScore = 0.0f;
    int topRank = 0;
    bool hitAt1 = false;
    float ndcgAt1 = 0.0f;
    float mrr = 0.0f;
};

CaseResult evaluateCase(RAGPipeline& rag, const Thoth::ChatRagGoldenCase& spec) {
    CaseResult result;
    result.spec = spec;

    GragDiagnostics diagnostics;
    const auto chunks = rag.retrieveRelevant(spec.query, {}, kTopK, spec.id, {}, {}, {}, {}, {}, &diagnostics);

    std::unordered_map<std::string, float> bestScoreByFile;
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        const std::string base = fileBasename(chunks[i].fileName);
        float score = 0.0f;
        if (i < diagnostics.breakdowns.size()) {
            score = diagnostics.breakdowns[i].final_score;
        }
        auto it = bestScoreByFile.find(base);
        if (it == bestScoreByFile.end() || score > it->second) {
            bestScoreByFile[base] = score;
        }
    }

    result.rankedDocs.reserve(bestScoreByFile.size());
    for (const auto& [file, score] : bestScoreByFile) {
        result.rankedDocs.push_back({file, score});
    }
    std::sort(result.rankedDocs.begin(), result.rankedDocs.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    for (std::size_t i = 0; i < result.rankedDocs.size(); ++i) {
        if (result.rankedDocs[i].first == spec.expected_top_file) {
            result.topRank = static_cast<int>(i) + 1;
            result.mrr = 1.0f / static_cast<float>(result.topRank);
            break;
        }
    }

    if (!result.rankedDocs.empty()) {
        result.topFile = result.rankedDocs.front().first;
        result.topScore = result.rankedDocs.front().second;
    }

    result.hitAt1 = !result.rankedDocs.empty() && result.rankedDocs.front().first == spec.expected_top_file;
    result.ndcgAt1 = result.hitAt1 ? 1.0f : 0.0f;
    return result;
}

void appendJsonLine(const std::string& path, const nlohmann::json& event) {
    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << event.dump() << '\n';
    }
}

} // namespace

int main() {
    std::cout << "C2 Phase 1 — Chat RAG Golden Corpus Benchmark\n";

    if (!Thoth::isOllamaReachable()) {
        std::cerr << "[FAIL] Ollama not reachable at localhost:11434 (embeddings required).\n";
        return 1;
    }

    const auto corpusPaths = Thoth::getChatRagGoldenCorpusPaths();
    if (corpusPaths.size() < 4) {
        std::cerr << "[FAIL] Golden corpus incomplete. Expected 4 markdown files under agent_workspace/rag/.\n";
        std::cerr << "       Found " << corpusPaths.size() << " file(s).\n";
        for (const auto& path : corpusPaths) {
            std::cerr << "         - " << path << '\n';
        }
        return 1;
    }

    Config config;
    Memory memory(config);
    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::External, &config);
    IndexManager indexManager(engine.get());

    std::cout << "Indexing golden corpus (" << corpusPaths.size() << " files)...\n";
    indexManager.clear();
    for (const auto& path : corpusPaths) {
        std::cout << "  " << fileBasename(path) << '\n';
        indexManager.indexFile(path);
    }
    indexManager.setActiveCorpusFiles(corpusPaths);
    std::cout << "Total chunks: " << indexManager.getChunks().size() << '\n';

    Thoth::BenchmarkRun benchmarkRun = Thoth::BenchmarkRun::create(
        makeChatRagBenchmarkInputs(engine.get(), &indexManager, corpusPaths, config));
    benchmarkRun.bindIndex(indexEnvironmentFrom(engine.get(), &indexManager));
    const Thoth::BenchmarkAttribution suiteAttribution = benchmarkRun.attribution();

    std::cout << "BENCHMARK_ENV run_id=" << benchmarkRun.run_id()
              << " env_hash=" << benchmarkRun.environment_hash()
              << " index_hash=" << benchmarkRun.index_hash() << " tier=ollama\n";

    ChatRagRunRecorder suiteRecorder(benchmarkRun);

    if (const char* abortSmoke = std::getenv("THOTH_CHAT_RAG_BENCHMARK_ABORT_SMOKE");
        abortSmoke && (std::string(abortSmoke) == "1" || std::string(abortSmoke) == "true")) {
        std::cerr << "CHAT_RAG: benchmark abort smoke — exiting before complete()\n";
        return 2;
    }

    RAGPipeline rag(std::move(engine), &indexManager, &config, &memory);

    const auto cases = Thoth::getChatRagGoldenCases();
    std::vector<CaseResult> results;
    results.reserve(cases.size());

    float meanNdcg1 = 0.0f;
    float meanMrr = 0.0f;
    int hitsAt1 = 0;

    std::cout << "\nRunning " << cases.size() << " golden queries (top_k=" << kTopK << ")...\n\n";

    for (const auto& spec : cases) {
        CaseResult result = evaluateCase(rag, spec);
        results.push_back(result);
        meanNdcg1 += result.ndcgAt1;
        meanMrr += result.mrr;
        hitsAt1 += result.hitAt1 ? 1 : 0;

        std::cout << spec.id << "  query: " << spec.query << '\n';
        std::cout << "  expected top: " << spec.expected_top_file << '\n';
        for (std::size_t i = 0; i < result.rankedDocs.size(); ++i) {
            std::cout << "    " << (i + 1) << ". " << result.rankedDocs[i].first
                      << "  score " << std::fixed << std::setprecision(3) << result.rankedDocs[i].second << '\n';
        }
        std::cout << "  hit@1: " << (result.hitAt1 ? "YES" : "NO")
                  << "  nDCG@1: " << result.ndcgAt1 << "  MRR: " << result.mrr << "\n\n";
    }

    meanNdcg1 /= static_cast<float>(cases.size());
    meanMrr /= static_cast<float>(cases.size());

    std::cout << "Summary\n";
    std::cout << "  cases: " << cases.size() << '\n';
    std::cout << "  hit@1: " << hitsAt1 << '/' << cases.size() << '\n';
    std::cout << "  mean nDCG@1: " << std::fixed << std::setprecision(3) << meanNdcg1 << '\n';
    std::cout << "  mean MRR: " << meanMrr << '\n';

    const std::string logPath = benchmarkLogPath();
    const std::int64_t ts = nowMs();

    for (const auto& result : results) {
        nlohmann::json ranked = nlohmann::json::array();
        for (std::size_t i = 0; i < result.rankedDocs.size(); ++i) {
            ranked.push_back({
                {"rank", static_cast<int>(i) + 1},
                {"file", result.rankedDocs[i].first},
                {"score", result.rankedDocs[i].second},
            });
        }

        appendJsonLine(logPath, {
            {"event", "CHAT_RAG_BENCHMARK_CASE"},
            {"timestamp_ms", ts},
            {"run_id", suiteAttribution.run_id},
            {"env_hash", suiteAttribution.env_hash},
            {"case_id", result.spec.id},
            {"query", result.spec.query},
            {"expected_top_file", result.spec.expected_top_file},
            {"top_k", kTopK},
            {"ranked_documents", ranked},
            {"actual_top_file", result.topFile},
            {"actual_top_score", result.topScore},
            {"hit_at_1", result.hitAt1},
            {"ndcg_at_1", result.ndcgAt1},
            {"mrr", result.mrr},
        });
    }

    appendJsonLine(logPath, {
        {"event", "CHAT_RAG_BENCHMARK_SUMMARY"},
        {"timestamp_ms", ts},
        {"run_id", suiteAttribution.run_id},
        {"env_hash", suiteAttribution.env_hash},
        {"case_count", cases.size()},
        {"hit_at_1", hitsAt1},
        {"mean_ndcg_at_1", meanNdcg1},
        {"mean_mrr", meanMrr},
        {"corpus_files", corpusPaths},
    });

    std::cout << "\nWrote " << (cases.size() + 1) << " lines to " << logPath << '\n';

    suiteRecorder.complete(hitsAt1, cases.size(), meanNdcg1, meanMrr);
    return hitsAt1 == static_cast<int>(cases.size()) ? 0 : 2;
}
