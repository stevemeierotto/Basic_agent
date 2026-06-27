/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C2 Phase 1: golden chat-RAG retrieval benchmark
 *
 * Retrieval-only evaluation (same path as processQuery). Requires Ollama for embeddings.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/chat_rag_golden_cases.h"
#include "../include/config.h"
#include "../include/embedding_engine.h"
#include "../include/index_manager.h"
#include "../include/memory.h"
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

bool ollamaReachable() {
    return std::system("curl -sf http://localhost:11434/api/tags >/dev/null 2>&1") == 0;
}

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string benchmarkLogPath() {
    FileHandler fh;
    fs::path logsDir = fs::path(fh.getProjectRoot()) / "logs";
    fs::create_directories(logsDir);
    return (logsDir / "chat_rag_benchmark.jsonl").string();
}

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

    if (!ollamaReachable()) {
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
        {"case_count", cases.size()},
        {"hit_at_1", hitsAt1},
        {"mean_ndcg_at_1", meanNdcg1},
        {"mean_mrr", meanMrr},
        {"corpus_files", corpusPaths},
    });

    std::cout << "\nWrote " << (cases.size() + 1) << " lines to " << logPath << '\n';

    return hitsAt1 == static_cast<int>(cases.size()) ? 0 : 2;
}
