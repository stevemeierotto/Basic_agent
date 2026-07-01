#include "../include/rag.h"
#include "../include/decision_trace.h"
#include "../include/file_handler.h"
#include "../include/chunkers/chunker.h"
#include "../include/grag_scorer.h"
#include "../include/logger.h"
#include "../include/grag_metrics.h"
#include "../include/memory.h"
#include "../include/chat_retrieval_boost.h"
#include "../include/chat_retrieval_config.h"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <shared_mutex>
#include <mutex>
#include <chrono>
#include <fstream>

namespace fs = std::filesystem;

RAGPipeline::RAGPipeline(std::unique_ptr<EmbeddingEngine> eng, IndexManager* idx, Config* cfg, Memory* mem)
    : engine(std::move(eng)), indexManager(idx), config(cfg), memory(mem) {
    
    if (config) {
        retrievalConfig.wq = config->wq;
        retrievalConfig.wd = config->wd;
        retrievalConfig.wt = config->wt;
        retrievalConfig.keyword_weight = config->keyword_weight;
        retrievalConfig.graph_weight = config->graph_weight;
    }
}

std::vector<CodeChunk> RAGPipeline::retrieveRelevant(const std::string& query, 
                                                   const std::vector<int>& /*errorLines*/, 
                                                   int topK,
                                                   const std::string& requestId,
                                                   const std::string& pId,
                                                   const std::string& sId,
                                                   const std::vector<float>& g_emb,
                                                   const std::vector<float>& c_emb,
                                                   const std::vector<float>& t_emb,
                                                   GragDiagnostics* outDiagnostics) {
    
    std::vector<CodeChunk> finalMatches;
    if (!indexManager) {
        if (outDiagnostics) {
            *outDiagnostics = GragDiagnostics{};
        }
        return finalMatches;
    }

    // Use passed embeddings if provided, otherwise fallback to internal state
    const std::vector<float>& activeGoal = g_emb.empty() ? goalEmbedding : g_emb;
    const std::vector<float>& activeCurrent = c_emb.empty() ? currentEmbedding : c_emb;
    const std::vector<float>& activeTraj = t_emb.empty() ? trajectoryEmbedding : t_emb;

    GragDiagnostics diagnostics;

    // Sync weights
    if (config && retrievalConfig.mode == RetrievalMode::AUTO) {
        retrievalConfig.wq = config->wq;
        retrievalConfig.wd = config->wd;
        retrievalConfig.wt = config->wt;
        retrievalConfig.keyword_weight = config->keyword_weight;
        retrievalConfig.graph_weight = config->graph_weight;
        retrievalConfig.grag_directional = config->grag_directional;
    }

    int recallK = std::max(topK * 4, 40);
    if (activeGoal.empty()) {
        recallK = std::max(topK * 12, 120);
    }
    
    // Note: indexManager->retrieveChunks usually handles its own locking for the search,
    // but we need the chunks themselves to stay valid while we rescore.
    auto rawResults = indexManager->retrieveChunks(query, recallK);

    std::vector<std::pair<CodeChunk, float>> rag_results;
    for (const auto& [chunkCode, score] : rawResults) {
        const auto* chunk = indexManager->getChunkByCode(chunkCode);
        if (chunk) rag_results.push_back({*chunk, score});
    }

    if (memory && engine) {
        const std::vector<float> q_emb = engine->embed(query);
        const bool goalDirected = !activeGoal.empty() || !pId.empty();
        const auto warmRows = goalDirected
                                  ? memory->searchWarmMemoryAllSessions(q_emb, std::max(topK, 3))
                                  : memory->searchWarmMemory(q_emb, std::max(topK, 3));
        float best_index_score = 0.0f;
        for (const auto& [chunk, score] : rag_results) {
            if (chunk.fileName.rfind("warm_memory:", 0) != 0) {
                best_index_score = std::max(best_index_score, score);
            }
        }
        auto querySharesToken = [](const std::string& query, const std::string& text) {
            std::istringstream iss(query);
            std::string word;
            while (iss >> word) {
                if (word.size() < 4) {
                    continue;
                }
                if (text.find(word) != std::string::npos) {
                    return true;
                }
            }
            return false;
        };
        for (const auto& row : warmRows) {
            if (row.rendered_summary.empty() || row.embedding.empty()) {
                continue;
            }
            if (goalDirected && !querySharesToken(query, row.rendered_summary)) {
                continue;
            }
            const float warm_score =
                GragScorer::cosine_similarity(q_emb, row.embedding) *
                (0.5f + 0.5f * row.importance);
            if (goalDirected && best_index_score > warm_score) {
                continue;
            }
            CodeChunk chunk;
            chunk.fileName = "warm_memory:" + row.id;
            chunk.symbolName = row.session_id;
            chunk.code = row.rendered_summary;
            chunk.embedding = row.embedding;
            rag_results.push_back({chunk, warm_score});
        }
    }

    if (rag_results.empty()) {
        if (outDiagnostics) {
            *outDiagnostics = diagnostics;
        }
        return finalMatches;
    }

    if (activeGoal.empty() && !query.empty()) {
        auto filenameTokens = Thoth::ChatRetrieval::extractFilenameTokens(query);
        if (Thoth::ChatRetrieval::isUsageQuery(query)) {
            filenameTokens.push_back("howto");
        }
        if (!filenameTokens.empty()) {
            Thoth::ChatRetrieval::ensureFilenameCoverage(
                indexManager, filenameTokens, query, rag_results, 3);
        }
    }

    try {
        const bool goalDirected = !activeGoal.empty() || !pId.empty();
        bool use_grag = false;
        if (retrievalConfig.mode == RetrievalMode::AUTO) {
            use_grag = goalDirected;
        } else if (retrievalConfig.mode == RetrievalMode::GRAG) {
            use_grag = true;
        }

        std::vector<std::pair<CodeChunk, float>> rescored_results;
        EmbeddingEngine* tfidf = indexManager->getTfIdfEngine();

        if (use_grag) {
            std::vector<float> q_emb;
            if (engine) q_emb = engine->embed(query);
            rescored_results = GragScorer::rescore(rag_results, q_emb, activeGoal, activeCurrent, activeTraj, retrievalConfig, diagnostics, {}, tfidf, query, memory);
        } else {
            rescored_results = GragScorer::rescore(rag_results, {}, {}, {}, {}, retrievalConfig, diagnostics, {}, tfidf, query, memory);
        }

        if (goalDirected) {
            for (const auto& [chunk, score] : rescored_results) {
                finalMatches.push_back(chunk);
            }
            if (finalMatches.size() > static_cast<size_t>(topK)) {
                finalMatches.resize(topK);
            }
        } else {
            Thoth::ChatRetrieval::applyConversationalBoosts(rescored_results, query, diagnostics);
            const auto selected = Thoth::ChatRetrieval::selectTopKForInjection(
                rescored_results, topK, Thoth::ChatRetrieval::kMinChunkChars, diagnostics);
            for (const auto& [chunk, score] : selected) {
                finalMatches.push_back(chunk);
            }
        }

        diagnostics.plan_id = pId.empty() ? planId : pId;
        diagnostics.step_id = sId.empty() ? stepId : sId;
        diagnostics.goal_present = goalDirected;

        if (eventCallback) {
            ControllerEvent ev;
            ev.type = EventType::RETRIEVAL_DIAGNOSTICS;
            ev.metadata = diagnostics.to_json();
            eventCallback(ev);
        }

        logGragBenchmark(requestId, query, diagnostics);

        if (outDiagnostics) {
            *outDiagnostics = diagnostics;
        }

    } catch (const std::exception& e) {
        std::cerr << "[RAG] Rescoring failed: " << e.what() << "\n";
        for (size_t i = 0; i < std::min(rag_results.size(), static_cast<size_t>(topK)); ++i) {
            finalMatches.push_back(rag_results[i].first);
        }
        if (outDiagnostics) {
            *outDiagnostics = diagnostics;
        }
    }

    return finalMatches;
}

void RAGPipeline::logGragBenchmark(const std::string& requestId, 
                                  const std::string& query,
                                  const GragDiagnostics& diags) {
    nlohmann::json j;
    j["request_id"] = requestId;
    j["query"] = query;
    j["diagnostics"] = diags.to_json();
    j["timestamp_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    FileHandler fh;
    std::string path = fh.getAgentWorkspacePath("grag_benchmark.jsonl");
    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << j.dump() << "\n";
    }
}

std::string RAGPipeline::query(const std::string& queryStr) {
    auto results = retrieveRelevant(queryStr);
    std::string out = "Top Results:\n";
    for (const auto& c : results) {
        out += "- " + c.fileName + "\n";
    }
    return out;
}

void RAGPipeline::clear() {
    std::lock_guard<std::shared_mutex> lock(chunksMutex);
    goalEmbedding.clear();
    currentEmbedding.clear();
    trajectoryEmbedding.clear();
}

std::string RAGPipeline::limitText(const std::string& text, size_t maxChars) {
    if (text.length() <= maxChars) return text;
    size_t cutoff = text.find_last_of(" \n\t", maxChars);
    if (cutoff == std::string::npos || cutoff < maxChars / 2) cutoff = maxChars;
    return text.substr(0, cutoff) + "...";
}
