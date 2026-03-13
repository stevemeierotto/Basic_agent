#include "../include/rag.h"
#include "../include/decision_trace.h"
#include "../include/file_handler.h"
#include "../include/chunkers/chunker.h"
#include "../include/grag_scorer.h"
#include "../include/logger.h"
#include "../include/memory.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>

using json = nlohmann::json;

RAGPipeline::RAGPipeline(std::unique_ptr<EmbeddingEngine> eng, IndexManager* idx, Config* cfg, Memory* mem)
    : engine(std::move(eng)), indexManager(idx), config(cfg), memory(mem) {
}

std::string RAGPipeline::query(const std::string& queryStr) {
    auto results = retrieveRelevant(queryStr, {}, 3);
    if (results.empty()) return "No relevant information found.";

    std::ostringstream oss;
    oss << "Relevant Context:\n\n";
    for (const auto& chunk : results) {
        oss << "File: " << chunk.fileName << "\n";
        oss << "Content:\n" << chunk.code << "\n";
        oss << "---\n";
    }
    return oss.str();
}

std::vector<CodeChunk> RAGPipeline::retrieveRelevant(const std::string& query, 
                                                    const std::vector<int>& /*errorLines*/, 
                                                    int topK,
                                                    const std::string& requestId,
                                                    const std::string& pId,
                                                    const std::string& sId) {
    std::unique_lock<std::shared_mutex> lock(chunksMutex);
    
    std::vector<CodeChunk> finalMatches;
    if (!indexManager) return finalMatches;

    // Sync weights from global config if available (Phase 5.1)
    // Only do this if we are in AUTO mode, otherwise we might overwrite benchmark weights
    if (config && retrievalConfig.mode == RetrievalMode::AUTO) {
        retrievalConfig.wq = config->wq;
        retrievalConfig.wd = config->wd;
        retrievalConfig.wt = config->wt;
        retrievalConfig.keyword_weight = config->keyword_weight;
    }

    GragDiagnostics diagnostics;
    diagnostics.plan_id = pId.empty() ? planId : pId;
    diagnostics.step_id = sId.empty() ? stepId : sId;
    diagnostics.routing_mode = "AUTO";

    try {
        GragRoutingMode routingMode = GragRoutingMode::PLAN_AWARE;
        
        std::vector<std::pair<std::string, float>> rawResults;
        bool routingComplete = false;

        while (!routingComplete) {
            if (routingMode == GragRoutingMode::PLAN_AWARE) {
                diagnostics.indexes_used = {"CONVERSATIONS", "KNOWLEDGE", "PLAN_HISTORY", "CODEBASE"};
                // Buffer increased to 40 for better rescore recall (Phase 7.1)
                rawResults = indexManager->retrieveChunks(query, 40);

                if (rawResults.empty()) {
                    routingMode = GragRoutingMode::GOAL_ONLY;
                    continue;
                }
                routingComplete = true;

            } else if (routingMode == GragRoutingMode::GOAL_ONLY) {
                diagnostics.indexes_used = {"CONVERSATIONS", "KNOWLEDGE", "PLAN_HISTORY", "CODEBASE"};
                rawResults = indexManager->retrieveChunks(query, 20);
                if (rawResults.empty()) {
                    routingMode = GragRoutingMode::CONVERSATIONAL;
                    continue;
                }
                routingComplete = true;

            } else if (routingMode == GragRoutingMode::CONVERSATIONAL) {
                diagnostics.indexes_used = {"CONVERSATIONS"};
                rawResults = indexManager->retrieveChunks(query, topK);
                routingComplete = true;
            }
        }

        std::vector<std::pair<CodeChunk, float>> rag_results;
        for (const auto& [text, score] : rawResults) {
            const CodeChunk* chunk = indexManager->getChunkByCode(text);
            if (chunk != nullptr) {
                rag_results.push_back({*chunk, score});
            }
        }

        bool use_grag = false;
        if (retrievalConfig.mode == RetrievalMode::AUTO) {
            if (routingMode != GragRoutingMode::CONVERSATIONAL) {
                use_grag = !goalEmbedding.empty() && !currentEmbedding.empty();
            }
        } else if (retrievalConfig.mode == RetrievalMode::GRAG) {
            use_grag = true;
        }

        std::vector<std::pair<CodeChunk, float>> rescored_results;
        EmbeddingEngine* tfidf = indexManager->getTfIdfEngine();

        if (use_grag) {
            std::vector<float> query_embedding;
            if (engine) {
                query_embedding = engine->embed(query);
            }

            std::unordered_map<std::string, float> graph_scores;
            if (memory) {
                auto edges = memory->getEdgesFrom(diagnostics.plan_id);
                for (const auto& e : edges) {
                    graph_scores[e.to_id] = e.weight;
                }
            }

            rescored_results = GragScorer::rescore(rag_results, query_embedding, goalEmbedding, currentEmbedding, trajectoryEmbedding, retrievalConfig, diagnostics, graph_scores, tfidf, query);
        } else {
            std::unordered_map<std::string, float> graph_scores;
            if (memory) {
                auto edges = memory->getEdgesFrom(diagnostics.plan_id);
                for (const auto& e : edges) {
                    graph_scores[e.to_id] = e.weight;
                }
            }
            
            rescored_results = GragScorer::rescore(rag_results, {}, {}, {}, {}, retrievalConfig, diagnostics, graph_scores, tfidf, query);
        }

        for (const auto& [chunk, score] : rescored_results) {
            finalMatches.push_back(chunk);
        }

        if (finalMatches.size() > static_cast<size_t>(topK)) {
            finalMatches.resize(topK);
        }

        if (eventCallback) {
            ControllerEvent ev;
            ev.type = EventType::RETRIEVAL_DIAGNOSTICS;
            ev.plan_id = diagnostics.plan_id;
            ev.step_id = diagnostics.step_id;
            ev.metadata = diagnostics.to_json();
            eventCallback(ev);
        }

        logGragBenchmark(requestId, query, diagnostics);

    } catch (const std::exception& e) {
        std::cerr << "[RAGPipeline] Exception in retrieveRelevant: " << e.what() << "\n";
    }

    return finalMatches;
}

void RAGPipeline::clear() {
    std::unique_lock<std::shared_mutex> lock(chunksMutex);
    if (indexManager) indexManager->clear();
}

void RAGPipeline::logGragBenchmark(const std::string& requestId, 
                                   const std::string& query,
                                   const GragDiagnostics& diagnostics) {
    FileHandler fh;
    std::string path = fh.getAgentWorkspacePath("grag_benchmark.jsonl");
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) return;

    json entry;
    entry["request_id"] = requestId;
    entry["timestamp_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    entry["query"] = query;
    entry["diagnostics"] = diagnostics.to_json();
    
    out << entry.dump() << "\n";
}

std::string RAGPipeline::limitText(const std::string& text, size_t maxChars) {
    if (text.size() <= maxChars) return text;
    
    size_t cutoff = text.find_last_of(" \n\t", maxChars);
    if (cutoff == std::string::npos || cutoff < maxChars / 2) {
        cutoff = maxChars;
    }
    
    return text.substr(0, cutoff) + "...";
}
