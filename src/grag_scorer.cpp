/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — GRAG Scorer Implementation (Refined for Adaptive Graph Learning)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/grag_scorer.h"
#include "../include/memory.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>

float GragScorer::cosine_similarity(const std::vector<float>& a,
                                   const std::vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0f;
    }

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a <= 1e-9f || norm_b <= 1e-9f) {
        return 0.0f;
    }

    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

std::vector<float> GragScorer::compute_direction(
    const std::vector<float>& goal_embedding,
    const std::vector<float>& current_embedding) {
    
    if (goal_embedding.empty()) {
        return {};
    }
    
    std::vector<float> direction(goal_embedding.size());
    if (current_embedding.empty() || current_embedding.size() != goal_embedding.size()) {
        // Treat as zero vector (start of plan)
        return goal_embedding;
    }

    for (size_t i = 0; i < goal_embedding.size(); ++i) {
        direction[i] = goal_embedding[i] - current_embedding[i];
    }

    return direction;
}

float GragScorer::compute_alpha(const std::vector<float>& direction,
                                float direction_threshold) {
    if (direction.empty()) return 0.0f;

    float magnitude_sq = 0.0f;
    for (float v : direction) magnitude_sq += v * v;
    float magnitude = std::sqrt(magnitude_sq);
    
    if (direction_threshold <= 0.0f) return 1.0f; 
    
    float raw = magnitude / direction_threshold;
    return std::clamp(raw, 0.0f, 1.0f);
}

std::vector<std::pair<CodeChunk, float>> GragScorer::rescore(
    const std::vector<std::pair<CodeChunk, float>>& rag_results,
    const std::vector<float>& /*query_embedding*/,
    const std::vector<float>& goal_embedding,
    const std::vector<float>& current_embedding,
    const std::vector<float>& trajectory_embedding,
    const RetrievalConfig& config,
    GragDiagnostics& diagnostics_out,
    const std::unordered_map<std::string, float>& /*deprecated_graph_scores*/,
    EmbeddingEngine* tfidf_engine,
    const std::string& query_text,
    const Memory* memory) {

    diagnostics_out.chunks_retrieved = static_cast<int>(rag_results.size());
    diagnostics_out.breakdowns.clear();
    
    // Initialize graph metrics
    diagnostics_out.graph_activations = 0;
    diagnostics_out.graph_max_contribution = 0.0f;
    
    // Collect graph statistics if memory is available
    if (memory) {
        auto stats = memory->getGraphStatistics();
        diagnostics_out.graph_total_nodes = stats.total_nodes;
        diagnostics_out.graph_total_edges = stats.total_edges;
        diagnostics_out.graph_avg_weight = stats.avg_edge_weight;
    }

    // Calculate raw direction D = G - C
    std::vector<float> D = compute_direction(goal_embedding, current_embedding);
    float magnitude = 0.0f;
    if (!D.empty()) {
        float magnitude_sq = 0.0f;
        for (float v : D) magnitude_sq += v * v;
        magnitude = std::sqrt(magnitude_sq);
    }
    diagnostics_out.direction_magnitude = magnitude;
    
    float alpha = compute_alpha(D, config.direction_threshold);
    diagnostics_out.alpha = alpha;

    if (!config.grag_directional || D.empty()) {
        diagnostics_out.scoring_type = "rag_hybrid";
        alpha = 0.0f;
    } else {
        if (alpha < 0.01f) diagnostics_out.scoring_type = "rag_hybrid";
        else if (alpha > 0.99f) diagnostics_out.scoring_type = "grag_hybrid";
        else diagnostics_out.scoring_type = "grag_blended_hybrid";
    }

    // Dynamic TF-IDF query embedding if needed
    std::vector<float> tfidf_query;
    if (tfidf_engine && !query_text.empty()) {
        tfidf_query = tfidf_engine->embed(query_text);
    }

    std::unordered_map<std::string, std::vector<float>> chunk_tfidf_cache;
    if (tfidf_engine && !tfidf_query.empty() && !rag_results.empty()) {
        std::vector<std::string> chunk_codes;
        chunk_codes.reserve(rag_results.size());
        for (const auto& [chunk, _] : rag_results) {
            chunk_codes.push_back(chunk.code);
        }
        const auto chunk_embeddings = tfidf_engine->embedBatch(chunk_codes);
        for (size_t i = 0; i < rag_results.size() && i < chunk_embeddings.size(); ++i) {
            chunk_tfidf_cache.emplace(rag_results[i].first.code, chunk_embeddings[i]);
        }
    }

    // Phase 5.6: Identify high-confidence query hits for graph activation
    struct HighConfHit {
        std::string hash;
        float query_score;
    };
    std::vector<HighConfHit> high_conf_hits;
    if (memory) {
        for (const auto& [chunk, original_score] : rag_results) {
            if (original_score >= 0.7f) {
                high_conf_hits.push_back({Memory::calculateContentHash(chunk.code), original_score});
            }
        }
    }

    std::vector<std::pair<CodeChunk, float>> results;
    results.reserve(rag_results.size());

    for (const auto& [chunk, original_score] : rag_results) {
        ScoreBreakdown sb;
        sb.file_name = chunk.fileName;
        sb.symbol = chunk.symbolName;
        sb.code_text = chunk.code;
        sb.query_sim = original_score; // Semantic(Q, chunk) from initial retrieval
        sb.goal_sim = D.empty() ? 0.0f : cosine_similarity(D, chunk.embedding);
        sb.trajectory_sim = trajectory_embedding.empty() ? 0.0f : cosine_similarity(trajectory_embedding, chunk.embedding);

        // Core Semantic Blend (using config weights Phase 5.1)
        // If alpha is 0.0 (Standard RAG), we must not penalize the semantic query signal.
        float semantic_weight = (1.0f - alpha) * config.wq;
        if (alpha < 0.001f) {
             semantic_weight = 1.0f; // Pure RAG uses full semantic signal
        }
        
        float vector_score = semantic_weight * sb.query_sim
                           + alpha           * config.wd * sb.goal_sim
                           + config.wt       * sb.trajectory_sim;

        // Dynamic Keyword Signal
        if (!tfidf_query.empty()) {
            const auto cached = chunk_tfidf_cache.find(chunk.code);
            if (cached != chunk_tfidf_cache.end()) {
                sb.keyword_score = cosine_similarity(tfidf_query, cached->second);
            } else if (tfidf_engine) {
                sb.keyword_score =
                    cosine_similarity(tfidf_query, tfidf_engine->embed(chunk.code));
            }
        } else {
            sb.keyword_score = chunk.keyword_score;
        }

        // Hybrid Boost
        float hybrid_score = vector_score + (config.keyword_weight * sb.keyword_score);

        // Phase 5.6: Advanced Graph Scoring (1-Hop Neighbor Activation)
        sb.graph_score = 0.0f;
        if (memory && !high_conf_hits.empty()) {
            std::string targetHash = Memory::calculateContentHash(chunk.code);
            for (const auto& hit : high_conf_hits) {
                if (hit.hash == targetHash) continue;
                
                auto edges = memory->getEdgesFrom(hit.hash);
                for (const auto& edge : edges) {
                    if (edge.to_id == targetHash) {
                        float contribution = edge.weight * hit.query_score;
                        if (contribution > sb.graph_score) {
                            sb.graph_score = contribution;
                            sb.source_node_id = hit.hash;
                        }
                    }
                }
            }
            
            // Track graph activations for metrics
            if (sb.graph_score > 0.0f) {
                diagnostics_out.graph_activations++;
                if (sb.graph_score > diagnostics_out.graph_max_contribution) {
                    diagnostics_out.graph_max_contribution = sb.graph_score;
                }
            }
        }

        sb.final_score = hybrid_score + (config.graph_weight * sb.graph_score);

        results.push_back({chunk, sb.final_score});
        diagnostics_out.breakdowns.push_back(sb);
    }

    // Sort results by final score descending
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Also sort breakdowns to match result order
    std::sort(diagnostics_out.breakdowns.begin(), diagnostics_out.breakdowns.end(), [](const auto& a, const auto& b) {
        return a.final_score > b.final_score;
    });

    diagnostics_out.chunks_reranked = static_cast<int>(results.size());
    diagnostics_out.final_scores.clear();
    for (const auto& res : results) {
        diagnostics_out.final_scores.push_back(res.second);
    }

    return results;
}
