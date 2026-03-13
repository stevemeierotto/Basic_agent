/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — GRAG Scorer Implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/grag_scorer.h"
#include <cmath>
#include <algorithm>
#include <cassert>

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
    
    if (goal_embedding.empty() || current_embedding.empty() || goal_embedding.size() != current_embedding.size()) {
        return {};
    }
    
    std::vector<float> direction(goal_embedding.size());
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
    const std::unordered_map<std::string, float>& graph_scores,
    EmbeddingEngine* tfidf_engine,
    const std::string& query_text) {

    diagnostics_out.chunks_retrieved = static_cast<int>(rag_results.size());
    diagnostics_out.breakdowns.clear();

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

    std::vector<std::pair<CodeChunk, float>> results;
    results.reserve(rag_results.size());

    for (const auto& [chunk, original_score] : rag_results) {
        ScoreBreakdown sb;
        sb.file_name = chunk.fileName;
        sb.query_sim = original_score; // Semantic(Q, chunk) from initial retrieval
        sb.goal_sim = D.empty() ? 0.0f : cosine_similarity(D, chunk.embedding);
        sb.trajectory_sim = trajectory_embedding.empty() ? 0.0f : cosine_similarity(trajectory_embedding, chunk.embedding);

        // Core Semantic Blend (using config weights Phase 5.1)
        float vector_score = (1.0f - alpha) * config.wq * sb.query_sim
                           + alpha          * config.wd * sb.goal_sim
                           + config.wt      * sb.trajectory_sim;

        // Dynamic Keyword Signal
        if (!tfidf_query.empty()) {
            auto tfidf_chunk = tfidf_engine->embed(chunk.code);
            sb.keyword_score = cosine_similarity(tfidf_query, tfidf_chunk);
        } else {
            sb.keyword_score = chunk.keyword_score;
        }

        // Hybrid Boost
        float hybrid_score = vector_score + (config.keyword_weight * sb.keyword_score);

        // Graph Weight
        sb.graph_score = 0.0f;
        auto it = graph_scores.find(chunk.code);
        if (it != graph_scores.end()) {
            sb.graph_score = it->second;
        }

        sb.final_score = ((1.0f - config.graph_weight) * hybrid_score)
                       + (config.graph_weight * sb.graph_score);

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
