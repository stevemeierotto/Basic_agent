/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — GRAG Phase 1 Implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include "chunkers/code_chunk.h"
#include "retrieval_config.h"
#include "grag_diagnostics.h"
#include "embedding_engine.h"
#include <vector>
#include <utility>
#include <unordered_map>

class GragScorer {
public:
    /**
     * @brief Computes cosine similarity between two vectors.
     */
    static float cosine_similarity(const std::vector<float>& a, 
                                   const std::vector<float>& b);

    /**
     * @brief Computes raw direction vector D = G - C.
     */
    static std::vector<float> compute_direction(
        const std::vector<float>& goal_embedding,
        const std::vector<float>& current_embedding);

    /**
     * @brief Calculates adaptive alpha [0, 1] based on |D| / threshold.
     */
    static float compute_alpha(const std::vector<float>& direction, 
                                float direction_threshold);

    /**
     * @brief Blends RAG results with Goal and Trajectory signals.
     * 
     * Formula (Phase 13):
     * score = (1-alpha)*wq*cosine(Q, chunk) + alpha*wd*cosine(D, chunk) + wt*cosine(T, chunk)
     */
    static std::vector<std::pair<CodeChunk, float>> rescore(
        const std::vector<std::pair<CodeChunk, float>>& rag_results,
        const std::vector<float>& query_embedding,
        const std::vector<float>& goal_embedding,
        const std::vector<float>& current_embedding,
        const std::vector<float>& trajectory_embedding,
        const RetrievalConfig& config,
        GragDiagnostics& diagnostics_out,
        const std::unordered_map<std::string, float>& graph_scores = {},
        EmbeddingEngine* tfidf_engine = nullptr,
        const std::string& query_text = "");

    // GRAG-FUTURE: rescore_multi_index() for merged index results
    // GRAG-FUTURE: Adaptive threshold tuning — learning DIRECTION_THRESHOLD from retrieval success metrics
};
