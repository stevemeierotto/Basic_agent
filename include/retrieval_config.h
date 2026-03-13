/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — GRAG Phase 1 Implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

enum class RetrievalMode {
    RAG,   // Standard cosine similarity only
    GRAG,  // Goal-Relative Adaptive Graph (Full)
    AUTO   // Dynamic switch based on goal presence
};

enum class GragRoutingMode {
    PLAN_AWARE,
    GOAL_ONLY,
    CONVERSATIONAL
};

/**
 * @struct RetrievalConfig
 * @brief Parameters for the GRAG scoring engine.
 */
struct RetrievalConfig {
    RetrievalMode mode = RetrievalMode::AUTO;

    // The distance G - C must exceed this to trigger directional navigation.
    // The goal is to see alpha values in the 0.3–0.7 range during active
    // mid-plan execution. Near 0 at plan start and near 1 near completion
    // is expected and correct.
    float direction_threshold = 0.3f;

    // Weights for rescoring (Phase 13 control)
    float wq = 0.4f; // Query weight
    float wd = 0.4f; // Directional (goal) weight
    float wt = 0.2f; // Trajectory weight

    // Phase 3.1: Hybrid Reranking
    float keyword_weight = 0.3f; // Weight for TF-IDF keyword boost

    int top_k = 5;
    int plan_history_top_k = 3;
    bool grag_directional = true; 
    float graph_weight = 0.3f; // Phase 8 weighting

    // GRAG-FUTURE: per-index top_k values for multi-index routing
    // GRAG-FUTURE: index selection hints (e.g. force codebase_index for code steps)
};
