/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — GRAG Phase 1 Implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include "json.hpp"
#include <string>
#include <vector>

/**
 * @struct ScoreBreakdown
 * @brief Detailed signals used to compute a single chunk's final score.
 */
struct ScoreBreakdown {
    std::string file_name;
    float query_sim = 0.0f;
    float goal_sim = 0.0f;
    float trajectory_sim = 0.0f;
    float keyword_score = 0.0f;
    float graph_score = 0.0f;
    float final_score = 0.0f;

    nlohmann::json to_json() const {
        return {
            {"file_name", file_name},
            {"query_sim", query_sim},
            {"goal_sim", goal_sim},
            {"trajectory_sim", trajectory_sim},
            {"keyword_score", keyword_score},
            {"graph_score", graph_score},
            {"final_score", final_score}
        };
    }
};

struct GragDiagnostics {
    std::string scoring_type;      // "rag", "grag", or "grag_blended"
    std::string routing_mode;      // "PLAN_AWARE", "GOAL_ONLY", "CONVERSATIONAL"
    std::vector<std::string> indexes_used;
    float alpha = 0.0f;            // Blend weight actually used (0 = pure RAG)
    float direction_magnitude = 0.0f; // ||G - C||, useful for detecting collapse
    int chunks_retrieved = 0;
    int chunks_reranked = 0;       
    std::vector<float> final_scores; // One per returned chunk, in order
    
    // Phase 6.1: Detailed breakdowns
    std::vector<ScoreBreakdown> breakdowns;

    std::string plan_id;
    std::string step_id;
    bool goal_present = false;

    // For benchmark logging
    nlohmann::json to_json() const {
        nlohmann::json j_breakdowns = nlohmann::json::array();
        for (const auto& b : breakdowns) j_breakdowns.push_back(b.to_json());

        return {
            {"scoring_type", scoring_type},
            {"routing_mode", routing_mode},
            {"indexes_used", indexes_used},
            {"alpha", alpha},
            {"direction_magnitude", direction_magnitude},
            {"chunks_retrieved", chunks_retrieved},
            {"chunks_reranked", chunks_reranked},
            {"final_scores", final_scores},
            {"breakdowns", j_breakdowns},
            {"plan_id", plan_id},
            {"step_id", step_id},
            {"goal_present", goal_present}
        };
    }
};
