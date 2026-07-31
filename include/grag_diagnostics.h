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
    std::string symbol;
    std::string code_text;
    std::string source_node_id; // For graph activation trace
    float query_sim = 0.0f;
    float goal_sim = 0.0f;
    float trajectory_sim = 0.0f;
    float keyword_score = 0.0f;
    float graph_score = 0.0f;
    float final_score = 0.0f;

    nlohmann::json to_json() const {
        return {
            {"file_name", file_name},
            {"symbol", symbol},
            {"code_text", code_text},
            {"source_node_id", source_node_id},
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
    /** CSG-A: "executive" | "session" | "none" */
    std::string goal_source = "none";
    
    // Adaptive Graph Memory Metrics (Step 4.6)
    int graph_total_nodes = 0;
    int graph_total_edges = 0;
    float graph_avg_weight = 0.0f;
    int graph_activations = 0;  // Number of chunks that received graph_score > 0
    float graph_max_contribution = 0.0f;  // Highest graph_score contribution in this retrieval

    // For benchmark logging
    nlohmann::json retrieval_trace = nlohmann::json::object();

    nlohmann::json to_json() const {
        nlohmann::json j_breakdowns = nlohmann::json::array();
        for (const auto& b : breakdowns) j_breakdowns.push_back(b.to_json());

        nlohmann::json j = {
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
            {"goal_present", goal_present},
            {"goal_source", goal_source},
            {"graph_total_nodes", graph_total_nodes},
            {"graph_total_edges", graph_total_edges},
            {"graph_avg_weight", graph_avg_weight},
            {"graph_activations", graph_activations},
            {"graph_max_contribution", graph_max_contribution}
        };
        if (!retrieval_trace.is_null() && !retrieval_trace.empty()) {
            j["retrieval_trace"] = retrieval_trace;
        }
        return j;
    }
};
