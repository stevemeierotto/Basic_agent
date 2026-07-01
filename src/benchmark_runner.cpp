/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — BenchmarkRunner implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/benchmark_runner.h"
#include "../include/memory.h"
#include "../include/memory_repository.h"
#include "../include/trajectory_builder.h"
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

namespace Thoth {

static float calculate_dcg(const std::vector<bool>& hits) {
    float dcg = 0.0f;
    for (size_t i = 0; i < hits.size(); ++i) {
        if (hits[i]) {
            dcg += 1.0f / std::log2(static_cast<float>(i) + 2.0f);
        }
    }
    return dcg;
}

static float calculate_idcg(int k) {
    float idcg = 0.0f;
    for (int i = 0; i < k; ++i) {
        idcg += 1.0f / std::log2(static_cast<float>(i) + 2.0f);
    }
    return idcg;
}

BenchmarkRunner::BenchmarkRunner(RAGPipeline& rag) : rag_(rag) {}

BenchmarkResult BenchmarkRunner::run(const BenchmarkConfig& config, const std::vector<BenchmarkCase>& cases) {
    BenchmarkResult result;
    
    // Save current RAG state
    auto old_wq = rag_.retrievalConfig.wq;
    auto old_wd = rag_.retrievalConfig.wd;
    auto old_wt = rag_.retrievalConfig.wt;
    auto old_kw = rag_.retrievalConfig.keyword_weight;
    auto old_mode = rag_.retrievalConfig.mode;
    auto old_goal = rag_.goalEmbedding;
    auto old_curr = rag_.currentEmbedding;
    auto old_traj = rag_.trajectoryEmbedding;

    // Set benchmark weights
    rag_.retrievalConfig.wq = config.wq;
    rag_.retrievalConfig.wd = config.wd;
    rag_.retrievalConfig.wt = config.wt;
    rag_.retrievalConfig.keyword_weight = config.keyword_weight;
    rag_.retrievalConfig.mode = RetrievalMode::GRAG; // Force rescoring path

    float total_precision = 0.0f;
    float total_rr = 0.0f;
    float total_ndcg = 0.0f;

    float idcg = calculate_idcg(config.top_k);

    for (const auto& bc : cases) {
        // Setup embeddings for this case
        if (rag_.engine) {
            if (!bc.goal.empty()) {
                auto g = rag_.engine->embed(bc.goal);
                rag_.setGoalEmbedding(g);
                std::vector<float> zero_curr(g.size(), 0.0f);
                rag_.setCurrentEmbedding(zero_curr);
            } else {
                rag_.setGoalEmbedding({});
                rag_.setCurrentEmbedding({});
            }

            if (config.force_empty_trajectory) {
                rag_.setTrajectoryEmbedding(std::vector<float>(rag_.engine->getDimension(), 0.0f));
            } else if (!bc.trajectory.empty()) {
                // Phase 5.5: Instead of direct embedding of raw string, 
                // use the new TrajectoryBuilder logic to match live behavior.
                // For benchmark cases, we'll wrap the 'trajectory' string in a mock EpisodeStep.
                
                std::string goal_id = "bench-" + bc.case_id;
                
                // Clear and inject mock history
                auto repo = rag_.memory ? rag_.memory->getRepo() : nullptr;
                if (repo) {
                    MemoryRepository::EpisodeStepRecord mockStep;
                    mockStep.episode_id = goal_id;
                    mockStep.goal_id = goal_id;
                    mockStep.step_index = 0;
                    mockStep.state_summary = "Prior Context";
                    mockStep.action_taken = "retrieval";
                    mockStep.result_status = "SUCCESS";
                    mockStep.timestamp_ms = 1000;
                    
                    // We need at least 3 steps for TrajectoryBuilder to activate
                    mockStep.step_index = 0;
                    mockStep.action_taken = "Starting goal research.";
                    repo->storeEpisodeStep(mockStep);
                    
                    mockStep.step_index = 1;
                    mockStep.action_taken = bc.trajectory; // Unique disambiguation context
                    repo->storeEpisodeStep(mockStep);
                    
                    mockStep.step_index = 2;
                    mockStep.action_taken = "Awaiting next retrieval.";
                    repo->storeEpisodeStep(mockStep);
                    
                    TrajectoryBuilder builder(repo, rag_.engine.get());
                    rag_.setTrajectoryEmbedding(builder.buildTrajectory(goal_id));
                } else {
                    rag_.setTrajectoryEmbedding(rag_.engine->embed(bc.trajectory));
                }
            } else {
                rag_.setTrajectoryEmbedding({});
            }
        }

        auto chunks = rag_.retrieveRelevant(bc.query, {}, config.top_k);

        CaseResult cr;
        cr.case_id = bc.case_id;
        cr.case_type = bc.case_type;
        
        int hits_count = 0;
        float first_hit_rank = 0.0f;
        std::vector<bool> hits_mask;

        for (size_t i = 0; i < chunks.size(); ++i) {
            const auto& chunk = chunks[i];
            cr.top_k_chunks.push_back(chunk.fileName);

            std::string baseName = fs::path(chunk.fileName).filename().string();

            bool is_hit = false;
            for (const auto& expected : bc.expected_source) {
                if (baseName.find(expected) != std::string::npos) {
                    is_hit = true;
                    break;
                }
            }

            hits_mask.push_back(is_hit);
            if (is_hit) {
                hits_count++;
                cr.hits.push_back(chunk.fileName);
                if (first_hit_rank == 0.0f) {
                    first_hit_rank = 1.0f / (static_cast<float>(i) + 1.0f);
                }
            }
        }

        cr.precision_at_k = static_cast<float>(hits_count) / static_cast<float>(config.top_k);
        cr.reciprocal_rank = first_hit_rank;
        
        float dcg = calculate_dcg(hits_mask);
        cr.ndcg_at_k = (idcg > 0) ? (dcg / idcg) : 0.0f;

        total_precision += cr.precision_at_k;
        total_rr += cr.reciprocal_rank;
        total_ndcg += cr.ndcg_at_k;

        result.cases.push_back(cr);
    }

    if (!cases.empty()) {
        result.mean_precision = total_precision / static_cast<float>(cases.size());
        result.mean_reciprocal_rank = total_rr / static_cast<float>(cases.size());
        result.mean_ndcg = total_ndcg / static_cast<float>(cases.size());
    }

    // Restore state
    rag_.retrievalConfig.wq = old_wq;
    rag_.retrievalConfig.wd = old_wd;
    rag_.retrievalConfig.wt = old_wt;
    rag_.retrievalConfig.keyword_weight = old_kw;
    rag_.retrievalConfig.mode = old_mode;
    rag_.goalEmbedding = old_goal;
    rag_.currentEmbedding = old_curr;
    rag_.trajectoryEmbedding = old_traj;

    return result;
}

ComparisonResult BenchmarkRunner::runComparison(const std::vector<BenchmarkCase>& cases) {
    BenchmarkConfig rag_cfg;
    rag_cfg.wq = 1.0f;
    rag_cfg.wd = 0.0f;
    rag_cfg.wt = 0.0f;
    rag_cfg.keyword_weight = 0.3f;

    BenchmarkConfig grag_cfg;
    grag_cfg.wq = 0.4f;
    grag_cfg.wd = 0.4f;
    grag_cfg.wt = 0.2f;
    grag_cfg.keyword_weight = 0.3f;

    auto rag_res = run(rag_cfg, cases);
    auto grag_res = run(grag_cfg, cases);

    ComparisonResult result;
    result.rag_mean_precision = rag_res.mean_precision;
    result.grag_mean_precision = grag_res.mean_precision;
    result.rag_mean_reciprocal_rank = rag_res.mean_reciprocal_rank;
    result.grag_mean_reciprocal_rank = grag_res.mean_reciprocal_rank;
    result.rag_mean_ndcg = rag_res.mean_ndcg;
    result.grag_mean_ndcg = grag_res.mean_ndcg;

    std::map<std::string, std::pair<float, int>> rag_prec_sums, grag_precision_sums;
    std::map<std::string, std::pair<float, int>> rag_ndcg_sums, grag_ndcg_sums;

    for (size_t i = 0; i < cases.size(); ++i) {
        CaseDelta cd;
        cd.case_id = cases[i].case_id;
        cd.case_type = cases[i].case_type;
        
        cd.rag_precision = rag_res.cases[i].precision_at_k;
        cd.grag_precision = grag_res.cases[i].precision_at_k;
        cd.precision_delta = cd.grag_precision - cd.rag_precision;
        
        cd.rag_reciprocal_rank = rag_res.cases[i].reciprocal_rank;
        cd.grag_reciprocal_rank = grag_res.cases[i].reciprocal_rank;

        cd.rag_ndcg = rag_res.cases[i].ndcg_at_k;
        cd.grag_ndcg = grag_res.cases[i].ndcg_at_k;
        cd.ndcg_delta = cd.grag_ndcg - cd.rag_ndcg;

        float r_rank = (cd.rag_reciprocal_rank > 0) ? (1.0f / cd.rag_reciprocal_rank) : 6.0f;
        float g_rank = (cd.grag_reciprocal_rank > 0) ? (1.0f / grag_res.cases[i].reciprocal_rank) : 6.0f;
        cd.directional_lift = r_rank - g_rank;
        
        cd.rag_top_chunks = rag_res.cases[i].top_k_chunks;
        cd.grag_top_chunks = grag_res.cases[i].top_k_chunks;
        
        result.deltas.push_back(cd);

        rag_prec_sums[cd.case_type].first += cd.rag_precision;
        rag_prec_sums[cd.case_type].second++;
        grag_precision_sums[cd.case_type].first += cd.grag_precision;
        grag_precision_sums[cd.case_type].second++;

        rag_ndcg_sums[cd.case_type].first += cd.rag_ndcg;
        rag_ndcg_sums[cd.case_type].second++;
        grag_ndcg_sums[cd.case_type].first += cd.grag_ndcg;
        grag_ndcg_sums[cd.case_type].second++;
    }

    for (auto const& [type, val] : rag_prec_sums) {
        result.rag_precision_by_type[type] = val.first / static_cast<float>(val.second);
    }
    for (auto const& [type, val] : grag_precision_sums) {
        result.grag_precision_by_type[type] = val.first / static_cast<float>(val.second);
    }
    for (auto const& [type, val] : rag_ndcg_sums) {
        result.rag_ndcg_by_type[type] = val.first / static_cast<float>(val.second);
    }
    for (auto const& [type, val] : grag_ndcg_sums) {
        result.grag_ndcg_by_type[type] = val.first / static_cast<float>(val.second);
    }

    return result;
}

} // namespace Thoth
