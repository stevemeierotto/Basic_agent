/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — TrajectoryBuilder Implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/trajectory_builder.h"
#include "../include/plan_reuse_config.h"
#include <iostream>
#include <algorithm>

namespace Thoth {

TrajectoryBuilder::TrajectoryBuilder(std::shared_ptr<MemoryRepository> repo, EmbeddingEngine* engine)
    : repo_(repo), engine_(engine) {}

std::vector<float> TrajectoryBuilder::buildTrajectory(const std::string& goal_id, int n) {
    if (!repo_ || !engine_) return {};

    auto steps = repo_->getRecentEpisodeSteps(goal_id, n);
    
    // Requirement: If fewer than 3 episode steps exist, return a zero vector
    if (steps.size() < static_cast<std::size_t>(TrajectoryReuse::kMinEpisodeStepsForEmbedding)) {
        return std::vector<float>(engine_->getDimension(), 0.0f);
    }

    nlohmann::json summary = buildSummaryJson(steps);
    auto vec = engine_->embed(summary.dump());
    if (vec.empty()) {
        return std::vector<float>(engine_->getDimension(), 0.0f);
    }
    return vec;
}

nlohmann::json TrajectoryBuilder::buildSummaryJson(const std::vector<MemoryRepository::EpisodeStepRecord>& steps) {
    nlohmann::json summary;
    summary["recent_actions"] = nlohmann::json::array();
    summary["failures"] = nlohmann::json::array();
    
    // We can't really generate a "summary" text without an LLM here easily,
    // so we'll use a structured representation that the embedding model can understand.
    std::string progress_desc = "Goal progress after " + std::to_string(steps.size()) + " steps.";
    summary["progress_summary"] = progress_desc;

    for (const auto& step : steps) {
        summary["recent_actions"].push_back(step.action_taken);
        if (step.result_status != "SUCCESS") {
            summary["failures"].push_back(step.state_summary + ": " + step.result_status);
        }
    }

    return summary;
}

} // namespace Thoth
