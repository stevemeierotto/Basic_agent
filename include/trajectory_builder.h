/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — TrajectoryBuilder Implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include "memory_repository.h"
#include "embedding_engine.h"
#include "json.hpp"

namespace Thoth {

/**
 * @brief Builds a semantic trajectory embedding from recent execution history.
 */
class TrajectoryBuilder {
public:
    TrajectoryBuilder(std::shared_ptr<MemoryRepository> repo, EmbeddingEngine* engine);

    /**
     * @brief Builds and embeds a trajectory summary for the given goal.
     * @param goal_id The unique ID of the current plan/goal.
     * @param n Number of recent steps to consider (default 7).
     * @return A vector embedding (T). Returns a zero vector if fewer than 3 steps exist.
     */
    std::vector<float> buildTrajectory(const std::string& goal_id, int n = 7);

    /**
     * @brief Constructs the JSON summary that will be embedded.
     */
    nlohmann::json buildSummaryJson(const std::vector<MemoryRepository::EpisodeStepRecord>& steps);

private:
    std::shared_ptr<MemoryRepository> repo_;
    EmbeddingEngine* engine_;
};

} // namespace Thoth
