/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — GraphRefiner Implementation (Adaptive Graph Learning)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include "memory.h"
#include <vector>
#include <string>

namespace Thoth {

/**
 * @brief Handles reinforcement learning logic for the knowledge graph.
 */
class GraphRefiner {
public:
    explicit GraphRefiner(std::shared_ptr<Memory> memory);

    /**
     * @brief Reinforces or penalizes edges based on the success of a completed plan.
     */
    void refineFromTrajectory(const std::vector<Memory::Edge>& trajectory_edges, float success_score);
    // Same arithmetic as refineFromTrajectory, without writing. The controller
    // commits the returned edges in one transaction with the consumed marker.
    std::vector<Memory::Edge> computeReinforcedEdges(
        const std::vector<Memory::Edge>& trajectory_edges,
        float success_score) const;

private:
    std::shared_ptr<Memory> memory_;
    const float learning_rate = 0.2f;
};

} // namespace Thoth
