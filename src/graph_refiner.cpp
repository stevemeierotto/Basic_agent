/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — GraphRefiner Implementation (Adaptive Graph Learning)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/graph_refiner.h"
#include <iostream>
#include <iomanip>

namespace Thoth {

GraphRefiner::GraphRefiner(std::shared_ptr<Memory> memory) : memory_(memory) {}

std::vector<Memory::Edge> GraphRefiner::computeReinforcedEdges(
    const std::vector<Memory::Edge>& trajectory_edges,
    float success_score) const {
    std::vector<Memory::Edge> updated;
    updated.reserve(trajectory_edges.size());
    const bool is_success = (success_score >= 0.8f);
    for (auto edge : trajectory_edges) {
        const float old_w = edge.weight;
        if (is_success) {
            edge.weight = old_w + learning_rate * (1.0f - old_w);
            edge.success_count++;
        } else {
            edge.weight = old_w - learning_rate * old_w;
            edge.failure_count++;
        }
        if (edge.weight < 0.01f) edge.weight = 0.01f;
        if (edge.weight > 1.0f) edge.weight = 1.0f;
        updated.push_back(std::move(edge));
    }
    return updated;
}

void GraphRefiner::refineFromTrajectory(const std::vector<Memory::Edge>& trajectory_edges, float success_score) {
    if (!memory_ || trajectory_edges.empty()) return;

    const bool is_success = (success_score >= 0.8f);
    const std::vector<Memory::Edge> updated = computeReinforcedEdges(trajectory_edges, success_score);

    int edges_updated = 0;
    int edges_created = 0;
    float total_weight_delta = 0.0f;
    float min_weight = 1.0f;
    float max_weight = 0.0f;

    for (size_t i = 0; i < updated.size(); ++i) {
        const float old_w = trajectory_edges[i].weight;
        const bool was_new = (old_w == 0.1f && trajectory_edges[i].success_count == 0 &&
                              trajectory_edges[i].failure_count == 0);
        const float weight_delta = updated[i].weight - old_w;
        total_weight_delta += weight_delta;
        if (updated[i].weight < min_weight) min_weight = updated[i].weight;
        if (updated[i].weight > max_weight) max_weight = updated[i].weight;
        memory_->addEdge(updated[i]);
        if (was_new) edges_created++;
        else edges_updated++;
    }
    
    float avg_weight_delta = trajectory_edges.empty() ? 0.0f : (total_weight_delta / trajectory_edges.size());
    
    std::cout << "[GraphRefiner] Refined " << trajectory_edges.size() 
              << " edges (Success: " << (is_success ? "YES" : "NO") 
              << ", Created: " << edges_created
              << ", Updated: " << edges_updated
              << ", Avg Δw: " << std::fixed << std::setprecision(4) << avg_weight_delta
              << ", Weight range: [" << min_weight << ", " << max_weight << "])\n";
}

} // namespace Thoth
