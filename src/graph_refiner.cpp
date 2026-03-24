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

void GraphRefiner::refineFromTrajectory(const std::vector<Memory::Edge>& trajectory_edges, float success_score) {
    if (!memory_ || trajectory_edges.empty()) return;

    bool is_success = (success_score >= 0.8f);
    
    int edges_updated = 0;
    int edges_created = 0;
    float total_weight_delta = 0.0f;
    float min_weight = 1.0f;
    float max_weight = 0.0f;
    
    for (auto edge : trajectory_edges) {
        float old_w = edge.weight;
        bool was_new = (old_w == 0.1f && edge.success_count == 0 && edge.failure_count == 0);
        
        if (is_success) {
            // Reward: Logistic update (Approaches 1.0 but never explodes)
            edge.weight = old_w + learning_rate * (1.0f - old_w);
            edge.success_count++;
        } else {
            // Penalty: Decrease weight
            edge.weight = old_w - learning_rate * old_w;
            edge.failure_count++;
        }

        // Clamp to prevent complete disappearance or negative values
        if (edge.weight < 0.01f) edge.weight = 0.01f;
        if (edge.weight > 1.0f) edge.weight = 1.0f;

        float weight_delta = edge.weight - old_w;
        total_weight_delta += weight_delta;
        if (edge.weight < min_weight) min_weight = edge.weight;
        if (edge.weight > max_weight) max_weight = edge.weight;
        
        memory_->addEdge(edge);
        
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
