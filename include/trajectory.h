/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — Cognate Phase 7.1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <string>
#include <vector>
#include "plan.h"
#include "json.hpp"

namespace Thoth {

/**
 * @brief Represents a single step as it was actually executed.
 */
struct RecordedStep {
    std::string step_id;
    std::string description;
    nlohmann::json tool;
    nlohmann::json result;
    std::string error;
    std::string revision_note;
    int64_t timestamp;

    nlohmann::json to_json() const;
    static RecordedStep from_json(const nlohmann::json& j);
};

/**
 * @brief Trajectory — the complete lifecycle of a goal execution.
 * Records the initial intent, the process (steps/results/revisions), 
 * and the final outcome.
 */
struct Trajectory {
    std::string trajectory_id;
    std::string goal;
    Plan plan_initial;
    std::vector<RecordedStep> steps;
    nlohmann::json results;           // Aggregated or final results
    std::vector<Plan> revisions;      // History of plan revisions
    PlanStatus final_status;
    float success_score = 0.0f;
    int64_t created_at = 0;
    std::vector<float> embedding;     // Goal embedding for similarity search

    nlohmann::json to_json() const;
    static Trajectory from_json(const nlohmann::json& j);
};

} // namespace Thoth
