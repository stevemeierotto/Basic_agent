/*
 * Copyright (c) 2026 Steve Meierotto
 * 
 * Thoth — ProblemState Definition (Cognate V2)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <string>
#include <vector>
#include <json.hpp>

namespace Thoth {

/**
 * @brief Represents the internal reasoning state of a scientific inquiry.
 * This structure is used by ScientificExecutionMode to track the evolution
 * of hypotheses and constraints over multiple iterations.
 */
struct ProblemState {
    std::string problem_id;
    std::string goal_id;
    std::string problem_description;
    
    std::vector<std::string> hypotheses;
    std::vector<std::string> constraints;
    std::vector<std::string> unknowns;
    std::vector<std::string> rejected_paths;
    
    int iteration_count = 0;
    float confidence_score = 0.0f;
    std::vector<float> confidence_history;
    
    long long created_at = 0;
    long long updated_at = 0;

    // Serialization
    nlohmann::json to_json() const {
        return {
            {"problem_id", problem_id},
            {"goal_id", goal_id},
            {"problem_description", problem_description},
            {"hypotheses", hypotheses},
            {"constraints", constraints},
            {"unknowns", unknowns},
            {"rejected_paths", rejected_paths},
            {"iteration_count", iteration_count},
            {"confidence_score", confidence_score},
            {"confidence_history", confidence_history},
            {"created_at", created_at},
            {"updated_at", updated_at}
        };
    }

    static ProblemState from_json(const nlohmann::json& j) {
        ProblemState s;
        s.problem_id = j.value("problem_id", "");
        s.goal_id = j.value("goal_id", "");
        s.problem_description = j.value("problem_description", "");
        s.hypotheses = j.value("hypotheses", std::vector<std::string>{});
        s.constraints = j.value("constraints", std::vector<std::string>{});
        s.unknowns = j.value("unknowns", std::vector<std::string>{});
        s.rejected_paths = j.value("rejected_paths", std::vector<std::string>{});
        s.iteration_count = j.value("iteration_count", 0);
        s.confidence_score = j.value("confidence_score", 0.0f);
        s.confidence_history = j.value("confidence_history", std::vector<float>{});
        s.created_at = j.value("created_at", 0LL);
        s.updated_at = j.value("updated_at", 0LL);
        return s;
    }
};

} // namespace Thoth
