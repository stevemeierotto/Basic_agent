/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C6 per-goal cognitive metrics
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_COGNITIVE_METRICS_H
#define THOTH_COGNITIVE_METRICS_H

#include "json.hpp"
#include <cstdint>
#include <string>

namespace Thoth {

struct GoalCognitiveMetricsRecord {
    std::string plan_id;
    std::string session_id;
    std::string goal;
    std::string outcome; // completed | failed | aborted

    std::int64_t goal_started_at_ms = 0;
    std::int64_t goal_finished_at_ms = 0;
    std::int64_t total_wall_clock_ms = 0;

    std::int64_t planning_time_ms = 0;
    std::int64_t retrieval_time_ms = 0;
    std::int64_t llm_synthesis_time_ms = 0;

    int step_count = 0;
    int retrieved_chunk_count = 0;
    float grag_alpha = 0.0f;
    std::string grag_routing_mode;

    float trajectory_score = 0.0f;
    float final_success_score = 0.0f;
    int reflection_count = 0;
    int revisions_count = 0;
    bool plan_reused = false;

    std::int64_t total_tokens = 0; // reserved; 0 until LLMInterface exposes counts
};

class CognitiveMetricsLogger {
public:
    static CognitiveMetricsLogger& instance();

    void logGoalMetrics(const GoalCognitiveMetricsRecord& record) const;
    static nlohmann::json toJson(const GoalCognitiveMetricsRecord& record);

private:
    CognitiveMetricsLogger();
    std::string logFilePath() const;
    void appendJsonLine(const nlohmann::json& event) const;
};

} // namespace Thoth

#endif // THOTH_COGNITIVE_METRICS_H
