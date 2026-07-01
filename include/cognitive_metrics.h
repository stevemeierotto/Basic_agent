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

    /** E1: optional benchmark run attribution (run_id + env_hash). */
    std::string run_id;
    std::string env_hash;

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
    int max_reflections = 0;
    bool plan_reused = false;

    std::int64_t total_tokens = 0;
    std::int64_t prompt_tokens = 0;
    std::int64_t completion_tokens = 0;
    std::int64_t planning_tokens = 0;
    std::int64_t synthesis_tokens = 0;
    std::string reflection_skip_reason;

    /** C7: last LLM synthesis prompt size and whether retrieved context was capped. */
    int synthesis_prompt_chars = 0;
    bool synthesis_context_truncated = false;
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
