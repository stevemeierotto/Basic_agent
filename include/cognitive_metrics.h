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
#include <vector>

namespace Thoth {

struct GoalCognitiveMetricsRecord {
    std::string plan_id;
    std::string session_id;
    std::string goal;
    std::string outcome; // completed | failed | aborted

    /** E1: optional benchmark run attribution (run_id + env_hash). */
    std::string run_id;
    std::string env_hash;

    /** C6.4 prospective window. Empty unless attributed at goal start. */
    std::string c64_window_id;
    std::string c64_protocol_version;
    std::string c64_metric_schema_version;
    std::string c64_environment_schema_version;
    std::string c64_cohort_fingerprint;

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

    /** Planner provenance joined to this goal. Empty when the plan did not record an origin. */
    std::string plan_origin = "unknown";
    std::string retrieval_query;
    std::vector<std::string> step_ids;
    bool learning_eligible = false;

    /**
     * True when the numeric counters are not an exact goal total.
     * not_checkpointed: legacy plan, counters omitted.
     * checkpoint_gap: values in available_counters are a lower bound. The
     * interval after the last durable checkpoint until the crash is missing.
     */
    bool metrics_partial = false;
    std::string metrics_partial_reason;
    bool emit_exact_counters = true;
    bool emit_available_counters = false;
    bool emit_goal_started = true;
    bool emit_wall_clock = true;
};

class CognitiveMetricsLogger {
public:
    static CognitiveMetricsLogger& instance();

    void logGoalMetrics(const GoalCognitiveMetricsRecord& record) const;

    /**
     * A JSONL line is a delivered metrics record only when it parses as a
     * JSON object whose metrics_id string equals the requested id. Substring
     * matches, malformed lines, and a truncated final line are not deliveries.
     * The file is never rewritten: a truncated tail stays byte-for-byte, and
     * a new record is appended after a separator newline only when the file
     * does not already end in one.
     *
     * Writers serialize on <log path>.lock with an exclusive flock. The same
     * lock covers the delivery scan and the append, and also direct
     * logGoalMetrics appends. It protects two processes that share one
     * database and two databases that target the same JSONL file.
     */
    bool metricsIdDelivered(const std::string& metrics_id) const;
    bool appendStoredRecord(const nlohmann::json& payload) const;
    bool appendIfNotDelivered(const std::string& metrics_id, const nlohmann::json& payload) const;
    static nlohmann::json toJson(const GoalCognitiveMetricsRecord& record);

    /**
     * Precedence: THOTH_COGNITIVE_METRICS_LOG > THOTH_LOGS_PATH > {projectRoot}/logs
     */
    static std::string resolveLogFilePath();

private:
    CognitiveMetricsLogger();
    std::string logFilePath() const;
    void appendJsonLine(const nlohmann::json& event) const;
};

} // namespace Thoth

#endif // THOTH_COGNITIVE_METRICS_H
