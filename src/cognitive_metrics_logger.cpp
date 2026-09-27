/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C6 append-only cognitive metrics log
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/cognitive_metrics.h"
#include "file_handler.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>

namespace fs = std::filesystem;

namespace Thoth {

CognitiveMetricsLogger& CognitiveMetricsLogger::instance() {
    static CognitiveMetricsLogger logger;
    return logger;
}

CognitiveMetricsLogger::CognitiveMetricsLogger() = default;

std::string CognitiveMetricsLogger::resolveLogFilePath() {
    if (const char* overridePath = std::getenv("THOTH_COGNITIVE_METRICS_LOG")) {
        if (*overridePath) {
            return overridePath;
        }
    }
    FileHandler fh;
    return fh.getLogsPath("cognitive_metrics.jsonl");
}

std::string CognitiveMetricsLogger::logFilePath() const {
    return resolveLogFilePath();
}

void CognitiveMetricsLogger::appendJsonLine(const nlohmann::json& event) const {
    static std::mutex writeMutex;
    std::lock_guard<std::mutex> lock(writeMutex);

    nlohmann::json line = event;
    line["emitted_at_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();

    std::ofstream out(logFilePath(), std::ios::app);
    if (out.is_open()) {
        out << line.dump() << '\n';
    }
}

nlohmann::json CognitiveMetricsLogger::toJson(const GoalCognitiveMetricsRecord& record) {
    nlohmann::json json = {
        {"event", "GOAL_COGNITIVE_METRICS"},
        {"plan_id", record.plan_id},
        {"session_id", record.session_id},
        {"goal", record.goal},
        {"outcome", record.outcome},
        {"goal_started_at_ms", record.goal_started_at_ms},
        {"goal_finished_at_ms", record.goal_finished_at_ms},
        {"total_wall_clock_ms", record.total_wall_clock_ms},
        {"planning_time_ms", record.planning_time_ms},
        {"retrieval_time_ms", record.retrieval_time_ms},
        {"llm_synthesis_time_ms", record.llm_synthesis_time_ms},
        {"step_count", record.step_count},
        {"retrieved_chunk_count", record.retrieved_chunk_count},
        {"grag_alpha", record.grag_alpha},
        {"grag_routing_mode", record.grag_routing_mode},
        {"trajectory_score", record.trajectory_score},
        {"final_success_score", record.final_success_score},
        {"reflection_count", record.reflection_count},
        {"revisions_count", record.revisions_count},
        {"max_reflections", record.max_reflections},
        {"reflection_skip_reason", record.reflection_skip_reason},
        {"plan_reused", record.plan_reused},
        {"total_tokens", record.total_tokens},
        {"prompt_tokens", record.prompt_tokens},
        {"completion_tokens", record.completion_tokens},
        {"planning_tokens", record.planning_tokens},
        {"synthesis_tokens", record.synthesis_tokens},
        {"synthesis_prompt_chars", record.synthesis_prompt_chars},
        {"synthesis_context_truncated", record.synthesis_context_truncated},
    };
    if (!record.run_id.empty()) {
        json["run_id"] = record.run_id;
    }
    if (!record.env_hash.empty()) {
        json["env_hash"] = record.env_hash;
    }
    if (!record.c64_window_id.empty()) {
        json["window_id"] = record.c64_window_id;
        json["protocol_version"] = record.c64_protocol_version;
        json["metric_schema_version"] = record.c64_metric_schema_version;
        json["environment_schema_version"] = record.c64_environment_schema_version;
        json["c64_cohort_fingerprint"] = record.c64_cohort_fingerprint;
    }
    return json;
}

void CognitiveMetricsLogger::logGoalMetrics(const GoalCognitiveMetricsRecord& record) const {
    appendJsonLine(toJson(record));
}

} // namespace Thoth
