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

std::string CognitiveMetricsLogger::logFilePath() const {
    FileHandler fh;
    fs::path logsDir = fs::path(fh.getProjectRoot()) / "logs";
    fs::create_directories(logsDir);
    return (logsDir / "cognitive_metrics.jsonl").string();
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
    return {
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
    };
}

void CognitiveMetricsLogger::logGoalMetrics(const GoalCognitiveMetricsRecord& record) const {
    appendJsonLine(toJson(record));
}

} // namespace Thoth
