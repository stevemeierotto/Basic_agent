/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — GRAG Phase 7 Metrics
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/grag_metrics.h"
#include "../include/file_handler.h"
#include <iostream>

namespace Thoth {

GragMetricsLogger& GragMetricsLogger::instance() {
    static GragMetricsLogger inst;
    return inst;
}

void GragMetricsLogger::logMetrics(const GragMetrics& metrics) {
    std::lock_guard<std::mutex> lock(mtx);
    
    nlohmann::json j;
    j["goal_id"] = metrics.goal_id;
    j["duration_ms"] = metrics.end_time_ms - metrics.start_time_ms;
    j["revision_count"] = metrics.revision_count;
    j["total_steps"] = metrics.total_steps;
    j["successful_steps"] = metrics.successful_steps;
    j["failed_steps"] = metrics.failed_steps;
    j["plan_reused"] = metrics.plan_reused;
    j["final_success_score"] = metrics.final_success_score;
    j["status"] = metrics.status;
    j["timestamp_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    FileHandler fh;
    std::string path = fh.getAgentWorkspacePath("grag_metrics.jsonl");
    
    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << j.dump() << "\n";
    }
}

} // namespace Thoth
