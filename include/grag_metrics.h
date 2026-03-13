/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — GRAG Phase 7 Metrics
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <string>
#include <vector>
#include <json.hpp>
#include <mutex>
#include <fstream>
#include <chrono>

namespace Thoth {

struct GragMetrics {
    std::string goal_id;
    int64_t start_time_ms = 0;
    int64_t end_time_ms = 0;
    int revision_count = 0;
    int total_steps = 0;
    int successful_steps = 0;
    int failed_steps = 0;
    bool plan_reused = false;
    float final_success_score = 0.0f;
    std::string status; // "COMPLETED", "FAILED", "ABORTED"
};

class GragMetricsLogger {
public:
    static GragMetricsLogger& instance();

    void logMetrics(const GragMetrics& metrics);

private:
    GragMetricsLogger() = default;
    std::mutex mtx;
};

} // namespace Thoth
