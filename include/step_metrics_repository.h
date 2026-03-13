/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — StepMetricsRepository Phase 1.4
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include "json.hpp"

namespace Thoth {

/**
 * @brief Repository for persisting and querying plan step execution metrics.
 */
class StepMetricsRepository {
public:
    struct StepMetricRecord {
        std::string step_id;
        std::string plan_id;
        std::string tool_name;
        int64_t latency_ms;
        int retry_count;
        std::string status;
        int64_t timestamp_ms;
    };

    explicit StepMetricsRepository(const std::string& dbPath);
    ~StepMetricsRepository();

    /**
     * @brief Persists a single step metric record.
     */
    bool storeMetric(const StepMetricRecord& metric);

    /**
     * @brief Calculates the success rate for a specific tool.
     */
    float getToolSuccessRate(const std::string& toolName);

    /**
     * @brief Calculates how often plan revisions followed a specific step type/tool.
     */
    float getPlanRevisionFrequency(const std::string& toolName);

private:
    struct DBHandle;
    std::unique_ptr<DBHandle> db_;
};

} // namespace Thoth
