/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — StepMetricsRepository Implementation Phase 1.4
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/step_metrics_repository.h"
#include <sqlite3.h>
#include <iostream>

namespace Thoth {

struct StepMetricsRepository::DBHandle {
    sqlite3* handle = nullptr;
    ~DBHandle() { if (handle) sqlite3_close(handle); }
};

StepMetricsRepository::StepMetricsRepository(const std::string& dbPath)
    : db_(std::make_unique<DBHandle>()) {
    
    int rc = sqlite3_open(dbPath.c_str(), &db_->handle);
    if (rc != SQLITE_OK) {
        std::cerr << "[StepMetricsRepository] Failed to open database: " << sqlite3_errmsg(db_->handle) << "\n";
        return;
    }

    const char* schema = 
        "CREATE TABLE IF NOT EXISTS step_metrics ("
        "  step_id TEXT PRIMARY KEY,"
        "  plan_id TEXT NOT NULL,"
        "  tool_name TEXT,"
        "  latency_ms INTEGER NOT NULL,"
        "  retry_count INTEGER NOT NULL,"
        "  status TEXT NOT NULL,"
        "  timestamp_ms INTEGER NOT NULL"
        ");";

    char* errMsg = nullptr;
    rc = sqlite3_exec(db_->handle, schema, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "[StepMetricsRepository] Schema creation failed: " << (errMsg ? errMsg : "Unknown error") << "\n";
        sqlite3_free(errMsg);
    }
}

StepMetricsRepository::~StepMetricsRepository() = default;

bool StepMetricsRepository::storeMetric(const StepMetricRecord& metric) {
    const char* sql = "INSERT OR REPLACE INTO step_metrics (step_id, plan_id, tool_name, latency_ms, retry_count, status, timestamp_ms) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?);";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
    
    sqlite3_bind_text(stmt, 1, metric.step_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, metric.plan_id.c_str(), -1, SQLITE_STATIC);
    if (metric.tool_name.empty()) {
        sqlite3_bind_null(stmt, 3);
    } else {
        sqlite3_bind_text(stmt, 3, metric.tool_name.c_str(), -1, SQLITE_STATIC);
    }
    sqlite3_bind_int64(stmt, 4, metric.latency_ms);
    sqlite3_bind_int(stmt, 5, metric.retry_count);
    sqlite3_bind_text(stmt, 6, metric.status.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 7, metric.timestamp_ms);

    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    return success;
}

float StepMetricsRepository::getToolSuccessRate(const std::string& toolName) {
    const char* sql = "SELECT COUNT(*) FROM step_metrics WHERE tool_name = ?;";
    const char* sqlSuccess = "SELECT COUNT(*) FROM step_metrics WHERE tool_name = ? AND status = 'success';";
    
    auto getCount = [&](const char* query) -> int {
        sqlite3_stmt* stmt;
        int count = 0;
        if (sqlite3_prepare_v2(db_->handle, query, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, toolName.c_str(), -1, SQLITE_STATIC);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                count = sqlite3_column_int(stmt, 0);
            }
            sqlite3_finalize(stmt);
        }
        return count;
    };

    int total = getCount(sql);
    if (total == 0) return 0.0f;
    int success = getCount(sqlSuccess);
    return static_cast<float>(success) / static_cast<float>(total);
}

float StepMetricsRepository::getPlanRevisionFrequency(const std::string& toolName) {
    // This is a stub for now as it requires analyzing plan history vs step metrics.
    (void)toolName;
    return 0.0f;
}

} // namespace Thoth
