/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C6 append-only cognitive metrics log
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/cognitive_metrics.h"
#include "../include/experience_provenance.h"
#include "file_handler.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

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

namespace {

std::mutex& cognitiveMetricsWriteMutex() {
    static std::mutex writeMutex;
    return writeMutex;
}

class ExclusiveMetricsLock {
public:
    explicit ExclusiveMetricsLock(const std::string& logPath) {
        std::error_code ec;
        const fs::path file(logPath);
        if (file.has_parent_path()) {
            fs::create_directories(file.parent_path(), ec);
        }
        fd_ = ::open((logPath + ".lock").c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) return;
        if (::flock(fd_, LOCK_EX) != 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }
    ~ExclusiveMetricsLock() {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
    }
    bool held() const { return fd_ >= 0; }
    ExclusiveMetricsLock(const ExclusiveMetricsLock&) = delete;
    ExclusiveMetricsLock& operator=(const ExclusiveMetricsLock&) = delete;
private:
    int fd_ = -1;
};

bool scanDelivered(const std::string& path, const std::string& metrics_id) {
    if (metrics_id.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        try {
            const auto parsed = nlohmann::json::parse(line);
            if (!parsed.is_object() || !parsed.contains("metrics_id")
                || !parsed["metrics_id"].is_string()) {
                continue;
            }
            if (parsed["metrics_id"].get<std::string>() == metrics_id) return true;
        } catch (...) {
            continue;
        }
    }
    return false;
}

bool appendPayloadLine(const std::string& path, const nlohmann::json& payload) {
    bool needs_separator = false;
    {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            in.seekg(0, std::ios::end);
            const auto size = in.tellg();
            if (size > 0) {
                in.seekg(-1, std::ios::end);
                char last = '\n';
                in.read(&last, 1);
                needs_separator = last != '\n';
            }
        }
    }
    nlohmann::json line = payload;
    line["emitted_at_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();
    std::string bytes;
    if (needs_separator) bytes.push_back('\n');
    bytes += line.dump();
    bytes.push_back('\n');
    const int fd = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd < 0) return false;
    const char* data = bytes.data();
    size_t remaining = bytes.size();
    while (remaining > 0) {
        const ssize_t wrote = ::write(fd, data, remaining);
        if (wrote <= 0) {
            ::close(fd);
            return false;
        }
        data += wrote;
        remaining -= static_cast<size_t>(wrote);
    }
    const bool synced = ::fsync(fd) == 0;
    ::close(fd);
    return synced;
}

void holdAfterScanForTests(const std::string& path) {
    const char* env = std::getenv("THOTH_INJECT_METRICS_FLUSH");
    if (!env || std::string(env) != "hold_after_scan") return;
    {
        std::ofstream marker(path + ".hold");
        marker << "1\n";
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (std::chrono::steady_clock::now() < deadline) {
        if (fs::exists(path + ".release")) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

}  // namespace

void CognitiveMetricsLogger::appendJsonLine(const nlohmann::json& event) const {
    std::lock_guard<std::mutex> lock(cognitiveMetricsWriteMutex());
    ExclusiveMetricsLock fileLock(logFilePath());
    if (!fileLock.held()) return;
    appendPayloadLine(logFilePath(), event);
}

bool CognitiveMetricsLogger::metricsIdDelivered(const std::string& metrics_id) const {
    std::lock_guard<std::mutex> lock(cognitiveMetricsWriteMutex());
    ExclusiveMetricsLock fileLock(logFilePath());
    if (!fileLock.held()) return false;
    return scanDelivered(logFilePath(), metrics_id);
}

bool CognitiveMetricsLogger::appendStoredRecord(const nlohmann::json& payload) const {
    std::lock_guard<std::mutex> lock(cognitiveMetricsWriteMutex());
    ExclusiveMetricsLock fileLock(logFilePath());
    if (!fileLock.held()) return false;
    return appendPayloadLine(logFilePath(), payload);
}

bool CognitiveMetricsLogger::appendIfNotDelivered(const std::string& metrics_id,
                                                  const nlohmann::json& payload) const {
    std::lock_guard<std::mutex> lock(cognitiveMetricsWriteMutex());
    const std::string path = logFilePath();
    ExclusiveMetricsLock fileLock(path);
    if (!fileLock.held()) return false;
    if (scanDelivered(path, metrics_id)) return true;
    holdAfterScanForTests(path);
    return appendPayloadLine(path, payload);
}

nlohmann::json CognitiveMetricsLogger::toJson(const GoalCognitiveMetricsRecord& record) {
    nlohmann::json json = {
        {"event", "GOAL_COGNITIVE_METRICS"},
        {"plan_id", record.plan_id},
        {"session_id", record.session_id},
        {"goal", record.goal},
        {"outcome", record.outcome},
        {"goal_finished_at_ms", record.goal_finished_at_ms},
        {"step_count", record.step_count},
        {"trajectory_score", record.trajectory_score},
        {"final_success_score", record.final_success_score},
        {"reflection_count", record.reflection_count},
        {"revisions_count", record.revisions_count},
        {"max_reflections", record.max_reflections},
        {"reflection_skip_reason", record.reflection_skip_reason},
        {"plan_reused", record.plan_reused},
    };
    if (record.emit_goal_started) {
        json["goal_started_at_ms"] = record.goal_started_at_ms;
    }
    if (record.emit_wall_clock) {
        json["total_wall_clock_ms"] = record.total_wall_clock_ms;
    }
    json["metrics_partial"] = record.metrics_partial;
    if (record.metrics_partial) {
        json["metrics_partial_reason"] = record.metrics_partial_reason;
    }
    if (record.emit_exact_counters) {
        json["planning_time_ms"] = record.planning_time_ms;
        json["retrieval_time_ms"] = record.retrieval_time_ms;
        json["llm_synthesis_time_ms"] = record.llm_synthesis_time_ms;
        json["retrieved_chunk_count"] = record.retrieved_chunk_count;
        json["grag_alpha"] = record.grag_alpha;
        json["grag_routing_mode"] = record.grag_routing_mode;
        json["total_tokens"] = record.total_tokens;
        json["prompt_tokens"] = record.prompt_tokens;
        json["completion_tokens"] = record.completion_tokens;
        json["planning_tokens"] = record.planning_tokens;
        json["synthesis_tokens"] = record.synthesis_tokens;
        json["synthesis_prompt_chars"] = record.synthesis_prompt_chars;
        json["synthesis_context_truncated"] = record.synthesis_context_truncated;
    }
    if (record.emit_available_counters) {
        json["available_counters"] = {
            {"counters_are_exact_totals", false},
            {"total_tokens", record.total_tokens},
            {"prompt_tokens", record.prompt_tokens},
            {"completion_tokens", record.completion_tokens},
            {"planning_tokens", record.planning_tokens},
            {"synthesis_tokens", record.synthesis_tokens},
            {"planning_time_ms", record.planning_time_ms},
            {"retrieval_time_ms", record.retrieval_time_ms},
            {"llm_synthesis_time_ms", record.llm_synthesis_time_ms},
            {"retrieved_chunk_count", record.retrieved_chunk_count},
            {"synthesis_prompt_chars", record.synthesis_prompt_chars}
        };
    }
    json["plan_origin"] = record.plan_origin.empty()
        ? std::string(kPlanOriginUnknown)
        : normalizePlanOrigin(record.plan_origin);
    json["learning_eligible"] = record.learning_eligible;
    if (!record.retrieval_query.empty()) {
        json["retrieval_query"] = record.retrieval_query;
    }
    if (!record.step_ids.empty()) {
        json["step_ids"] = record.step_ids;
    }
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
