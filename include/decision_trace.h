#pragma once

#include <json.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct DecisionStage {
    std::string name;
    bool success = true;
    std::string summary;
    nlohmann::json metadata = nlohmann::json::object();
};

struct DecisionTrace {
    std::string requestId;
    std::string traceType;
    std::size_t inputLength = 0;
    std::int64_t startedAtMs = 0;
    std::int64_t finishedAtMs = 0;
    bool success = true;
    std::string resultSummary;
    std::vector<DecisionStage> stages;
};

class DecisionTraceLogger {
public:
    DecisionTraceLogger();

    DecisionTrace startTrace(const std::string& traceType, std::size_t inputLength) const;
    void addStage(
        DecisionTrace& trace,
        const std::string& name,
        bool success,
        const std::string& summary,
        const nlohmann::json& metadata = nlohmann::json::object()) const;
    void finishTrace(DecisionTrace& trace, bool success, const std::string& resultSummary) const;
    void writeTrace(const DecisionTrace& trace) const;

private:
    static constexpr const char* TRACE_SCHEMA_VERSION = "1.0";
    static constexpr std::size_t DEFAULT_MAX_TRACE_ENTRIES = 5000;
    static constexpr std::size_t DEFAULT_MAX_TRACE_FILE_BYTES = 5 * 1024 * 1024;

    static std::int64_t nowMs();
    std::string generateRequestId() const;
    void enforceRetentionLocked() const;

    std::string traceFilePath;
    std::size_t maxTraceEntries = DEFAULT_MAX_TRACE_ENTRIES;
    std::size_t maxTraceFileBytes = DEFAULT_MAX_TRACE_FILE_BYTES;
    mutable std::atomic<std::uint64_t> requestCounter{0};
    mutable std::mutex writeMutex;
};
