#include "decision_trace.h"

#include "file_handler.h"
#include "logger.h"

#include <chrono>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

static std::size_t readEnvSizeOrDefault(const char* key, std::size_t fallbackValue) {
    const char* raw = std::getenv(key);
    if (!raw || !*raw) {
        return fallbackValue;
    }

    try {
        const auto parsed = std::stoull(raw);
        return parsed > 0 ? static_cast<std::size_t>(parsed) : fallbackValue;
    } catch (...) {
        return fallbackValue;
    }
}

DecisionTraceLogger::DecisionTraceLogger() {
    FileHandler fileHandler;
    traceFilePath = fileHandler.getAgentWorkspacePath("decision_trace.jsonl");
    maxTraceEntries = readEnvSizeOrDefault("THOTH_TRACE_MAX_ENTRIES", DEFAULT_MAX_TRACE_ENTRIES);
    maxTraceFileBytes = readEnvSizeOrDefault("THOTH_TRACE_MAX_BYTES", DEFAULT_MAX_TRACE_FILE_BYTES);
}

DecisionTrace DecisionTraceLogger::startTrace(const std::string& traceType, std::size_t inputLength) const {
    DecisionTrace trace;
    const std::string contextualRequestId = StructuredLogger::instance().currentRequestId();
    trace.requestId = contextualRequestId.empty() ? generateRequestId() : contextualRequestId;
    trace.traceType = traceType;
    trace.inputLength = inputLength;
    trace.startedAtMs = nowMs();
    return trace;
}

void DecisionTraceLogger::addStage(
    DecisionTrace& trace,
    const std::string& name,
    bool success,
    const std::string& summary,
    const json& metadata) const {
    DecisionStage stage;
    stage.name = name;
    stage.success = success;
    stage.summary = summary;
    stage.metadata = metadata;
    trace.stages.push_back(std::move(stage));
}

void DecisionTraceLogger::finishTrace(DecisionTrace& trace, bool success, const std::string& resultSummary) const {
    trace.success = success;
    trace.resultSummary = resultSummary;
    trace.finishedAtMs = nowMs();
}

void DecisionTraceLogger::writeTrace(const DecisionTrace& trace) const {
    json traceJson;
    traceJson["schema_version"] = TRACE_SCHEMA_VERSION;
    traceJson["emitted_at_ms"] = nowMs();
    traceJson["request_id"] = trace.requestId;
    traceJson["trace_type"] = trace.traceType;
    traceJson["input_length"] = trace.inputLength;
    traceJson["started_at_ms"] = trace.startedAtMs;
    traceJson["finished_at_ms"] = trace.finishedAtMs;
    traceJson["duration_ms"] = (trace.finishedAtMs >= trace.startedAtMs)
        ? (trace.finishedAtMs - trace.startedAtMs)
        : 0;
    traceJson["success"] = trace.success;
    traceJson["result_summary"] = trace.resultSummary;

    traceJson["stages"] = json::array();
    for (const auto& stage : trace.stages) {
        traceJson["stages"].push_back({
            {"name", stage.name},
            {"success", stage.success},
            {"summary", stage.summary},
            {"metadata", stage.metadata}
        });
    }

    std::lock_guard<std::mutex> lock(writeMutex);
    enforceRetentionLocked();

    std::ofstream out(traceFilePath, std::ios::app);
    if (!out.is_open()) {
        return;
    }

    out << traceJson.dump() << "\n";
}

std::int64_t DecisionTraceLogger::nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string DecisionTraceLogger::generateRequestId() const {
    const auto ms = nowMs();
    const auto counter = requestCounter.fetch_add(1);
    return "req-" + std::to_string(ms) + "-" + std::to_string(counter);
}

void DecisionTraceLogger::enforceRetentionLocked() const {
    if (!fs::exists(traceFilePath)) {
        return;
    }

    std::error_code ec;
    const auto currentSize = fs::file_size(traceFilePath, ec);
    if (ec) {
        return;
    }

    std::ifstream in(traceFilePath);
    if (!in.is_open()) {
        return;
    }

    std::deque<std::string> lines;
    std::string line;
    std::size_t totalBytes = 0;

    while (std::getline(in, line)) {
        totalBytes += line.size() + 1;
        lines.push_back(line);
    }

    const bool overEntryLimit = lines.size() >= maxTraceEntries;
    const bool overSizeLimit = static_cast<std::size_t>(currentSize) > maxTraceFileBytes || totalBytes > maxTraceFileBytes;

    if (!overEntryLimit && !overSizeLimit) {
        return;
    }

    while (lines.size() >= maxTraceEntries && !lines.empty()) {
        totalBytes -= (lines.front().size() + 1);
        lines.pop_front();
    }

    while (totalBytes > maxTraceFileBytes && !lines.empty()) {
        totalBytes -= (lines.front().size() + 1);
        lines.pop_front();
    }

    std::ofstream out(traceFilePath, std::ios::trunc);
    if (!out.is_open()) {
        return;
    }

    for (const auto& retained : lines) {
        out << retained << '\n';
    }
}
