/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Phase 4 decision summary resource (Engine-authored; pure helpers)
 *
 * Contract: structured JSON with mandatory schema_version. Storage format
 * (JSONL, etc.) is an implementation detail — never part of the GUI API.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_DECISION_SUMMARY_H
#define THOTH_DECISION_SUMMARY_H

#include "json.hpp"

#include <fstream>
#include <sstream>
#include <string>

namespace Thoth {
namespace DecisionSummary {

inline constexpr int kSchemaVersion = 1;

/** Locked HTTP path (resource-oriented — not a log/file path). */
inline constexpr const char* kHttpPath = "/v1/diagnostics/latest-decision";

/** Engine /ready capability token when this resource is served. */
inline constexpr const char* kReadyCapability = "diagnostics";

inline nlohmann::json emptyV1Summary() {
    return nlohmann::json{
        {"schema_version", kSchemaVersion},
        {"session_id", ""},
        {"goal", ""},
        {"executive_summary", ""},
        {"planner_summary", ""},
        {"retrieved_chunks", nlohmann::json::array()},
        {"selected_strategy", nullptr},
        {"execution_time_ms", 0},
    };
}

/**
 * Map one decision_trace.jsonl object → v1 decision summary.
 * Engine-owned presentation model (D13); callers must not invent fields beyond this map.
 */
inline nlohmann::json fromDecisionTraceObject(const nlohmann::json& trace) {
    nlohmann::json out = emptyV1Summary();
    if (!trace.is_object()) {
        return out;
    }

    out["session_id"] = trace.value("session_id", "");
    out["execution_time_ms"] = trace.value("duration_ms", 0);
    out["executive_summary"] = trace.value("result_summary", "");

    std::string goal;
    std::string planner;
    nlohmann::json chunks = nlohmann::json::array();
    nlohmann::json strategy = nullptr;

    if (trace.contains("stages") && trace["stages"].is_array()) {
        for (const auto& stage : trace["stages"]) {
            if (!stage.is_object()) {
                continue;
            }
            const std::string name = stage.value("name", "");
            const std::string summary = stage.value("summary", "");
            std::string name_lower = name;
            for (char& c : name_lower) {
                if (c >= 'A' && c <= 'Z') {
                    c = static_cast<char>(c - 'A' + 'a');
                }
            }
            if (planner.empty()
                && (name_lower.find("plan") != std::string::npos
                    || name_lower.find("planner") != std::string::npos)) {
                planner = summary;
            }
            if (stage.contains("metadata") && stage["metadata"].is_object()) {
                const auto& meta = stage["metadata"];
                if (goal.empty() && meta.contains("goal") && meta["goal"].is_string()) {
                    goal = meta["goal"].get<std::string>();
                }
                if (chunks.empty() && meta.contains("retrieved_chunks")
                    && meta["retrieved_chunks"].is_array()) {
                    chunks = meta["retrieved_chunks"];
                }
                if (strategy.is_null() && meta.contains("selected_strategy")) {
                    strategy = meta["selected_strategy"];
                }
            }
        }
        if (out["executive_summary"].get<std::string>().empty() && !trace["stages"].empty()) {
            const auto& last = trace["stages"].back();
            if (last.is_object()) {
                out["executive_summary"] = last.value("summary", "");
            }
        }
    }

    if (goal.empty()) {
        goal = trace.value("trace_type", "");
    }

    out["goal"] = goal;
    out["planner_summary"] = planner;
    out["retrieved_chunks"] = std::move(chunks);
    out["selected_strategy"] = strategy;
    return out;
}

/** Read last non-empty line of decision_trace.jsonl and map to v1 summary. */
inline nlohmann::json loadLatestFromDecisionTraceFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return emptyV1Summary();
    }
    std::string line;
    std::string last;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            last = line;
        }
    }
    if (last.empty()) {
        return emptyV1Summary();
    }
    try {
        return fromDecisionTraceObject(nlohmann::json::parse(last));
    } catch (...) {
        return emptyV1Summary();
    }
}

inline bool hasRequiredV1Fields(const nlohmann::json& body, std::string& error_out) {
    if (!body.is_object()) {
        error_out = "decision summary must be an object";
        return false;
    }
    if (!body.contains("schema_version") || !body["schema_version"].is_number_integer()) {
        error_out = "schema_version missing or not an integer";
        return false;
    }
    if (body["schema_version"].get<int>() < 1) {
        error_out = "schema_version must be >= 1";
        return false;
    }
    static const char* kRequired[] = {
        "session_id", "goal", "executive_summary", "planner_summary",
        "retrieved_chunks", "execution_time_ms"};
    for (const char* key : kRequired) {
        if (!body.contains(key)) {
            error_out = std::string("missing field: ") + key;
            return false;
        }
    }
    if (!body["retrieved_chunks"].is_array()) {
        error_out = "retrieved_chunks must be an array";
        return false;
    }
    return true;
}

/** Render Engine-authored fields for a dialog — labels only; no invented content (D12). */
inline std::string formatForDisplay(const nlohmann::json& summary) {
    if (!summary.is_object()) {
        return "Unavailable\n\nNo decision summary available.";
    }
    std::ostringstream out;
    out << "Goal: " << summary.value("goal", "") << "\n\n";
    out << "Executive summary:\n"
        << summary.value("executive_summary", "") << "\n\n";
    out << "Planner summary:\n"
        << summary.value("planner_summary", "") << "\n\n";
    if (summary.contains("selected_strategy") && !summary["selected_strategy"].is_null()) {
        out << "Selected strategy: " << summary["selected_strategy"].dump() << "\n\n";
    }
    if (summary.contains("retrieved_chunks") && summary["retrieved_chunks"].is_array()
        && !summary["retrieved_chunks"].empty()) {
        out << "Retrieved chunks: " << summary["retrieved_chunks"].size() << "\n\n";
    }
    out << "Execution time (ms): " << summary.value("execution_time_ms", 0) << "\n";
    if (summary.contains("schema_version")) {
        out << "schema_version: " << summary["schema_version"].dump() << "\n";
    }
    return out.str();
}

inline bool isEffectivelyEmpty(const nlohmann::json& summary) {
    if (!summary.is_object()) {
        return true;
    }
    return summary.value("goal", "").empty()
        && summary.value("executive_summary", "").empty()
        && summary.value("planner_summary", "").empty();
}

} // namespace DecisionSummary
} // namespace Thoth

#endif // THOTH_DECISION_SUMMARY_H
