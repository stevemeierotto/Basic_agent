/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Phase 12A graph statistics singleton resource (Engine-authored; pure helpers)
 *
 * Contract: Engine owns graph state and statistics; GUI presents via stable API.
 * Storage layout is an implementation detail — never part of the GUI API.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_GRAPH_STATISTICS_H
#define THOTH_GRAPH_STATISTICS_H

#include "json.hpp"

#include <cstdint>
#include <string>

namespace Thoth {
namespace GraphStatistics {

inline constexpr int kSchemaVersion = 1;

inline constexpr const char* kHttpPath = "/v1/graph/stats";

/** Engine /ready capability token when graph statistics resource is served. */
inline constexpr const char* kReadyCapability = "graph_stats";

/** Invalid payload — forces GUI Error (not Empty) after failed remote fetch. */
inline nlohmann::json unavailableFetchResult() {
    return nlohmann::json::object();
}

inline nlohmann::json makeStatisticsPayload(int total_nodes,
                                            int total_edges,
                                            float avg_edge_weight,
                                            float max_edge_weight,
                                            float min_edge_weight,
                                            int total_success_count,
                                            int total_failure_count) {
    return nlohmann::json{
        {"total_nodes", total_nodes},
        {"total_edges", total_edges},
        {"avg_edge_weight", avg_edge_weight},
        {"max_edge_weight", max_edge_weight},
        {"min_edge_weight", min_edge_weight},
        {"total_success_count", total_success_count},
        {"total_failure_count", total_failure_count},
    };
}

inline nlohmann::json makeResponse(const nlohmann::json& statistics,
                                   std::int64_t generated_at_ms,
                                   const nlohmann::json& session_id = nullptr) {
    nlohmann::json body = nlohmann::json{
        {"schema_version", kSchemaVersion},
        {"generated_at", generated_at_ms},
        {"statistics", statistics.is_object() ? statistics : nlohmann::json::object()},
    };
    if (session_id.is_null()) {
        body["session_id"] = nullptr;
    } else {
        body["session_id"] = session_id;
    }
    return body;
}

inline nlohmann::json emptyResponse(std::int64_t generated_at_ms,
                                    const nlohmann::json& session_id = nullptr) {
    return makeResponse(makeStatisticsPayload(0, 0, 0.0f, 0.0f, 0.0f, 0, 0),
                        generated_at_ms,
                        session_id);
}

inline bool isFetchError(const nlohmann::json& body) {
    return body.is_object() && body.empty();
}

inline bool hasRequiredFields(const nlohmann::json& body, std::string& error_out) {
    if (!body.is_object()) {
        error_out = "graph statistics must be an object";
        return false;
    }
    if (body.empty()) {
        error_out = "graph statistics fetch failed";
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
    if (!body.contains("generated_at") || !body["generated_at"].is_number_integer()) {
        error_out = "generated_at missing or not an integer";
        return false;
    }
    if (!body.contains("statistics") || !body["statistics"].is_object()) {
        error_out = "statistics must be an object";
        return false;
    }
    if (body.contains("session_id")
        && !(body["session_id"].is_null() || body["session_id"].is_string())) {
        error_out = "session_id must be null or string";
        return false;
    }
    const auto& stats = body["statistics"];
    static const char* kRequired[] = {"total_nodes",
                                      "total_edges",
                                      "avg_edge_weight",
                                      "max_edge_weight",
                                      "min_edge_weight",
                                      "total_success_count",
                                      "total_failure_count"};
    for (const char* key : kRequired) {
        if (!stats.contains(key) || !stats[key].is_number()) {
            error_out = std::string("statistics missing numeric field: ") + key;
            return false;
        }
    }
    return true;
}

/** Empty = valid resource with no meaningful graph data (Phase 12A lock). */
inline bool isEffectivelyEmpty(const nlohmann::json& body) {
    std::string err;
    if (!hasRequiredFields(body, err)) {
        return false;
    }
    const auto& stats = body["statistics"];
    const int nodes = stats["total_nodes"].get<int>();
    const int edges = stats["total_edges"].get<int>();
    return nodes == 0 && edges == 0;
}

inline nlohmann::json statisticsPayload(const nlohmann::json& body) {
    if (!body.is_object() || !body.contains("statistics") || !body["statistics"].is_object()) {
        return nlohmann::json::object();
    }
    return body["statistics"];
}

inline bool readyCapabilitiesIncludeGraphStats(const nlohmann::json& body) {
    if (!body.contains("capabilities") || !body["capabilities"].is_array()) {
        return false;
    }
    for (const auto& cap : body["capabilities"]) {
        if (cap.is_string() && cap.get<std::string>() == kReadyCapability) {
            return true;
        }
    }
    return false;
}

} // namespace GraphStatistics
} // namespace Thoth

#endif // THOTH_GRAPH_STATISTICS_H
