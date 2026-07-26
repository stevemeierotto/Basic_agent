/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Phase 11 research resource collections (Engine-authored; pure helpers)
 *
 * Contract: Engine owns research knowledge; GUI presents via stable versioned
 * collection resources. Storage layout is an implementation detail.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_RESEARCH_RESOURCES_H
#define THOTH_RESEARCH_RESOURCES_H

#include "json.hpp"

#include <string>

namespace Thoth {
namespace ResearchResources {

inline constexpr int kSchemaVersion = 1;

inline constexpr const char* kHttpPathStrategies = "/v1/research/strategies";
inline constexpr const char* kHttpPathTrajectories = "/v1/research/trajectories";
inline constexpr const char* kHttpPathEpisodes = "/v1/research/episodes";

inline constexpr const char* kReadyCapabilityStrategies = "strategies";
inline constexpr const char* kReadyCapabilityTrajectories = "trajectories";
inline constexpr const char* kReadyCapabilityEpisodes = "episodes";

/** Invalid payload — forces GUI Error (not Empty) after failed remote fetch. */
inline nlohmann::json unavailableFetchResult() {
    return nlohmann::json::object();
}

inline nlohmann::json makeCollection(const nlohmann::json& items,
                                     const nlohmann::json& next_page = nullptr,
                                     const nlohmann::json& total_items = nullptr) {
    nlohmann::json body = nlohmann::json{
        {"schema_version", kSchemaVersion},
        {"items", items.is_array() ? items : nlohmann::json::array()},
        {"next_page", next_page},
    };
    if (total_items.is_null()) {
        body["total_items"] = items.is_array() ? static_cast<int>(items.size()) : 0;
    } else {
        body["total_items"] = total_items;
    }
    return body;
}

inline nlohmann::json emptyCollection() {
    return makeCollection(nlohmann::json::array(), nullptr, 0);
}

inline bool isFetchError(const nlohmann::json& body) {
    return body.is_object() && body.empty();
}

inline bool hasRequiredCollectionFields(const nlohmann::json& body, std::string& error_out) {
    if (!body.is_object()) {
        error_out = "collection must be an object";
        return false;
    }
    if (body.empty()) {
        error_out = "collection fetch failed";
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
    if (!body.contains("items") || !body["items"].is_array()) {
        error_out = "items must be an array";
        return false;
    }
    if (!body.contains("next_page")) {
        error_out = "next_page required";
        return false;
    }
    if (!(body["next_page"].is_null() || body["next_page"].is_string())) {
        error_out = "next_page must be null or opaque string";
        return false;
    }
    if (!body.contains("total_items")) {
        error_out = "total_items required";
        return false;
    }
    if (!(body["total_items"].is_null() || body["total_items"].is_number_integer())) {
        error_out = "total_items must be null or integer";
        return false;
    }
    return true;
}

inline bool hasRequiredStrategyItemFields(const nlohmann::json& item, std::string& error_out) {
    if (!item.is_object()) {
        error_out = "strategy item must be an object";
        return false;
    }
    if (!item.contains("strategy_id") || !item["strategy_id"].is_string()
        || item["strategy_id"].get<std::string>().empty()) {
        error_out = "strategy_id required";
        return false;
    }
    return true;
}

inline bool hasRequiredTrajectoryItemFields(const nlohmann::json& item, std::string& error_out) {
    if (!item.is_object()) {
        error_out = "trajectory item must be an object";
        return false;
    }
    if (!item.contains("trajectory_id") || !item["trajectory_id"].is_string()
        || item["trajectory_id"].get<std::string>().empty()) {
        error_out = "trajectory_id required";
        return false;
    }
    return true;
}

inline bool hasRequiredEpisodeItemFields(const nlohmann::json& item, std::string& error_out) {
    if (!item.is_object()) {
        error_out = "episode item must be an object";
        return false;
    }
    if (!item.contains("episode_id") || !item["episode_id"].is_string()
        || item["episode_id"].get<std::string>().empty()) {
        error_out = "episode_id required";
        return false;
    }
    return true;
}

inline bool isEffectivelyEmpty(const nlohmann::json& body) {
    if (!body.is_object() || !body.contains("items") || !body["items"].is_array()) {
        return true;
    }
    return body["items"].empty();
}

inline nlohmann::json itemsArray(const nlohmann::json& body) {
    if (!body.is_object() || !body.contains("items") || !body["items"].is_array()) {
        return nlohmann::json::array();
    }
    return body["items"];
}

inline bool readyCapabilitiesInclude(const nlohmann::json& body, const char* token) {
    if (!body.contains("capabilities") || !body["capabilities"].is_array()) {
        return false;
    }
    for (const auto& cap : body["capabilities"]) {
        if (cap.is_string() && cap.get<std::string>() == token) {
            return true;
        }
    }
    return false;
}

inline bool readyCapabilitiesIncludeStrategies(const nlohmann::json& body) {
    return readyCapabilitiesInclude(body, kReadyCapabilityStrategies);
}

inline bool readyCapabilitiesIncludeTrajectories(const nlohmann::json& body) {
    return readyCapabilitiesInclude(body, kReadyCapabilityTrajectories);
}

inline bool readyCapabilitiesIncludeEpisodes(const nlohmann::json& body) {
    return readyCapabilitiesInclude(body, kReadyCapabilityEpisodes);
}

} // namespace ResearchResources
} // namespace Thoth

#endif // THOTH_RESEARCH_RESOURCES_H
