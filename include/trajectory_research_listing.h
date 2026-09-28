/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — Research trajectory collection for the Trajectories panel.
 * Presentation/query only. Does not score, rank, or select trajectories
 * for injection, plan reuse, strategy, or consolidation.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_TRAJECTORY_RESEARCH_LISTING_H
#define THOTH_TRAJECTORY_RESEARCH_LISTING_H

#include "memory_repository.h"
#include "research_resources.h"

#include <algorithm>
#include <vector>

namespace Thoth {

/**
 * Full research payload for persisted trajectories.
 * Every row is included. Age is not a reason to drop a terminal score.
 */
inline nlohmann::json listTrajectoryResearchCollection(
    std::vector<MemoryRepository::CognateTrajectoryRecord> trajs) {
    std::sort(trajs.begin(), trajs.end(), [](const auto& a, const auto& b) {
        if (a.created_at != b.created_at) {
            return a.created_at > b.created_at;
        }
        return a.trajectory_id < b.trajectory_id;
    });

    nlohmann::json items = nlohmann::json::array();
    for (const auto& t : trajs) {
        nlohmann::json trajectory = nlohmann::json::object();
        try {
            trajectory = nlohmann::json::parse(t.trajectory_json);
        } catch (...) {
        }
        items.push_back({
            {"trajectory_id", t.trajectory_id},
            {"goal", t.goal},
            {"trajectory", trajectory},
            {"success_score", t.success_score},
            {"created_at", t.created_at},
            {"usage_count", t.usage_count},
            {"tier", t.tier},
        });
    }
    return ResearchResources::makeCollection(items);
}

} // namespace Thoth

#endif // THOTH_TRAJECTORY_RESEARCH_LISTING_H
