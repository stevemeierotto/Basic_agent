/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Planner prompt memory budgets (C1)
 *
 * Core instructions are never truncated; experience competes for the remainder.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_PLANNER_INJECTION_CONFIG_H
#define THOTH_PLANNER_INJECTION_CONFIG_H

#include <cstddef>

namespace Thoth {

namespace PlannerInjection {

/** Guaranteed (never truncated) section budgets. */
inline constexpr std::size_t kMaxRulesChars = 4096;
inline constexpr std::size_t kMaxSchemaChars = 2048;
inline constexpr std::size_t kMaxGoalChars = 1024;

/** Optional experience slots — dropped entirely when over total budget. */
inline constexpr std::size_t kMaxPlanReuseChars = 1024;
inline constexpr std::size_t kMaxStrategyContextChars = 512;
inline constexpr std::size_t kMaxTrajectoryChars = 512;

/** Minimum total planner prompt budget (chars). */
inline constexpr std::size_t kMinPlanPromptBudget = 8192;

/** Minimum cosine similarity to inject a promoted strategy (description embedding). */
inline constexpr float kMinStrategySimilarity = 0.40f;

/** Max trajectories injected into planner (planner channel). */
inline constexpr int kMaxTrajectoryInject = 1;

} // namespace PlannerInjection

struct PlannerPromptMetrics {
    std::size_t rules_bytes = 0;
    std::size_t schema_bytes = 0;
    std::size_t goal_bytes = 0;
    std::size_t strategy_bytes = 0;
    std::size_t trajectory_bytes = 0;
    std::size_t plan_reuse_bytes = 0;
    std::size_t total_bytes = 0;
    bool experience_dropped = false;
};

} // namespace Thoth

#endif // THOTH_PLANNER_INJECTION_CONFIG_H
