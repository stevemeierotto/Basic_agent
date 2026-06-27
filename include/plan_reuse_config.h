/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Tunable constants for plan history reuse and related cognition paths
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_PLAN_REUSE_CONFIG_H
#define THOTH_PLAN_REUSE_CONFIG_H

#include <cstddef>

namespace Thoth {

/**
 * @brief Plan history reuse (past_plans table / retrieveSimilarPlans).
 * See docs/plan_reuse_tuning.md for tuning guidance.
 */
namespace PlanReuse {
    /** Minimum success_score for a stored plan to be eligible for injection. */
    inline constexpr float kMinSuccessScore = 0.6f;
    /** success_score at or above this receives a ranking boost. */
    inline constexpr float kSuccessBoostThreshold = 0.8f;
    /** Added to cosine similarity when success_score >= kSuccessBoostThreshold. */
    inline constexpr float kSuccessBoost = 0.1f;
    /** Default top-K returned by retrieveSimilarPlans (planner gets one similar plan). */
    inline constexpr int kDefaultRetrieveLimit = 1;
    /** Minimum cosine similarity to inject a past plan (below this: inject nothing). */
    inline constexpr float kMinSimilarityFloor = 0.55f;
    /** Max characters of sanitized plan outline injected into planner context. */
    inline constexpr std::size_t kOutlineMaxChars = 500;
}

/**
 * @brief Trajectory injection into planner prompts (separate from GRAG wt).
 */
namespace PlannerTrajectory {
    /** Minimum cosine similarity to inject a trajectory snippet. */
    inline constexpr float kMinSimilarityFloor = 0.55f;
}

/**
 * @brief Trajectory embedding for GRAG (TrajectoryBuilder + ExecutiveController wt gating).
 */
namespace TrajectoryReuse {
    /** Episode steps required before T embedding is non-zero (TrajectoryBuilder). */
    inline constexpr int kMinEpisodeStepsForEmbedding = 3;
    /** ExecutiveController forces wt=0 when T is all zeros regardless of retrieval_config. */
    inline constexpr float kZeroVectorEpsilon = 1e-6f;
}

/**
 * @brief Reflection replan loop (ExecutiveController).
 */
namespace Reflection {
    /** Trajectory score below this triggers reflection replan. */
    inline constexpr float kScoreThreshold = 0.6f;
    /** Maximum reflection cycles per goal (ExecutiveController::MAX_REFLECTIONS). */
    inline constexpr int kMaxReflections = 2;
}

} // namespace Thoth

#endif // THOTH_PLAN_REUSE_CONFIG_H
