/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — G1d / G1e trajectory bucket ablation helpers
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_TRAJECTORY_ABLATION_H
#define THOTH_TRAJECTORY_ABLATION_H

#include "benchmark_runner.h"

#include <string>
#include <vector>

namespace Thoth {

/** Protocol v1.0 — see docs/trajectory_ablation_benchmark.md */
constexpr float kTrajectoryAblationNdcgTieEpsilon = 0.001f;
/** Default Arm B/C trajectory weight (methodology production arm). */
constexpr float kTrajectoryAblationDefaultWt = 0.2f;

enum class TrajectoryAblationArm { A, B, C };

enum class G1dDecision {
    PENDING,
    KEEP,
    TUNE,
    DROP,
    CONSTRUCTION_BUG,
};

std::string trajectoryAblationArmLabel(TrajectoryAblationArm arm);
std::string g1dDecisionToString(G1dDecision decision);

/** True for Phase C TUNE micro-run weights only (`0.05` or `0.1`). */
bool isTrajectoryAblationTuneWt(float wt);

/**
 * True for G1e polarity weights (`-0.05`, `-0.10`, `-0.20`, and Phase 3b `-0.30`).
 * See docs/G1E_POLARITY_PROTOCOL.md v1.2.
 */
bool isTrajectoryAblationG1eWt(float wt);

/** True if `--wt` is allowed for G1d TUNE or G1e polarity. */
bool isTrajectoryAblationCliWt(float wt);

/** Filter to TRAJECTORY_DISAMBIGUATES cases only. */
std::vector<BenchmarkCase> filterTrajectoryDisambiguatesCases(
    const std::vector<BenchmarkCase>& cases);

/**
 * Fixed arm configs per docs/trajectory_ablation_benchmark.md v1.0.
 * @param wt_bc Trajectory weight for arms B and C (A always uses 0).
 *              Default 0.2; G1d TUNE / G1e pass scheduled override weights.
 */
BenchmarkConfig trajectoryAblationArmConfig(
    TrajectoryAblationArm arm,
    float wt_bc = kTrajectoryAblationDefaultWt);

/** Winner among A/B/C by nDCG@5; returns A|B|C|TIE. */
std::string computeTrajectoryAblationWinner(float ndcg_a, float ndcg_b, float ndcg_c);

struct TrajectoryAblationSummary {
    int cases_run = 0;
    int a_wins = 0;
    int b_wins = 0;
    int c_wins = 0;
    int ties = 0;
    int b_wins_vs_a = 0;
    int a_wins_vs_b = 0;
    float mean_ndcg_a = 0.0f;
    float mean_ndcg_b = 0.0f;
    float mean_ndcg_c = 0.0f;
    float mean_ndcg_delta_b_vs_a = 0.0f;
    G1dDecision decision = G1dDecision::PENDING;
    std::string decision_rationale;
};

/** Apply locked decision matrix (B vs A for KEEP; C pattern for CONSTRUCTION_BUG). */
TrajectoryAblationSummary computeTrajectoryAblationSummary(
    const std::vector<BenchmarkCase>& cases,
    const BenchmarkResult& result_a,
    const BenchmarkResult& result_b,
    const BenchmarkResult& result_c);

} // namespace Thoth

#endif // THOTH_TRAJECTORY_ABLATION_H
