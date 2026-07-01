/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — G1d trajectory bucket ablation helpers
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/trajectory_ablation.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace Thoth {

namespace {

bool ndcgWithinEpsilon(float a, float b) {
    return std::fabs(a - b) < kTrajectoryAblationNdcgTieEpsilon;
}

float meanNdcg(const BenchmarkResult& result) {
    return result.mean_ndcg;
}

} // namespace

std::string trajectoryAblationArmLabel(TrajectoryAblationArm arm) {
    switch (arm) {
        case TrajectoryAblationArm::A:
            return "A";
        case TrajectoryAblationArm::B:
            return "B";
        case TrajectoryAblationArm::C:
            return "C";
    }
    return "?";
}

std::string g1dDecisionToString(G1dDecision decision) {
    switch (decision) {
        case G1dDecision::PENDING:
            return "PENDING";
        case G1dDecision::KEEP:
            return "KEEP";
        case G1dDecision::TUNE:
            return "TUNE";
        case G1dDecision::DROP:
            return "DROP";
        case G1dDecision::CONSTRUCTION_BUG:
            return "CONSTRUCTION_BUG";
    }
    return "PENDING";
}

std::vector<BenchmarkCase> filterTrajectoryDisambiguatesCases(
    const std::vector<BenchmarkCase>& cases) {
    std::vector<BenchmarkCase> filtered;
    filtered.reserve(cases.size());
    for (const auto& c : cases) {
        if (c.case_type == "TRAJECTORY_DISAMBIGUATES") {
            filtered.push_back(c);
        }
    }
    return filtered;
}

BenchmarkConfig trajectoryAblationArmConfig(TrajectoryAblationArm arm) {
    BenchmarkConfig cfg;
    cfg.wq = 0.4f;
    cfg.wd = 0.4f;
    cfg.keyword_weight = 0.3f;
    cfg.top_k = 5;
    switch (arm) {
        case TrajectoryAblationArm::A:
            cfg.wt = 0.0f;
            cfg.force_empty_trajectory = false;
            break;
        case TrajectoryAblationArm::B:
            cfg.wt = 0.2f;
            cfg.force_empty_trajectory = false;
            break;
        case TrajectoryAblationArm::C:
            cfg.wt = 0.2f;
            cfg.force_empty_trajectory = true;
            break;
    }
    return cfg;
}

std::string computeTrajectoryAblationWinner(float ndcg_a, float ndcg_b, float ndcg_c) {
    const float scores[3] = {ndcg_a, ndcg_b, ndcg_c};
    const char* labels[3] = {"A", "B", "C"};

    int best_idx = 0;
    for (int i = 1; i < 3; ++i) {
        if (scores[i] > scores[best_idx]) {
            best_idx = i;
        }
    }

    int tie_count = 0;
    for (int i = 0; i < 3; ++i) {
        if (ndcgWithinEpsilon(scores[i], scores[best_idx])) {
            ++tie_count;
        }
    }
    if (tie_count > 1) {
        return "TIE";
    }
    return labels[best_idx];
}

TrajectoryAblationSummary computeTrajectoryAblationSummary(
    const std::vector<BenchmarkCase>& cases,
    const BenchmarkResult& result_a,
    const BenchmarkResult& result_b,
    const BenchmarkResult& result_c) {
    TrajectoryAblationSummary summary;
    summary.cases_run = static_cast<int>(cases.size());

    if (cases.empty()) {
        summary.decision = G1dDecision::PENDING;
        summary.decision_rationale = "no cases";
        return summary;
    }

    summary.mean_ndcg_a = meanNdcg(result_a);
    summary.mean_ndcg_b = meanNdcg(result_b);
    summary.mean_ndcg_c = meanNdcg(result_c);
    summary.mean_ndcg_delta_b_vs_a = summary.mean_ndcg_b - summary.mean_ndcg_a;

    int construction_bug_cases = 0;

    for (std::size_t i = 0; i < cases.size(); ++i) {
        const float ndcg_a = result_a.cases[i].ndcg_at_k;
        const float ndcg_b = result_b.cases[i].ndcg_at_k;
        const float ndcg_c = result_c.cases[i].ndcg_at_k;

        const std::string winner = computeTrajectoryAblationWinner(ndcg_a, ndcg_b, ndcg_c);
        if (winner == "A") {
            ++summary.a_wins;
        } else if (winner == "B") {
            ++summary.b_wins;
        } else if (winner == "C") {
            ++summary.c_wins;
        } else {
            ++summary.ties;
        }

        if (ndcg_b > ndcg_a + kTrajectoryAblationNdcgTieEpsilon) {
            ++summary.b_wins_vs_a;
        } else if (ndcg_a > ndcg_b + kTrajectoryAblationNdcgTieEpsilon) {
            ++summary.a_wins_vs_b;
        }

        if (ndcgWithinEpsilon(ndcg_c, ndcg_a) && ndcg_b + kTrajectoryAblationNdcgTieEpsilon < ndcg_a) {
            ++construction_bug_cases;
        }
    }

    const int decided_ab = summary.b_wins_vs_a + summary.a_wins_vs_b;
    const float b_win_rate =
        decided_ab > 0 ? static_cast<float>(summary.b_wins_vs_a) / static_cast<float>(decided_ab) : 0.0f;

    const int majority = static_cast<int>(cases.size()) / 2 + 1;
    if (construction_bug_cases >= majority) {
        summary.decision = G1dDecision::CONSTRUCTION_BUG;
        summary.decision_rationale =
            "C≈A and B<A on majority (" + std::to_string(construction_bug_cases) + "/" +
            std::to_string(cases.size()) + " cases)";
    } else if (b_win_rate >= 0.60f && summary.mean_ndcg_delta_b_vs_a > 0.0f) {
        summary.decision = G1dDecision::KEEP;
        std::ostringstream oss;
        oss << "B wins " << summary.b_wins_vs_a << "/" << decided_ab << " vs A ("
            << static_cast<int>(b_win_rate * 100.0f) << "%) and mean delta "
            << summary.mean_ndcg_delta_b_vs_a;
        summary.decision_rationale = oss.str();
    } else if (b_win_rate >= 0.40f) {
        summary.decision = G1dDecision::TUNE;
        summary.decision_rationale = "B win rate >= 40% but KEEP dual criteria not met";
    } else {
        summary.decision = G1dDecision::DROP;
        summary.decision_rationale = "B win rate < 40% vs A";
    }

    return summary;
}

} // namespace Thoth
