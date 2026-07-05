/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2-C5 path equivalence harness (proof only — not production API)
 *
 * Protocol: docs/C_PHASE_PROTOCOL.md § E2-C5
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_E2_PATH_EQUIVALENCE_H
#define THOTH_E2_PATH_EQUIVALENCE_H

#include "diagnostic_service.h"
#include "e2_strict_enforcement.h"
#include "episode_events.h"
#include "episodic_learning_cases.h"
#include "episodic_learning_eval.h"

#include <optional>
#include <string>
#include <vector>

namespace Thoth {

/** Single-case artifacts from one orchestration path (eval + diagnostics). */
struct E2PathEquivalenceArtifacts {
    EpisodicLearningCaseEvaluation case_eval;
    EpisodicLearningSummary summary;
    EvaluationDiagnosticsSummary diagnostics;
    E2EvaluationFingerprint fingerprint;
};

/** Result of comparing benchmark vs production path artifacts. */
struct E2PathEquivalenceDiff {
    bool equivalent = true;
    std::vector<std::string> mismatches;
};

/**
 * Checkpoint 0 — synthetic EpisodeCompleted for mapping fidelity tests only.
 * NOT production functionality.
 */
EpisodeCompleted episodeFromBenchmarkArmsForTests(
    const EpisodicLearningCase& spec,
    const EpisodicLearningArmObservation& cold,
    const EpisodicLearningArmObservation& warm,
    const EpisodicLearningExpectations& expectations,
    E2RunBlockReason run_block_reason = E2RunBlockReason::NONE);

/** Compare evaluation-relevant observation fields (arms + expectations). */
bool armsEvaluationRelevantEqual(const EpisodicLearningArmObservation& a,
                                 const EpisodicLearningArmObservation& b,
                                 std::string* mismatch_out = nullptr);

bool expectationsEvaluationRelevantEqual(const EpisodicLearningExpectations& a,
                                         const EpisodicLearningExpectations& b,
                                         std::string* mismatch_out = nullptr);

/**
 * Checkpoint 0 gate — benchmark arms → synthetic episode → mapper → compare to benchmark arms.
 */
bool validateMappingFidelityForCase(const EpisodicLearningCase& spec,
                                    const EpisodicLearningArmObservation& cold,
                                    const EpisodicLearningArmObservation& warm,
                                    const EpisodicLearningExpectations& expectations,
                                    std::string* report_out = nullptr);

/** Benchmark orchestration kernel path (direct service evaluation). */
E2PathEquivalenceArtifacts runBenchmarkPathArtifacts(
    const std::string& case_id,
    const EpisodicLearningExpectations& expectations,
    const EpisodicLearningArmObservation& cold,
    const EpisodicLearningArmObservation& warm,
    const E2EvalConfig& config,
    E2RunBlockReason run_block_reason = E2RunBlockReason::NONE);

/** Production orchestration path (subscriber + mapper) under pinned config. */
E2PathEquivalenceArtifacts runProductionPathArtifacts(const EpisodeCompleted& event,
                                                      const E2EvalConfig& pinned_config);

E2PathEquivalenceDiff diffPathEquivalence(const E2PathEquivalenceArtifacts& benchmark,
                                          const E2PathEquivalenceArtifacts& production);

nlohmann::json pathEquivalenceCaseEvalSnapshot(const EpisodicLearningCaseEvaluation& eval);
nlohmann::json pathEquivalenceSummarySnapshot(const EpisodicLearningSummary& summary);
nlohmann::json pathEquivalenceDiagnosticsSnapshot(const EvaluationDiagnosticsSummary& diagnostics);

} // namespace Thoth

#endif // THOTH_E2_PATH_EQUIVALENCE_H
