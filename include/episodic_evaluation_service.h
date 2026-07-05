/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2 episodic evaluation service (Phase C1 façade)
 *
 * Protocol: docs/C_PHASE_PROTOCOL.md § C1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_EPISODIC_EVALUATION_SERVICE_H
#define THOTH_EPISODIC_EVALUATION_SERVICE_H

#include "e2_strict_enforcement.h"
#include "episodic_learning_eval.h"

namespace Thoth {

/**
 * Phase C1 — stateless evaluation service boundary.
 * Façade over Phase B free functions; consolidation deferred until after Phase C.
 * No benchmark, Executive, or harness dependencies.
 */
class IEpisodicEvaluationService {
public:
    virtual ~IEpisodicEvaluationService() = default;

    virtual EpisodicLearningCaseEvaluation evaluateCase(
        const std::string& case_id,
        const EpisodicLearningExpectations& expectations,
        const EpisodicLearningArmObservation& cold,
        const EpisodicLearningArmObservation& warm,
        const E2EvalConfig& config) const = 0;

    virtual void applyCaseResolution(EpisodicLearningCaseEvaluation& eval) const = 0;

    virtual EpisodicLearningSummary summarize(
        const std::vector<EpisodicLearningCaseEvaluation>& case_results,
        const std::vector<EpisodicLearningExpectations>& case_expectations,
        const E2EvalConfig& config) const = 0;

    virtual E2EvaluationFingerprint computeFingerprint(const E2EvalConfig& config) const = 0;

    virtual nlohmann::json scopedEquivalenceSnapshot(
        const EpisodicLearningSummary& summary,
        const nlohmann::json& evaluation_fingerprint,
        const nlohmann::json& e2_eval_config) const = 0;

    virtual bool scopedEquivalenceEqual(const nlohmann::json& a,
                                        const nlohmann::json& b) const = 0;

    virtual int fingerprintMismatchBucket(const nlohmann::json& snapshot_a,
                                          const nlohmann::json& snapshot_b,
                                          const std::string& corpus_hash_a,
                                          const std::string& corpus_hash_b) const = 0;
};

/** Stateless singleton — retains no cross-run state. */
const IEpisodicEvaluationService& episodicEvaluationService();

} // namespace Thoth

#endif // THOTH_EPISODIC_EVALUATION_SERVICE_H
