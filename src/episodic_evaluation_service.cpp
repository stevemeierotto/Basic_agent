/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2 episodic evaluation service (Phase C1 façade)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "episodic_evaluation_service.h"

namespace Thoth {

namespace {

class EpisodicEvaluationService final : public IEpisodicEvaluationService {
public:
    EpisodicLearningCaseEvaluation evaluateCase(
        const std::string& case_id,
        const EpisodicLearningExpectations& expectations,
        const EpisodicLearningArmObservation& cold,
        const EpisodicLearningArmObservation& warm,
        const E2EvalConfig& config) const override {
        return evaluateEpisodicLearningCase(case_id, expectations, cold, warm, config);
    }

    void applyCaseResolution(EpisodicLearningCaseEvaluation& eval) const override {
        applyCaseEvaluationResolution(eval);
    }

    EpisodicLearningSummary summarize(
        const std::vector<EpisodicLearningCaseEvaluation>& case_results,
        const std::vector<EpisodicLearningExpectations>& case_expectations,
        const E2EvalConfig& config) const override {
        return summarizeEpisodicLearning(case_results, case_expectations, config);
    }

    E2EvaluationFingerprint computeFingerprint(const E2EvalConfig& config) const override {
        return computeEvaluationFingerprint(config);
    }

    nlohmann::json scopedEquivalenceSnapshot(
        const EpisodicLearningSummary& summary,
        const nlohmann::json& evaluation_fingerprint,
        const nlohmann::json& e2_eval_config) const override {
        return episodicLearningScopedEquivalenceSnapshot(summary, evaluation_fingerprint,
                                                         e2_eval_config);
    }

    bool scopedEquivalenceEqual(const nlohmann::json& a,
                                const nlohmann::json& b) const override {
        return episodicLearningScopedEquivalenceEqual(a, b);
    }

    int fingerprintMismatchBucket(const nlohmann::json& snapshot_a,
                                  const nlohmann::json& snapshot_b,
                                  const std::string& corpus_hash_a,
                                  const std::string& corpus_hash_b) const override {
        return episodicLearningFingerprintMismatchBucket(snapshot_a, snapshot_b, corpus_hash_a,
                                                         corpus_hash_b);
    }
};

} // namespace

const IEpisodicEvaluationService& episodicEvaluationService() {
    static const EpisodicEvaluationService instance;
    return instance;
}

} // namespace Thoth
