/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2-C3 EvaluationDiagnosticService
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "diagnostic_service.h"

#include <sstream>

namespace Thoth {

namespace {

std::string armStatusLabel(E2ArmScoringStatus status) {
    return e2ArmScoringStatusToString(status);
}

int diagnosisBucketFromCase(const EpisodicLearningCaseEvaluation& eval) {
    if (eval.run_block_reason == E2RunBlockReason::WIRING_GATE ||
        eval.run_block_reason == E2RunBlockReason::STRICT_BOUNDARY_VIOLATION) {
        return 1;
    }
    if (eval.run_block_reason == E2RunBlockReason::PROVENANCE_VIOLATION) {
        return 2;
    }
    if (eval.run_block_reason == E2RunBlockReason::RUNTIME_HEURISTIC_GUARD) {
        return 3;
    }
    if (!eval.evaluation_resolution.has_value()) {
        return 0;
    }
    if (*eval.evaluation_resolution == E2EvaluationResolution::SCORED_SUCCESS && eval.passes) {
        return 0;
    }
    return 4;
}

E2DiagnosticFailureClassification failureClassificationFromCase(
    const EpisodicLearningCaseEvaluation& eval) {
    switch (eval.run_block_reason) {
    case E2RunBlockReason::WIRING_GATE:
    case E2RunBlockReason::STRICT_BOUNDARY_VIOLATION:
        return E2DiagnosticFailureClassification::CONFIG_MISMATCH;
    case E2RunBlockReason::PROVENANCE_VIOLATION:
        return E2DiagnosticFailureClassification::CORPUS_DRIFT;
    case E2RunBlockReason::RUNTIME_HEURISTIC_GUARD:
        return E2DiagnosticFailureClassification::RETRIEVAL_NONDETERMINISM;
    default:
        break;
    }
    if (eval.evaluation_resolution.has_value() &&
        *eval.evaluation_resolution == E2EvaluationResolution::SCORED_FAILURE) {
        return E2DiagnosticFailureClassification::SEMANTIC_DRIFT;
    }
    if (eval.evaluation_resolution.has_value() &&
        *eval.evaluation_resolution == E2EvaluationResolution::SCORED_SUCCESS && !eval.passes) {
        return E2DiagnosticFailureClassification::SEMANTIC_DRIFT;
    }
    return E2DiagnosticFailureClassification::NONE;
}

std::string buildCaseExplanation(const EpisodicLearningCaseEvaluation& eval,
                                 const E2EvalConfig& config) {
    std::ostringstream oss;
    oss << "case=" << eval.case_id;
    if (eval.evaluation_resolution.has_value()) {
        oss << ";resolution=" << e2EvaluationResolutionToString(*eval.evaluation_resolution);
    }
    oss << ";block=" << e2RunBlockReasonToString(eval.run_block_reason);
    const E2ArmScoringStatus arm_status = caseArmStatusForResolution(eval.cold, eval.warm);
    oss << ";arm=" << armStatusLabel(arm_status);
    oss << ";lift=" << eval.lift << ";passes=" << (eval.passes ? "true" : "false");
    if (!eval.failure_reason.empty()) {
        oss << ";failure_reason=" << eval.failure_reason;
    }
    oss << ";tier=" << e2EvalTierToString(config.tier);
    return oss.str();
}

int worstDiagnosisBucket(const std::vector<EvaluationDiagnostics>& cases) {
    int worst = 0;
    for (const auto& item : cases) {
        worst = std::max(worst, item.diagnosis_bucket);
    }
    return worst;
}

E2DiagnosticFailureClassification worstFailureClassification(
    const std::vector<EvaluationDiagnostics>& cases) {
    auto rank = [](E2DiagnosticFailureClassification c) {
        switch (c) {
        case E2DiagnosticFailureClassification::NONE:
            return 0;
        case E2DiagnosticFailureClassification::CONFIG_MISMATCH:
            return 1;
        case E2DiagnosticFailureClassification::CORPUS_DRIFT:
            return 2;
        case E2DiagnosticFailureClassification::RETRIEVAL_NONDETERMINISM:
            return 3;
        case E2DiagnosticFailureClassification::SEMANTIC_DRIFT:
            return 4;
        }
        return 0;
    };
    E2DiagnosticFailureClassification worst = E2DiagnosticFailureClassification::NONE;
    for (const auto& item : cases) {
        if (rank(item.failure_classification) > rank(worst)) {
            worst = item.failure_classification;
        }
    }
    return worst;
}

class EvaluationDiagnosticService final : public IDiagnosticService {
public:
    EvaluationDiagnostics generateDiagnostics(
        const EpisodicLearningCaseEvaluation& eval,
        const E2EvalConfig& config,
        const EvaluationDiagnosticsContext& context) const override {
        EvaluationDiagnostics diagnostic;
        diagnostic.run_id = context.run_id;
        diagnostic.case_id = eval.case_id;
        diagnostic.evaluation_resolution_snapshot = eval.evaluation_resolution;
        diagnostic.diagnosis_bucket = diagnosisBucketFromCase(eval);
        diagnostic.failure_classification = failureClassificationFromCase(eval);
        diagnostic.structured_explanation = buildCaseExplanation(eval, config);
        if (!context.fingerprint_hash.empty()) {
            diagnostic.trace_links.push_back("fingerprint:" + context.fingerprint_hash);
        }
        if (!context.env_hash.empty()) {
            diagnostic.trace_links.push_back("env:" + context.env_hash);
        }
        return diagnostic;
    }

    EvaluationDiagnosticsSummary generateRunDiagnostics(
        const EpisodicLearningSummary& summary,
        const EvaluationDiagnosticsContext& context) const override {
        EvaluationDiagnosticsSummary runDiag;
        runDiag.run_id = context.run_id;
        runDiag.evaluation_resolution_snapshot = summary.evaluation_resolution;

        E2EvalConfig config;
        config.tier = summary.scoring_tier;
        if (context.e2_eval_config.is_object()) {
            if (context.e2_eval_config.contains("tier") &&
                context.e2_eval_config["tier"].is_string()) {
                const std::string tier = context.e2_eval_config["tier"].get<std::string>();
                if (tier == "INTEGRATION") {
                    config.tier = E2EvalTier::INTEGRATION;
                } else if (tier == "STRICT") {
                    config.tier = E2EvalTier::STRICT;
                }
            }
        }

        runDiag.case_diagnostics.reserve(summary.case_results.size());
        for (const auto& eval : summary.case_results) {
            runDiag.case_diagnostics.push_back(generateDiagnostics(eval, config, context));
        }

        runDiag.diagnosis_bucket = worstDiagnosisBucket(runDiag.case_diagnostics);
        runDiag.failure_classification = worstFailureClassification(runDiag.case_diagnostics);

        std::ostringstream oss;
        oss << "run=" << context.run_id << ";cases=" << summary.case_results.size()
            << ";scorable=" << summary.scorable_cases
            << ";not_scorable=" << summary.not_scorable_cases
            << ";mean_lift=" << summary.mean_episodic_lift;
        if (summary.evaluation_resolution.has_value()) {
            oss << ";resolution="
                << e2EvaluationResolutionToString(*summary.evaluation_resolution);
        }
        if (runDiag.failure_classification != E2DiagnosticFailureClassification::NONE) {
            oss << ";classification="
                << e2DiagnosticFailureClassificationToString(runDiag.failure_classification);
        }
        runDiag.structured_explanation = oss.str();
        return runDiag;
    }
};

const EvaluationDiagnosticService& diagnosticServiceInstance() {
    static const EvaluationDiagnosticService instance;
    return instance;
}

} // namespace

std::string e2DiagnosticFailureClassificationToString(E2DiagnosticFailureClassification c) {
    switch (c) {
    case E2DiagnosticFailureClassification::NONE:
        return "none";
    case E2DiagnosticFailureClassification::CONFIG_MISMATCH:
        return "config_mismatch";
    case E2DiagnosticFailureClassification::CORPUS_DRIFT:
        return "corpus_drift";
    case E2DiagnosticFailureClassification::RETRIEVAL_NONDETERMINISM:
        return "retrieval_nondeterminism";
    case E2DiagnosticFailureClassification::SEMANTIC_DRIFT:
        return "semantic_drift";
    }
    return "none";
}

const IDiagnosticService& episodicDiagnosticService() {
    return diagnosticServiceInstance();
}

nlohmann::json evaluationDiagnosticToJson(const EvaluationDiagnostics& diagnostic) {
    nlohmann::json j = {{"event_type", "E2_EVAL_DIAGNOSTIC_CASE"},
                        {"run_id", diagnostic.run_id},
                        {"diagnosis_bucket", diagnostic.diagnosis_bucket},
                        {"failure_classification",
                         e2DiagnosticFailureClassificationToString(diagnostic.failure_classification)},
                        {"structured_explanation", diagnostic.structured_explanation}};
    if (!diagnostic.case_id.empty()) {
        j["case_id"] = diagnostic.case_id;
    }
    if (diagnostic.evaluation_resolution_snapshot.has_value()) {
        j["evaluation_resolution_snapshot"] =
            e2EvaluationResolutionToString(*diagnostic.evaluation_resolution_snapshot);
    }
    if (!diagnostic.trace_links.empty()) {
        j["trace_links"] = diagnostic.trace_links;
    }
    return j;
}

nlohmann::json evaluationDiagnosticSummaryToJson(const EvaluationDiagnosticsSummary& summary) {
    nlohmann::json cases = nlohmann::json::array();
    for (const auto& item : summary.case_diagnostics) {
        cases.push_back(evaluationDiagnosticToJson(item));
    }
    nlohmann::json j = {{"event_type", "E2_EVAL_DIAGNOSTIC_SUMMARY"},
                        {"run_id", summary.run_id},
                        {"diagnosis_bucket", summary.diagnosis_bucket},
                        {"failure_classification",
                         e2DiagnosticFailureClassificationToString(summary.failure_classification)},
                        {"structured_explanation", summary.structured_explanation},
                        {"case_diagnostics", cases}};
    if (summary.evaluation_resolution_snapshot.has_value()) {
        j["evaluation_resolution_snapshot"] =
            e2EvaluationResolutionToString(*summary.evaluation_resolution_snapshot);
    }
    return j;
}

} // namespace Thoth
