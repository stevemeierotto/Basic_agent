/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2-C3 evaluation diagnostic service (presentation only)
 *
 * Protocol: docs/C_PHASE_PROTOCOL.md § E2-C3
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_DIAGNOSTIC_SERVICE_H
#define THOTH_DIAGNOSTIC_SERVICE_H

#include "episodic_learning_eval.h"

namespace Thoth {

/** E2-C3 failure taxonomy — presentation only; not part of evaluation contract. */
enum class E2DiagnosticFailureClassification {
    NONE,
    CONFIG_MISMATCH,
    CORPUS_DRIFT,
    RETRIEVAL_NONDETERMINISM,
    SEMANTIC_DRIFT,
};

std::string e2DiagnosticFailureClassificationToString(E2DiagnosticFailureClassification c);

/**
 * Run/config attribution only — must NOT carry evaluation results or derived fields.
 * See C_PHASE_PROTOCOL.md § E2-C3 C3.1 ownership rule.
 */
struct EvaluationDiagnosticsContext {
    std::string run_id;
    std::string env_hash;
    std::string fingerprint_hash;
    nlohmann::json e2_eval_config = nlohmann::json::object();
};

/** Per-case or run-level structured diagnostic — downstream of evaluation artifacts. */
struct EvaluationDiagnostics {
    std::string run_id;
    std::string case_id;
    std::optional<E2EvaluationResolution> evaluation_resolution_snapshot;
    int diagnosis_bucket = 0;
    E2DiagnosticFailureClassification failure_classification =
        E2DiagnosticFailureClassification::NONE;
    std::string structured_explanation;
    std::vector<std::string> trace_links;
};

struct EvaluationDiagnosticsSummary {
    std::string run_id;
    std::optional<E2EvaluationResolution> evaluation_resolution_snapshot;
    int diagnosis_bucket = 0;
    E2DiagnosticFailureClassification failure_classification =
        E2DiagnosticFailureClassification::NONE;
    std::string structured_explanation;
    std::vector<EvaluationDiagnostics> case_diagnostics;
};

/**
 * Phase C3 — stateless diagnostic boundary.
 * Consumes exported evaluation artifacts only; never invokes evaluation or scoring.
 */
class IDiagnosticService {
public:
    virtual ~IDiagnosticService() = default;

    virtual EvaluationDiagnostics generateDiagnostics(
        const EpisodicLearningCaseEvaluation& eval,
        const E2EvalConfig& config,
        const EvaluationDiagnosticsContext& context) const = 0;

    virtual EvaluationDiagnosticsSummary generateRunDiagnostics(
        const EpisodicLearningSummary& summary,
        const EvaluationDiagnosticsContext& context) const = 0;
};

const IDiagnosticService& episodicDiagnosticService();

nlohmann::json evaluationDiagnosticToJson(const EvaluationDiagnostics& diagnostic);
nlohmann::json evaluationDiagnosticSummaryToJson(const EvaluationDiagnosticsSummary& summary);

} // namespace Thoth

#endif // THOTH_DIAGNOSTIC_SERVICE_H
